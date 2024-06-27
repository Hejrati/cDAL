import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from PIL import Image

from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess_dataset.MoNu import MonuDataset


def calculate_metrics(x, gt):
    predict = x.detach().cpu().numpy().astype('uint8')
    target = gt.detach().cpu().numpy().astype('uint8')
    return f1_score(target.flatten(), predict.flatten()), jaccard_score(target.flatten(), predict.flatten()),

def set_random_seed_for_iterations(seed):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def sampling_major_vote_func(pos_coeff, sample_from_model, netG, output_folder, dataset, logger, step, args, device):
    batch_size = 1
    if isinstance(dataset, MonuDataset):
        major_vote_number = 5
    else:
        major_vote_number = 30
    loader = DataLoader(dataset, batch_size=batch_size)
    loader_iter = iter(loader)
    n_rounds = len(loader_iter)

    f1_score_list = []
    miou_list = []

    with torch.no_grad():
        for _ in tqdm(range(n_rounds), desc="Generating image samples for Dice and mIoU evaluation."):
            gt_mask, condition_on, name = next(loader_iter)
            set_random_seed_for_iterations(major_vote_number)
            gt_mask = (gt_mask + 1.0) / 2.0
            condition_on = condition_on["conditioned_image"]
            former_frame_for_feature_extraction = condition_on.to(device)
            gt_mask = gt_mask.to(device)

            for i in range(gt_mask.shape[0]):
                gt_img = Image.fromarray((gt_mask[i][0].detach().cpu().numpy() - 1).astype(np.uint8))
                gt_img.save(os.path.join(output_folder, f"{name[i]}_gt.png"))

            if isinstance(dataset, MonuDataset):
                _, _, W, H = former_frame_for_feature_extraction.shape
                kernel_size = dataset.image_size
                stride = 256
                patches = []

                for y, x in np.ndindex((((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1)):
                    y = y * stride
                    x = x * stride
                    patches.append(former_frame_for_feature_extraction[0,
                                   :,
                                   y: min(y + kernel_size, W),
                                   x: min(x + kernel_size, H)])
                patches = torch.stack(patches)

                major_vote_list = []
                for i in range(major_vote_number):
                    x_list = []

                    for index in range(math.ceil(patches.shape[0] / 4)):
                        model_kwargs = {"conditioned_image": patches[index * 4: min((index + 1) * 4, patches.shape[0])]}
                        x_t_1 = torch.randn_like(
                            torch.zeros(model_kwargs["conditioned_image"].shape[0], gt_mask.shape[1],
                                        model_kwargs["conditioned_image"].shape[2],
                                        model_kwargs["conditioned_image"].shape[3])).to(device)
                        y_cond = model_kwargs["conditioned_image"]

                        x = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, y_cond, args)

                        x_list.append(x)
                    out = torch.cat(x_list)

                    output = torch.zeros((former_frame_for_feature_extraction.shape[0], gt_mask.shape[1],
                                          former_frame_for_feature_extraction.shape[2],
                                          former_frame_for_feature_extraction.shape[3]))
                    idx_sum = torch.zeros((former_frame_for_feature_extraction.shape[0], gt_mask.shape[1],
                                           former_frame_for_feature_extraction.shape[2],
                                           former_frame_for_feature_extraction.shape[3]))
                    for index, val in enumerate(out):
                        y, x = np.unravel_index(index,
                                                (((W - kernel_size) // stride) + 1, ((H - kernel_size) // stride) + 1))
                        y = y * stride
                        x = x * stride

                        idx_sum[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)] += 1

                        output[0,
                        :,
                        y: min(y + kernel_size, W),
                        x: min(x + kernel_size, H)] += val[:, :min(y + kernel_size, W) - y,
                                                       :min(x + kernel_size, H) - x].cpu().data.numpy()

                    output = output / idx_sum
                    major_vote_list.append(output)

                x = torch.cat(major_vote_list)

            else:
                model_kwargs = {
                    "conditioned_image": torch.cat([former_frame_for_feature_extraction] * major_vote_number)}

                x_t_1 = torch.randn_like(
                    torch.zeros(major_vote_number, gt_mask.shape[1],
                                model_kwargs["conditioned_image"].shape[2],
                                model_kwargs["conditioned_image"].shape[3])).to(device)
                y_cond = model_kwargs["conditioned_image"]

                x = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, y_cond, args)

            x = (x + 1.0) / 2.0
            if x.shape[2] != gt_mask.shape[2] or x.shape[3] != gt_mask.shape[3]:
                x = F.interpolate(x, gt_mask.shape[2:], mode='bilinear')

            x = torch.clamp(x, 0.0, 1.0)

            x = x.mean(dim=0, keepdim=True).round()

            for i in range(x.shape[0]):
                # save as outer training ids
                out_img = Image.fromarray((x[i][0].detach().cpu().numpy() - 1).astype(np.uint8))
                out_img.save(os.path.join(output_folder, f"{name[i]}_model_output.png"))

            for index, (gt_im, out_im) in enumerate(zip(gt_mask, x)):
                ### We need to ensure that the background is 0 and the foreground (object) is 1
                if isinstance(dataset, MonuDataset):
                    f1, miou = calculate_metrics(out_im[0], gt_im[0])
                else:
                    f1, miou = calculate_metrics(-out_im[0] + 1, -gt_mask[0].squeeze() + 1)

                f1_score_list.append(f1)
                miou_list.append(miou)

                logger.info(f"{name[index]} iou {miou_list[-1]}, f1_Score {f1_score_list[-1]}")

    my_length = len(miou_list)
    length_of_data = torch.tensor(len(miou_list), device=device)
    gathered_length_of_data = [torch.tensor(1, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_length_of_data, length_of_data)
    max_len = torch.max(torch.stack(gathered_length_of_data))

    iou_tensor = torch.tensor(miou_list + [torch.tensor(-1)] * (max_len - my_length), device=device)
    f1_tensor = torch.tensor(f1_score_list + [torch.tensor(-1)] * (max_len - my_length), device=device)

    gathered_miou = [torch.ones_like(iou_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_f1 = [torch.ones_like(f1_tensor) * -1 for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_miou, iou_tensor)
    dist.all_gather(gathered_f1, f1_tensor)

    logger.info("measure total avg")
    gathered_miou = torch.cat(gathered_miou)
    gathered_miou = gathered_miou[gathered_miou != -1]
    logger.info(f"mean iou {gathered_miou.mean()}")

    gathered_f1 = torch.cat(gathered_f1)
    gathered_f1 = gathered_f1[gathered_f1 != -1]
    logger.info(f"mean f1 {gathered_f1.mean()}")

    dist.barrier()
    return gathered_miou.mean().item(), gathered_f1.mean().item()
