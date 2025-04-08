import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from PIL import Image
import random

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import Compose, EnsureType, AsDiscrete
from monai.visualize import plot_2d_or_3d_image
from monai.visualize.img2tensorboard import SummaryWriter
from sklearn.metrics import f1_score, jaccard_score
from tqdm import tqdm



def set_random_seed_for_iterations(seed):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def dev(gpu):
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(gpu)
    return torch.device("cpu")


def calculate_metrics(x, gt):
    predict = x.detach().cpu().numpy().astype('uint8')
    target = gt.detach().cpu().numpy().astype('uint8')
    return f1_score(target.flatten(), predict.flatten()), jaccard_score(target.flatten(), predict.flatten()),
    # WCov_metric(predict, target), FBound_metric(predict, target)


def print_metric(metric_name, scores, logger):
    scores = np.mean(scores, axis=0)
    agg_score = np.mean(scores)

    logger.log("Validation {} score average: {:4f}".format(metric_name, agg_score))
    for i, score in enumerate(scores):
        print("Validation {} score class {}: {:4f}".format(metric_name, i + 1, score))
        logger.log("Validation {} score class {}: {:4f}".format(metric_name, i + 1, score))


def sampling_major_vote_func(pos_coeff, sample_from_model, netG, output_folder, dataset, logger, step, args, device):
    # ddp_model.eval()
    # batch_size = 1
    major_vote_number = 5
    # loader = DataLoader(dataset, batch_size=batch_size)
    # loader_iter = iter(loader)
    n_rounds = len(dataset)

    f1_score_list = []
    miou_list = []
    fbound_list = []
    wcov_list = []

    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)
    precision_metric = ConfusionMatrixMetric(
        include_background=False, metric_name="precision", compute_sample=True, reduction="none", get_not_nans=False
    )
    sensitivity_metric = ConfusionMatrixMetric(
        include_background=False, metric_name="sensitivity", compute_sample=True, reduction="none", get_not_nans=False
    )

    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=3)])
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=3)])
    experiment = SummaryWriter(output_folder)

    with torch.no_grad():
        # set_random_seed_for_iterations(step)

        for i, data in enumerate(tqdm(dataset, total=n_rounds, desc="Major vote sampling")):
            labels = data["label"].to(dev(device))
            condition_on = data['image'].to(dev(device))
            prediction = Predictor(pos_coeff, netG, args.num_timesteps, args, dev(device), major_vote_number)

            val_outputs = sliding_window_inference(
                condition_on,
                roi_size=[32, 32, 32],
                sw_batch_size=4,
                predictor=prediction.forward,
                overlap=0.75,
            )

            plot_2d_or_3d_image(condition_on, step=i, writer=experiment, max_channels=1,
                                tag=f"Input Image_{i}", )
            plot_2d_or_3d_image(labels * 20, step=i, writer=experiment, tag=f"Label_{i}")
            plot_2d_or_3d_image(torch.argmax(val_outputs, dim=1, keepdim=True) * 20, step=i, writer=experiment,
                                tag=f"Prediction{i}", )

            val_outputs = [post_pred(val_output) for val_output in decollate_batch(val_outputs)]
            labels = [post_label(label) for label in decollate_batch(labels)]

            # for index, (gt_im, out_im) in enumerate(zip(gt_mask, x)):
            # f1, miou = calculate_metrics(-out_im[0] + 1, -gt_mask[0].squeeze() + 1)
            dice = dice_metric(y_pred=val_outputs, y=labels)
            ant_dice = dice[0][0]
            post_dice = dice[0][1]

            ant_dice_list.append(ant_dice)
            post_dice_list.append(post_dice)
            logger.info(f"{i} Post: {post_dice_list[-1]:.4f}, Ant: {ant_dice_list[-1]:.4f}")

            precision_metric(y_pred=val_outputs, y=labels)
            sensitivity_metric(y_pred=val_outputs, y=labels)

    print_metric("dice", dice_metric.aggregate().cpu().numpy(), logger)

    print_metric("precision", precision_metric.aggregate()[0].cpu().numpy(), logger)

    print_metric("sensitivity", sensitivity_metric.aggregate()[0].cpu().numpy(), logger)

    scores = np.mean(dice_metric.aggregate().cpu().numpy(), axis=0)

    dist.barrier()
    return scores[0], scores[1]


class Predictor:
    def __init__(self, coefficients, generator, n_time, args, device, major_vote_number=5):
        self.coefficients = coefficients
        self.generator = generator
        self.n_time = n_time
        self.args = args
        self.major_vote_number = major_vote_number
        self.device = device

    def forward(self, image):
        # condition_on = [b, 1, 32, 32, 32]
        condition_on = image.permute(0, 4, 1, 2, 3).reshape(-1, 1, 32, 32).to(self.device)  # image
        condition_on_copies = []

        # Loop 5 times to create 5 copies
        for _ in range(self.major_vote_number):
            # Create a copy of the condition_on tensor
            tensor_copy = condition_on.clone()

            # Add the copy to the list
            condition_on_copies.append(tensor_copy)

        # Stack the copies together
        stacked_condition_on = torch.stack(condition_on_copies).reshape(-1, 1, 32, 32)

        condition_on = 2 * stacked_condition_on - 1
        x_t_1 = torch.randn_like(torch.zeros(condition_on.size(0), 3, image.shape[2], image.shape[3])).to(
            self.device)  # label

        y_cond = condition_on
        x = x_t_1
        # print(f'size of y_cond: {y_cond.size()}')
        with torch.no_grad():
            for i in reversed(range(self.n_time)):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

                t_time = t
                latent_z = torch.randn(x.size(0), self.args.nz, device=x.device)
                x_0 = self.generator(x, t_time, y_cond, latent_z)
                x_new = sample_posterior(self.coefficients, x_0, x, t)
                x = x_new.detach()

        x = x.resize(self.major_vote_number, image.size(0), image.shape[2], 3, image.shape[2], image.shape[3])
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        x = x.mean(0).round()
        x = x.resize(image.size(0), image.shape[2], 3, image.shape[2], image.shape[3]).permute(0, 2, 3, 4, 1)
        return x


def sample_posterior(coefficients, x_0, x_t, t):
    def extract(input, t, shape):
        out = torch.gather(input, 0, t)
        reshape = [shape[0]] + [1] * (len(shape) - 1)
        out = out.reshape(*reshape)

        return out

    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos
