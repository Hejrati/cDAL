"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import json
import os
from pathlib import Path

import torch.distributed
from torch import nn
import torch.distributed as dist

from preprocess_dataset.dataset import create_dataset

import logger
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from train_cDal_monu_and_lung import Posterior_Coefficients, sample_from_model
from utils import set_random_seed_for_iterations, dev, broadcast_params

from metrics import sampling_major_vote_func


def main():
    args = create_argparser().parse_args()

    device = dev(args.local_rank)

    if args.dataset == 'monu':
        dataset_test = create_dataset(data_dir="/home/share/Data/Medical", mode='val', image_size=256,
                                      dataset_name='monu')
    elif args.dataset == 'lung':
        dataset_test = create_dataset(data_dir="/home/share/Data/Lung", mode='test', image_size=256,
                                      dataset_name='lung')

    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=args.num_process_per_node)

    checkpoint = torch.load(args.model_path, map_location=device)
    netG = NCSNpp(args).to(device)
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    broadcast_params(netG.parameters())
    netG.load_state_dict(checkpoint, strict=True)
    netG.eval()
    pos_coeff = Posterior_Coefficients(args, device)

    if args.__dict__.get("seed") is None:
        seed = 1234
    else:
        seed = int(args.__dict__.get("seed"))

    set_random_seed_for_iterations(seed)
    logger.log("sampling major vote val")

    sampling_major_vote_func(pos_coeff, sample_from_model, netG, args.output_folder, dataset_test,
                             logger, None, args, device)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        model_path="./saved_info/dd_gan/lung/experiment_lung_cDAL_fold0/",
        model_name="netG_0_mIoU_0.8486591200665663_F1_0.9160836870891729.pth",
        output_folder="output",
    )

    parameters_path = f"parameters_{defaults['dataset']}.json"
    if os.path.exists(parameters_path):
        with open('parameters.json', 'r') as f:
            defaults.update(json.load(f))
    output_dir = os.path.join(defaults["model_path"], defaults["output_folder"])
    defaults.update(model_path=os.path.join(defaults["model_path"], defaults["model_name"]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    defaults.update(output_folder=output_dir)
    parser = argparse.ArgumentParser()
    logger.add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
