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
from preprocess_dataset.hippo import HippocampusDecathlonDataModule
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from train_cDal_monu_and_lung import Posterior_Coefficients, sample_from_model
from utils import set_random_seed_for_iterations, dev, broadcast_params

from metrics_hippo import sampling_major_vote_func


def main():
    args = create_argparser().parse_args()

    device = dev(args.local_rank)

    if args.dataset == 'hippo':
        data_module = HippocampusDecathlonDataModule(root_dir="/home/share/Data/")
        data_module.setup("fit")
        dataset_test = data_module.val_dataloader()
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
        model_path="./saved_info/",
        model_name="fold0_mean_0.8735198974609375_Post_0.880092203617096_Ant_0.8669475317001343.pth",
        output_folder="output",
        seed=10,
        resume=False,
        image_size=32,
        num_channels=3,
        num_channels_disc=6,
        centered=True,
        use_geometric=False,
        beta_min=0.1,
        beta_max=20.,

        cond_enc_layers=3,
        cond_enc_num_res_blocks=2,

        num_channels_dae=32,
        n_mlp=3,
        ch_mult=(1, 2, 2, 2,),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.,
        resamp_with_conv=True,  # False in ddpm, True in biggan
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type='biggan',  # 'type of resnet block, choice in biggan and ddpm'
        progressive='none',  # choices=['none', 'output_skip', 'residual']
        progressive_input='residual',  # choices=['none', 'input_skip', 'residual']
        progressive_combine='sum',  # choices=['sum', 'cat']

        embedding_type='positional',
        fourier_scale=16.,
        not_use_tanh=False,

        # geenrator and training
        exp='biggann_',
        dataset='hippo',
        nz=50,
        num_timesteps=2,
        attn_scale=2,

        z_emb_dim=128,
        t_emb_dim=128,
        batch_size=32 * 16,
        num_epoch=12000,
        ngf=32,

        lr_g=1.6e-4,
        lr_d=1.25e-4,
        beta1=0.5,
        beta2=0.9,
        no_lr_decay=False,

        use_ema=True,
        ema_decay=0.999,

        r1_gamma=1.0,
        lazy_reg=10,

        save_content=False,  # to save all models and data
        save_content_every=5,
        save_ckpt_every=5,  # to save the model each checkpoint

        log_step=20,
        ###ddp
        num_proc_node=1,
        num_process_per_node=1,
        node_rank=1,
        local_rank=0,
        master_address='127.0.0.1',
        master_port='6020'
    )

    parameters_path = f"parameters_{defaults['dataset']}.json"
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
