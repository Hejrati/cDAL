# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license,  01see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import json

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist

from metrics import sampling_major_vote_func
from utils import *
from score_sde.models.discriminator import Discriminator_large
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from EMA import EMA

from preprocess_dataset.dataset import create_dataset
import logger


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
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


def sample_from_model(coefficients, generator, n_time, x_init, y, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, y, latent_z)
            # x_0 = generator(x, t_time, y)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x


# %%
def train(rank, gpu, args):

    set_random_seed_for_iterations(args.seed)
    device = dev(gpu)

    exp = args.exp
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/', os.path.join(exp_path, 'score_sde/'))

    logger.configure(dir=str(exp_path))
    logger.info('device: {}'.format(device))

    batch_size = args.batch_size

    nz = args.nz  # latent dimension

    if args.dataset == 'monu':
        train_data = create_dataset(data_dir="/home/share/Data/Medical", mode='train', image_size=256, dataset_name='monu')
        test_data = create_dataset(data_dir="/home/share/Data/Medical", mode='test', image_size=256, dataset_name='monu')
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    elif args.dataset == 'lung':
        train_data = create_dataset(data_dir="/home/share/Data/Lung", mode='train', image_size=256, dataset_name='lung')
        test_data = create_dataset(data_dir="/home/share/Data/Lung", mode='test', image_size=256, dataset_name='lung')
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=True)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    netG = NCSNpp(args).to(device)
    netD = Discriminator_large(args.num_channels_disc, ngf=args.ngf,
                               t_emb_dim=args.t_emb_dim,
                               act=nn.LeakyReLU(0.2)).to(device)

    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.T_max, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.T_max, eta_min=1e-5)

    # ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    netD = CustomDDPWrapper(netD, device_ids=[gpu])

    netD.apply(weights_init_normal)

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G

        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, init_epoch = 0, 0

    # Beginning of training
    terms = dict()
    best_mIoU, errD_total, mse_total = 0, 0, 0

    for epoch in range(init_epoch, args.num_epoch + 1):

        for _, pairs in enumerate(train_loader):

            x = pairs[0]  # This is 'x' (label)
            cond_img = pairs[1]["conditioned_image"]  # This is 'I' (image)

            for p in netD.parameters():
                p.requires_grad = True

            netD.zero_grad()

            # sample from p(x_0)
            real_data = x.to(device, non_blocking=True)

            # sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # I- train Discriminator with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)

            errD_real = F.softplus(-D_real)
            terms["errD_real"] = errD_real
            errD_real = errD_real.mean()

            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True
                )[0]
                grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_t, create_graph=True
                    )[0]
                    grad_penalty = (
                            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # II- train Discriminator with fake
            latent_z = torch.randn(batch_size, nz, device=device)

            # calculate attention w.r.t real data
            x_0_predict = netG(x_tp1.detach(), t, cond_img, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

            errD_fake = F.softplus(output)
            terms["errD_fake"] = errD_fake
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake

            # Update D
            optimizerD.step()

            # III- train Generator
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

            latent_z = torch.randn(batch_size, nz, device=device)
            # get attention
            hs = netD.get_features(x_t, t, x_tp1.detach())
            # print(hs[args.attn_scale].size())
            attn = F.interpolate(hs[args.attn_scale].mean(dim=1).unsqueeze(1), real_data.size()[-1], mode='bilinear')
            att_real_data = attn * real_data

            _, x_tp1_attn = q_sample_pairs(coeff, att_real_data, t)
            x_0_predict = netG(x_tp1_attn.detach(), t, cond_img, latent_z)

            mse = mean_flat((real_data - x_0_predict) ** 2)

            terms["mse"] = mse

            mse.mean().backward()
            optimizerG.step()
            global_step += 1

            mse_total += mse
            errD_total += errD.item()

            if global_step % args.log_step == 0:
                terms["total_errD"] = errD_total / global_step
                terms["total_mse"] = mse_total / global_step
                logger.log_loss_dict(terms)
                logger.log_loss_dict(global_step)
                logger.dumpkvs()
                logger.log('epoch: {}, step: {}'.format(epoch, global_step))

        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()

        if rank == 0:
            if epoch % args.save_ckpt_every == 0:
                sample_path = os.path.join(exp_path, 'major_sampling_epoch_{}'.format(epoch))
                if not os.path.exists(sample_path):
                    os.makedirs(sample_path)
                mIoU, f1 = sampling_major_vote_func(pos_coeff, sample_from_model, netG, sample_path, test_data,
                                                    logger, epoch, args, device)
                if mIoU > best_mIoU:
                    best_mIoU = mIoU

                    if args.use_ema:
                        optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

                    torch.save(netG.state_dict(), os.path.join(exp_path, f'netG_{epoch}_mIoU_{mIoU}_F1_{f1}.pth'))

            if args.save_content and epoch % args.save_content_every == 0:
                print('Saving content...')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                           'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                           'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                           'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                torch.save(content, os.path.join(exp_path, 'content.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    defaults = dict(
        seed=47,
        resume=False,
        image_size=256,
        num_channels=1,
        num_channels_disc=2,
        centered=True,
        use_geometric=False,
        beta_min=0.1,
        beta_max=20.,

        cond_enc_layers=3,
        cond_enc_num_res_blocks=2,

        num_channels_dae=32,
        n_mlp=3,
        ch_mult=(1, 1, 2, 2, 4, 4),
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.,
        resamp_with_conv=True,  # False in ddpm, True in biggan
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type='biggan',  # choice in biggan and ddpm
        progressive='none',
        progressive_input='residual',
        progressive_combine='sum',
        attn_scale=2,

        embedding_type='positional',
        fourier_scale=16.,
        not_use_tanh=False,

        # geenrator and training
        exp='cDAL',
        dataset='lung',
        fold=0,
        nz=100,
        num_timesteps=2,

        z_emb_dim=256,
        t_emb_dim=256,
        batch_size=3,
        num_epoch=12000,
        T_max=500,
        ngf=64,

        lr_g=2e-4,
        lr_d=1e-5,
        beta1=0.5,
        beta2=0.9,
        no_lr_decay=False,

        use_ema=False,
        ema_decay=0.9999,
        ###ddp

        r1_gamma=1.,
        lazy_reg=None,

        save_content=False,  # to save all models and data
        save_content_every=2,
        save_ckpt_every=2,  # to save the model each checkpoint
        log_step=10,
        num_proc_node=1,
        num_process_per_node=1,
        node_rank=0,
        local_rank=1,
        master_address='127.0.0.1',
        master_port="6021",

    )

    defaults.update(exp=f"experiment_attn{defaults.get('attn_scale')}_{defaults.get('dataset')}_{defaults.get('exp')}_fold{defaults.get('fold')}")
    parameters_path = f"parameters_{defaults['dataset']}.json"
    if os.path.exists(parameters_path):
        with open(parameters_path, 'r') as f:
            defaults = json.load(f)
    else:
        with open(parameters_path, 'w') as f:
            json.dump(defaults, f, indent=4)

    logger.add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=args.num_process_per_node)
    train(0, gpu, args)
    dist.barrier()
    dist.destroy_process_group()
