"""
Various utilities for neural networks.
"""
import abc
import math

import torch as th
import torch.nn as nn
import torch
from typing import Any, Optional, Type


class CrossAttentionEncoder(th.nn.Module, abc.ABC):
    """
    An attention encoder that uses cross attention to encode the input image and the condition image

    * extends: torch.nn.Module
    * abstract class

    * methods to implement:
        * `forward`: Accept the input image as a `torch.Tensor` and the condition image as a `torch.Tensor` and return the encoded image as a `torch.Tensor`
    """

    def __call__(self, x: th.Tensor, hs: Optional[tuple[th.Tensor, ...]] = None) -> th.Tensor:
        return super().__call__(x, hs)

    @abc.abstractmethod
    def forward(self, x: th.Tensor, hs: Optional[tuple[th.Tensor, ...]] = None) -> th.Tensor:
        return NotImplemented

class AttentionedFFParserConditionalEncoder(CrossAttentionEncoder):
    """
    Encoder that uses attention and FFParser to encode the input image and the condition image
    Code editted from `Generic_UNet` in https://github.com/WuJunde/MedSegDiff/blob/master/guided_diffusion/unet.py#L720
    Removed redundant decoder part

    * extends: CrossAttentionEncoder
    """

    def __init__(self, input_channels: int, base_num_features: int, num_pool: int, num_conv_per_stage: int = 2, feat_map_mul_on_downscale: int = 2, convolutional_pooling=False):
        super().__init__()
        self.convolutional_pooling = convolutional_pooling
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.conv_blocks_context = th.nn.ModuleList([])
        self.conv_trans_blocks_a = th.nn.ModuleList([])
        self.conv_trans_blocks_b = th.nn.ModuleList([])
        self.conv_blocks_localization = th.nn.ModuleList([])
        self.td = th.nn.ModuleList([])
        self.ffparser = th.nn.ModuleList([])

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = 2
            else:
                first_stride = None

            conv_kwargs['kernel_size'] = 3
            conv_kwargs['padding'] = 1
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage, conv_kwargs=conv_kwargs, norm_op_kwargs=norm_op_kwargs, dropout_op_kwargs=dropout_op_kwargs, nonlin_kwargs=nonlin_kwargs, first_stride=first_stride))
            if d < num_pool - 1:
                self.conv_trans_blocks_a.append(th.nn.Conv2d(int(d/2 + 1) * 128, 2 **(d+5), 1))
                self.conv_trans_blocks_b.append(th.nn.Conv2d(2 **(d+5), 1, 1))
            if d != num_pool - 1:
                self.ffparser.append(FFParser(output_features, 256 // (2 **(d+1)), 256 // (2 **(d+2))+1))

            if not self.convolutional_pooling:
                self.td.append(th.nn.MaxPool2d((2, 2)))
            input_features = output_features
            output_features = int(round(output_features * feat_map_mul_on_downscale))

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = 2
        else:
            first_stride = None

        conv_kwargs['kernel_size'] = 3
        conv_kwargs['padding'] = 1
        self.conv_blocks_context.append(th.nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, conv_kwargs=conv_kwargs, norm_op_kwargs=norm_op_kwargs, dropout_op_kwargs=dropout_op_kwargs, nonlin_kwargs=nonlin_kwargs, first_stride=first_stride),
            StackedConvLayers(output_features, output_features, 1, conv_kwargs=conv_kwargs, norm_op_kwargs=norm_op_kwargs, dropout_op_kwargs=dropout_op_kwargs, nonlin_kwargs=nonlin_kwargs)))

    def forward(self, x: th.Tensor, hs: Optional[tuple[th.Tensor, ...]] = None) -> th.Tensor:
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
            if hs:
                h = hs[d]
                h = self.conv_trans_blocks_a[d](h)
                h = self.ffparser[d](h)
                ha = self.conv_trans_blocks_b[d](h)
                hb = th.mean(h, (2, 3))
                hb = hb[:, :, None, None]
                x = x * ha * hb

        x = self.conv_blocks_context[-1](x)
        emb: th.Tensor = th.nn.Conv2d(x.size(1), 512, 1).to(device=x.device)(x)
        return emb


class ConvDropoutNormNonlin(torch.nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels: int, output_channels: int,
                 conv_op: Type[torch.nn.Conv2d] = torch.nn.Conv2d, conv_kwargs: Optional[dict[str, Any]] = None,
                 norm_op: Type[torch.nn.BatchNorm2d] = torch.nn.BatchNorm2d, norm_op_kwargs: Optional[dict[str, Any]] = None,
                 dropout_op: Type[torch.nn.Dropout2d] = torch.nn.Dropout2d, dropout_op_kwargs: Optional[dict[str, Any]] = None,
                 nonlin: Type[torch.nn.LeakyReLU] = torch.nn.LeakyReLU, nonlin_kwargs: Optional[dict[str, Any]] = None) -> None:
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class FFParser(torch.nn.Module):
    def __init__(self, dim: int, h: int = 128, w: int = 65) -> None:
        super().__init__()
        self.complex_weight = torch.nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x


class StackedConvLayers(torch.nn.Module):
    def __init__(self, input_feature_channels: int, output_feature_channels: int, num_convs: int, conv_kwargs: Optional[dict[str, Any]] = None, norm_op_kwargs: Optional[dict[str, Any]] = None, dropout_op_kwargs: Optional[dict[str, Any]] = None, nonlin_kwargs: Optional[dict[str, Any]] = None, first_stride: Optional[int] = None):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        if first_stride is not None:
            conv_kwargs_first_conv = {}
            conv_kwargs_first_conv.update(conv_kwargs)
            conv_kwargs_first_conv['stride'] = first_stride
        else:
            conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = torch.nn.Sequential(
            *([ConvDropoutNormNonlin(input_feature_channels, output_feature_channels, conv_kwargs=conv_kwargs_first_conv, norm_op_kwargs=norm_op_kwargs, dropout_op_kwargs=dropout_op_kwargs,nonlin_kwargs=nonlin_kwargs)] +
              [ConvDropoutNormNonlin(output_feature_channels, output_feature_channels, conv_kwargs=conv_kwargs_first_conv, norm_op_kwargs=norm_op_kwargs, dropout_op_kwargs=dropout_op_kwargs,nonlin_kwargs=nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def conv_transpose_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D transposed convolution module.
    """
    if dims == 1:
        return nn.ConvTranspose1d(*args, **kwargs)
    elif dims == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    elif dims == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def swap_ema(target_params, source_params):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    """
    for targ, src in zip(target_params, source_params):
        temp = targ.data.clone()
        targ.data.copy_(src.data)
        src.data.copy_(temp)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
