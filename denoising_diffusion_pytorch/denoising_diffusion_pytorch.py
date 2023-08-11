import os
import cv2
import math
import copy
import torch
import pickle
import imageio
import inspect

import numpy as np
# import nori2 as nori
import torch.nn.functional as F

from pathlib import Path
from random import random
from torch.optim import Adam
from torch import nn, einsum
from functools import partial
from collections import namedtuple
from einops import rearrange, reduce
from multiprocessing import cpu_count
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
from termcolor import colored
from accelerate import Accelerator
from collections import defaultdict

from denoising_diffusion_pytorch.version import __version__
# from Things.flyingthings import flyingthings_nori
# from Things.Sintel import MpiSintel_cd_dm, MpiSintel_Nori
import torchvision
import time
from denoising_diffusion_pytorch.modules import *


gb_exp_id = 0

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num)**2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1))


def Downsample(dim, dim_out=None):
    return nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2), nn.Conv2d(dim * 4, default(dim_out, dim), 1))


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(nn.Module):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(
            time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


# model

if False:
    class Unet(nn.Module):

        def __init__(self,
                     dim,
                     init_dim=None,
                     out_dim=None,
                     dim_mults=(1, 2, 4, 8),
                     channels=3,
                     self_condition=False,
                     resnet_block_groups=8,
                     learned_variance=False,
                     learned_sinusoidal_cond=False,
                     random_fourier_features=False,
                     learned_sinusoidal_dim=16):
            super().__init__()

            # determine dimensions

            self.channels = channels
            self.self_condition = self_condition
            input_channels = channels * (2 if self_condition else 1)

            init_dim = default(init_dim, dim)
            self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

            dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
            in_out = list(zip(dims[:-1], dims[1:]))

            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

            # time embeddings

            time_dim = dim * 4

            self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

            if self.random_or_learned_sinusoidal_cond:
                sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                    learned_sinusoidal_dim, random_fourier_features)
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(dim)
                fourier_dim = dim

            self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(
                fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

            # layers

            self.downs = nn.ModuleList([])
            self.ups = nn.ModuleList([])
            num_resolutions = len(in_out)

            for ind, (dim_in, dim_out) in enumerate(in_out):
                is_last = ind >= (num_resolutions - 1)

                self.downs.append(
                    nn.ModuleList([
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                            dim_in, dim_out, 3, padding=1)
                    ]))

            mid_dim = dims[-1]
            self.mid_block1 = block_klass(
                mid_dim, mid_dim, time_emb_dim=time_dim)
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
            self.mid_block2 = block_klass(
                mid_dim, mid_dim, time_emb_dim=time_dim)

            for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
                is_last = ind == (len(in_out) - 1)

                self.ups.append(
                    nn.ModuleList([
                        block_klass(dim_out + dim_in, dim_out,
                                    time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out,
                                    time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                            dim_out, dim_in, 3, padding=1)
                    ]))

            default_out_dim = channels * (1 if not learned_variance else 2)
            self.out_dim = default(out_dim, default_out_dim)

            self.final_res_block = block_klass(
                dim * 2, dim, time_emb_dim=time_dim)
            self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        def forward(self, x, time, x_self_cond=None):
            if self.self_condition:
                x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
                x = torch.cat((x_self_cond, x), dim=1)

            x = self.init_conv(x)
            r = x.clone()

            t = self.time_mlp(time)

            h = []

            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                h.append(x)

                x = block2(x, t)
                x = attn(x)
                h.append(x)

                x = downsample(x)

            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)

            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)

                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x, t)
                x = attn(x)

                x = upsample(x)

            x = torch.cat((x, r), dim=1)

            x = self.final_res_block(x, t)
            return self.final_conv(x)


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    # 图解gather函数 https://zhuanlan.zhihu.com/p/352877584
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):

    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            objective='pred_noise',
            beta_schedule='cosine',
            # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_gamma=0.,
            p2_loss_weight_k=1,
            ddim_sampling_eta=1.):
        super().__init__()
        assert not (
            type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {
            'pred_noise', 'pred_x0', 'pred_v'
        }, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k +
                        alphas_cumprod / (1 - alphas_cumprod))**-p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start)

    def predict_start_from_v(self, x_t, t, v):
        return (extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                          extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0], ), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x,
                                                                          t=batched_times,
                                                                          x_self_cond=x_self_cond,
                                                                          clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = unnormalize_to_zero_to_one(img)
        # unnormalize flow from [0, 1] to [-1, 1]
        img[:, -2:] = img[:, -2:] * 2 - 1
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch, ), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, * \
                _ = self.model_predictions(
                    img, time_cond, self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) *
                           (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

        img = unnormalize_to_zero_to_one(img)
        # unnormalize flow from [0, 1] to [-1, 1] and rescale with 255
        img[:, -2:] = (img[:, -2:] * 2 - 1) * 512
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b, ), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        # todo: what is self-conditioning?
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b, ), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


class CifarDataset(Dataset):

    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png', 'tiff'], augment_horizontal_flip=False, convert_image_to=None):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        self.cifar_data = [self.unpickle(os.path.join(folder, "data_batch_{}".format(i)))[
            b'data'] for i in range(1, 6)]
        self.cifar_data = np.concatenate(self.cifar_data, 0)

        self.datas = self.cifar_data.reshape(-1,
                                             3, 32, 32).transpose(0, 2, 3, 1)
        # print("cifar data shape is {}".format(self.datas.shape))
        # for i in range(10):
        #     import cv2
        #     cv2.imwrite("/data/denoising-diffusion-pytorch/SIGNS_dataset/test_{}.png".format(i), self.datas[i])

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(
            convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            cifar_dict = pickle.load(fo, encoding='bytes')
        return cifar_dict

    def __len__(self):
        # print("cifar dataset contains {} items".format(len(self.datas)))
        return len(self.datas)

    def __getitem__(self, index):
        img = self.datas[index]
        img = Image.fromarray(np.uint8(img))
        return self.transform(img)


class GHOFTestDataset(Dataset):
    def __init__(
        self,
        benchmark_path,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None,
    ):
        super().__init__()

        self.samples = np.load(benchmark_path, allow_pickle=True)

    def __len__(self):
        return len(self.samples)

    def upsample2d_flow_as(self, inputs, res=256, mode="bilinear", if_rate=False):
        # _, _, h, w = target_as.size()
        h, w = res, res
        res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
        if if_rate:
            _, _, h_, w_ = inputs.size()
            u_scale = (w / w_)
            v_scale = (h / h_)
            u, v = res.chunk(2, dim=1)
            u = u * u_scale
            v = v * v_scale
            res = torch.cat([u, v], dim=1)
        return res

    def __getitem__(self, idx):
        img1 = self.samples[idx]["img1"]
        img2 = self.samples[idx]["img2"]
        gt_flow = self.samples[idx]["gt_flow"]

        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.
        gt_flow = torch.from_numpy(gt_flow[None]).permute(0, 3, 1, 2).float()
        gt_flow = self.upsample2d_flow_as(gt_flow, 256, if_rate=True).squeeze()
        return torch.cat([img1, img2, gt_flow], dim=0), 0


class FrameDatasetNori(Dataset):

    def __init__(
        self,
        benchmark_path,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None,
    ):
        self.samples = self.collect_samples()

        self.fetcher = nori.Fetcher()

        self.image_size = image_size

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(
            convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            # T.RandomCrop(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def collect_samples(self):
        files = np.load(
            "/data/denoising-diffusion-pytorch/dataset/GOF9K.npy", allow_pickle=True).item()
        self.sources = list(files.keys())
        return files

    def bytes2np(self, data, c=3, h=600, w=800):
        data = np.fromstring(data, np.float32)
        data = data.reshape((h, w, c))
        return data

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        file = self.samples[idx]
        # print(file)

        h, w = file["h"], file["w"]

        img1_bytes = self.fetcher.get(file["img1"])
        # img2_bytes = self.fetcher.get(file["img2"])
        # gyro_filed_bytes = self.fetcher.get(file["gyro_field"])

        img1 = self.bytes2np(img1_bytes, c=3, h=h, w=w)
        # img2 = self.bytes2np(img2_bytes, c=3, h=h, w=w)
        # gyro_filed = self.bytes2np(gyro_filed_bytes, c=2, h=h, w=w)
        img = Image.fromarray(np.uint8(img1))
        return self.transform(img)


def imdecode(data, require_chl3=True, require_alpha=False):
    img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_UNCHANGED)

    assert img is not None, 'failed to decode'
    if img.ndim == 2 and require_chl3:
        img = img.reshape(img.shape + (1, ))
    if img.shape[2] == 1 and require_chl3:
        img = np.tile(img, (1, 1, 3))
    if img.ndim == 3 and img.shape[2] == 3 and require_alpha:
        assert img.dtype == np.uint8
        img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=2)
    return img


class HomoTrainData(Dataset):

    def __init__(
        self,
        benchmark_path,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None,
    ):
        # 路径
        self.nf = nori.Fetcher()

        self.data_infor = open(benchmark_path, 'r').readlines()

        self.image_size = image_size

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(
            convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            # T.RandomCrop(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):

        # img loading
        img_names = self.data_infor[idx]
        img_names = img_names.split(' ')

        data1 = self.nf.get(img_names[0])  # Read image according to data list
        data2 = self.nf.get(img_names[1][:-1])

        img1 = imdecode(data1)
        img2 = imdecode(data2)

        img = img1 if random() <= 0.5 else img2

        img = Image.fromarray(np.uint8(img1))
        return self.transform(img).float()


def resize_flow(flow, size):
    h, w, _ = flow.shape

    res = cv2.resize(flow, (size, size))

    u_scale = (size / w)
    v_scale = (size / h)

    res[:, :, 0] = res[:, :, 0] * u_scale
    res[:, :, 1] = res[:, :, 1] * v_scale
    return res


def flow_warp(x, flow12, pad="border", mode="bilinear"):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if "align_corners" in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(
            x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(
            x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


class FlyingThings(Dataset):

    def __init__(
        self,
        benchmark_path,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None,
    ):
        # 路径
        self.nf = nori.Fetcher()

        self.data_infor = flyingthings_nori(data_pass='clean')

        self.image_size = image_size

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(
            convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.ToTensor(),
            # T.Lambda(maybe_convert_fn),
            # T.Resize(image_size),
            # T.RandomCrop(image_size),
            # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.CenterCrop(image_size),
        ])

        # -=- Things

    def __len__(self):
        # return size of dataset
        print(self.__class__.__name__ +
              ' {}'.format(self.data_infor.len['train']))
        return self.data_infor.len['train']

    def __getitem__(self, idx):
        sample = self.data_infor.sample(index=idx, split='train')
        # im1, im2 [0, 255] uint8
        # flow [-255., -255.] float32
        im1, im2, flow = sample['im1'].astype(
            np.float32), sample['im2'].astype(np.float32), sample['flow']
        im1, im2 = im1 / 255., im2 / 255.
        # assert im1.dtype is np.dtype(np.float32) and im2.dtype is np.dtype(np.float32) and flow.dtype is np.dtype(
        #     np.float32), "images or flow should be float， im1:{}, im2:{}, flow:{}".format(im1.dtype, im2.dtype, flow.dtype)
        im1, im2 = cv2.resize(im1, (self.image_size, self.image_size)), cv2.resize(
            im2, (self.image_size, self.image_size))
        flow = resize_flow(flow, self.image_size)
        # normalize flow from [-1, 1] to [0, 1]
        # flow = (flow + 1) / 2.
        img = np.concatenate((im1, im2, flow), axis=2)
        # print("img range {}-{}".format(np.min(img), np.max(img)))

        # flow_torch = torch.from_numpy(flow[None].transpose(0, 3, 1, 2)) * 255
        # im2_torch = torch.from_numpy(im2[None].transpose(0, 3, 1, 2))
        # print("im2_torch shape {}".format(im2_torch.shape))
        # im2_warp = flow_warp(im2_torch, flow_torch)
        # im2_warp_np = im2_warp.detach().cpu().numpy().squeeze().transpose(1, 2, 0)

        # with imageio.get_writer(f'images/{"test"}.gif', mode='I', duration=0.5) as writer:
        #     buf1 = np.concatenate((im1, im1), 1)
        #     print("im2 shape {} | im2_warp_np shape {}".format(im2.shape, im2_warp_np.shape))
        #     buf2 = np.concatenate((im2, im2_warp_np), 1)
        #     writer.append_data(buf1[:, :, ::-1] * 255)
        #     writer.append_data(buf2[:, :, ::-1] * 255)
        # input()
        return self.transform(img).float()


def flow_to_image_luo(flow, display=False):
    """

        :param flow: H,W,2
        :param display:
        :return: H,W,3
        """

    def compute_color(u, v):

        def make_color_wheel():
            """
                Generate color wheel according Middlebury color code
                :return: Color wheel
                """
            RY = 15
            YG = 6
            GC = 4
            CB = 11
            BM = 13
            MR = 6

            ncols = RY + YG + GC + CB + BM + MR

            colorwheel = np.zeros([ncols, 3])

            col = 0

            # RY
            colorwheel[0:RY, 0] = 255
            colorwheel[0:RY, 1] = np.transpose(
                np.floor(255 * np.arange(0, RY) / RY))
            col += RY

            # YG
            colorwheel[col:col + YG, 0] = 255 - \
                np.transpose(np.floor(255 * np.arange(0, YG) / YG))
            colorwheel[col:col + YG, 1] = 255
            col += YG

            # GC
            colorwheel[col:col + GC, 1] = 255
            colorwheel[col:col + GC,
                       2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
            col += GC

            # CB
            colorwheel[col:col + CB, 1] = 255 - \
                np.transpose(np.floor(255 * np.arange(0, CB) / CB))
            colorwheel[col:col + CB, 2] = 255
            col += CB

            # BM
            colorwheel[col:col + BM, 2] = 255
            colorwheel[col:col + BM,
                       0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
            col += +BM

            # MR
            colorwheel[col:col + MR, 2] = 255 - \
                np.transpose(np.floor(255 * np.arange(0, MR) / MR))
            colorwheel[col:col + MR, 0] = 255

            return colorwheel

        """
            compute optical flow color map
            :param u: optical flow horizontal map
            :param v: optical flow vertical map
            :return: optical flow in color code
            """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u**2 + v**2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a + 1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

        return img

    UNKNOWN_FLOW_THRESH = 1e7
    """
        Convert flow into middlebury color code image
        :param flow: optical flow map
        :return: optical flow image in middlebury color
        """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (
            maxrad, minu, maxu, minv, maxv))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    # _min, _mean, _max = np.min(img), np.mean(img), np.max(img)
    # print(_min, _mean, _max)

    return img / 255.


def visulize_flow(all_images):
    np_flow = all_images.detach().cpu().numpy().transpose([0, 2, 3, 1])

    vis_flow = []

    for _, flow in enumerate(np_flow):
        vis_flow.append(flow_to_image_luo(flow))

    vis_flow_np = np.array(vis_flow, dtype=np.float32)
    vis_flow_np = vis_flow_np.transpose([0, 3, 1, 2])

    vis_flow_torch = torch.from_numpy(vis_flow_np)
    # print("vis_flow_torch shape ", vis_flow_torch.shape)
    return vis_flow_torch


def make_gif(img1, img2, name, back2ford_psnr):
    img1, img2 = cv2.imread(img1), cv2.imread(img2)

    global gb_exp_id
    ph_gif = 'Outputs/GIF_vis_%d' % gb_exp_id
    if not os.path.exists(ph_gif):
        os.makedirs(ph_gif)

    imageio.mimsave(f'{ph_gif}/{name}-psnr[{back2ford_psnr}].gif',
                    [img1[..., ::-1], img2[..., ::-1]], format='GIF', duration=500, loop=0)


class Trainer(object):

    def __init__(self,
                 diffusion_model,
                 folder,
                 *,
                 train_batch_size=16,
                 gradient_accumulate_every=1,
                 augment_horizontal_flip=True,
                 train_lr=1e-4,
                 train_num_steps=100000,
                 ema_update_every=10,
                 ema_decay=0.995,
                 adam_betas=(0.9, 0.99),
                 save_and_sample_every=1000,
                 num_samples=4,  # 9
                 results_folder='./results',
                 amp=False,
                 fp16=False,
                 split_batches=True,
                 convert_image_to=None,
                 exp_id=0):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches, mixed_precision='fp16' if fp16 else 'no')

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        # self.ds = Dataset(folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip, convert_image_to=convert_image_to)
        # self.ds = CifarDataset(folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip, convert_image_to=convert_image_to)
        self.ds = GHOFTestDataset(folder,
                                  self.image_size,
                                  augment_horizontal_flip=augment_horizontal_flip,
                                  convert_image_to=convert_image_to)
        # self.ds = FrameDatasetNori(folder,
        #                            self.image_size,
        #                            augment_horizontal_flip=augment_horizontal_flip,
        #                            convert_image_to=convert_image_to)
        # self.ds = HomoTrainData(folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip, convert_image_to=convert_image_to)
        # self.ds = FlyingThings(folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip, convert_image_to=convert_image_to)

        # -- sintel
        # aug_params = {'crop_size': [self.image_size, self.image_size], 'do_flip': True}
        # self.ds = MpiSintel_cd_dm(aug_params=aug_params, image_size=self.image_size + 16)  # 16 / 32 / 64
        # self.ds = MpiSintel_Nori(aug_params=aug_params, image_size=self.image_size + 16)

        dl = DataLoader(self.ds, batch_size=train_batch_size,
                        shuffle=False, pin_memory=True, num_workers=8)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(),
                        lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # -=- Trainer

        global gb_exp_id
        gb_exp_id = exp_id

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        if not self.accelerator.is_main_process:
            return

        data = torch.load(
            str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        # data = torch.load(os.path.join(self.results_folder, f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self, ts):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    # data = next(self.dl).to(device)

                    # -- with occ, 0113
                    data, occ = next(self.dl)
                    # -- data warp, 0118  -#-1
                    data, data_vis = data_warp(data, self.step)
                    data = data.to(device)
                    occ = occ.to(device)

                    with self.accelerator.autocast():
                        if False:  # 1:#  for debug
                            dir_debug = '/data/Diffusion/cd_diffusion/Outputs/results_99/debug_99/'
                        # if self.step % 10 == 0:  # for training
                        #     dir_debug = '/data/Diffusion/cd_diffusion/Outputs/results_99/debug/'

                            if not os.path.exists(dir_debug):
                                os.mkdir(dir_debug)
                            torchvision.utils.save_image(
                                data[0, :3], dir_debug + '%02d_img1.png' % (self.step % 100))
                            torchvision.utils.save_image(
                                data[0, 3:6], dir_debug + '%02d_img2.png' % (self.step % 100))
                            flow_vis = data[0, 6:].unsqueeze(0)
                            flow_vis = visulize_flow(
                                flow_vis).cuda(flow_vis.device)
                            torchvision.utils.save_image(
                                flow_vis, dir_debug + '%02d_flow.png' % (self.step % 100))
                            # import pdb;pdb.set_trace()

                        # data.shape: [b, 8, 128, 128]
                        loss = self.model(data, occ, classes=torch.randint(
                            0, 1, (data.shape[0], )).cuda())

                        # if torch.isnan(loss):
                        #     continue
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                if self.accelerator.is_local_main_process:
                    if self.step % 100 == 0:
                        time_str = time.strftime(
                            "%Y-%m-%d_Experiment time %H:%M:%S", time.localtime()).split('_')
                        ts.print('iter = %d, loss: %.4f , %s\n' %
                                 (self.step, total_loss, time_str))

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(
                                self.num_samples, self.batch_size)
                            # specifiy a certain flow
                            # flows = data[:, -2:].repeat(self.num_samples, 1, 1, 1)[:self.num_samples]
                            # all_images_list = list(
                            #     map(lambda n: self.ema.ema_model.sample(classes=torch.randint(0, 1, (n, )).cuda(), flow=flows[:n, ]), batches))

                            # -- 6i3o, 1119  -#-2
                            flows = data[:, 6:8].repeat(self.num_samples, 1, 1, 1)[
                                :self.num_samples]
                            flow_plus = data.repeat(self.num_samples, 1, 1, 1)[
                                :self.num_samples]
                            all_images_list = list(
                                map(lambda n: self.ema.ema_model.sample(classes=torch.randint(0, 1, (n, )).cuda(), flow=flow_plus[:n, ]), batches))

                        all_images = torch.cat(all_images_list, dim=0)
                        img1s, warp_img2s = postProcess(
                            all_images, batches, flows=flows, data_vis=data_vis)

                        # from rgb to bgr
                        permute = [2, 1, 0]
                        img1s, warp_img2s = img1s[:,
                                                  permute], warp_img2s[:, permute]

                        utils.save_image(img1s,
                                         str(self.results_folder /
                                             f'sample-{milestone}-source.png'),
                                         nrow=int(math.sqrt(self.num_samples)))
                        utils.save_image(warp_img2s,
                                         str(self.results_folder /
                                             f'sample-{milestone}-target.png'),
                                         nrow=int(math.sqrt(self.num_samples)))
                        make_gif(str(self.results_folder / f'sample-{milestone}-source.png'),
                                 str(self.results_folder / f'sample-{milestone}-target.png'), milestone)
                        self.save(milestone)

                # -- to avoid: ProcessGroupNCCL.cpp:821] [Rank 0] Watchdog caught collective operation -pvc  RUNNING  192974848/192974848 timeout: WorkNCCL(SeqNum=188661, OpType=ALLREDUCE, Timeout(ms)=1800000) ran for 1801470 milliseconds before timing out.
                # torch.barrier()  # how to use ?

                pbar.update(1)

        accelerator.print('training complete')

    def print_metrics(self, title="Eval", color="red"):
        print_str = " | ".join("{}: val:{:.4f}/avg:{:.4f}".format(k, v.val, v.avg)
                               for k, v in self.val_status.items())
        self.accelerator.print(colored("{} Results: {}".format(
            title, print_str), color, attrs=["bold"]))

    def postProcess(self, torch_tensor, flows, data_vis):
        img1s = torch_tensor[:, :3]
        img2s = torch_tensor[:, 3:6]

        if torch_tensor.shape[1] > 6:
            flows = torch_tensor[:, 6:]

            # flows = torch.clamp(flows, -1, 1)
            # flows = flows * flows.shape[-1]  # issue(0119): if flow not normed

        flows_vis = visulize_flow(flows).cuda(flows.device)

        # -- vis add forward warp, 230627
        img1, img2, img1_wp = data_vis

        # backward warping
        warp_img2 = flow_warp(img2, flows)

        # compute psnr
        B = img1s.shape[0]

        identity_psnr = compute_psnr(img1, img2)
        backWarp_psnr = compute_psnr(img1, warp_img2)
        fordWarp_naive_psnr = compute_psnr(img1_wp, img2)
        fordWarp_dm_psnr = compute_psnr(img2s, img2)

        self.val_status["identity_psnr"].update(
            val=identity_psnr.item(), num=B)
        self.val_status["backWarp_psnr"].update(
            val=backWarp_psnr.item(), num=B)
        self.val_status["fordWarp_naive_psnr"].update(
            val=fordWarp_naive_psnr.item(), num=B)
        self.val_status["fordWarp_dm_psnr"].update(
            val=fordWarp_dm_psnr.item(), num=B)

        buf1_back = torch.concat(
            [img1, torch.zeros_like(img1), img1, flows_vis], -1)
        buf2_back = torch.concat(
            [img2, torch.zeros_like(img1), warp_img2, flows_vis], -1)

        buf1_forw = torch.concat([img1, img1_wp, img2s, flows_vis], -1)
        buf2_forw = torch.concat([img2, img2, img2, flows_vis], -1)
        return torch.cat([buf1_back, buf1_forw], 2), torch.cat([buf2_back, buf2_forw], 2)

    def sampling(self):
        self.ema.to(self.accelerator.device)
        self.ema.update()
        self.ema.ema_model.eval()

        self.val_status = defaultdict(AverageMeter)

        for idx, data_blob in enumerate(self.dl):
            milestone = str(idx)

            data, occ = data_blob
            # -- data warp, 0118  -#-1
            # data = data.cuda()
            # occ = occ.cuda()
            data, data_vis = data_warp(data, 0)

            with torch.no_grad():
                # -- 6i3o, 1119  -#-2
                #     data = torch.cat((img2, masked_img1_wp, flow, img_diff, img1), 1)    # :3  3:6  6:8  8:9  9:12
                # .repeat(self.num_samples, 1, 1, 1)[:self.num_samples]
                flows = data[:, 6:8]
                # .repeat(self.num_samples, 1, 1, 1)[:self.num_samples]
                flow_plus = data
                out = self.ema.ema_model.sample(
                    classes=torch.randint(0, 1, (1, )).cuda(), flow=flow_plus)

            img1s, warp_img2s = self.postProcess(
                out, flows=flows, data_vis=data_vis)

            # from rgb to bgr
            permute = [2, 1, 0]
            img1s, warp_img2s = img1s[:, permute], warp_img2s[:, permute]

            # print PSNR
            self.print_metrics(title="val", color="green")

            backWarp_psnr = self.val_status['backWarp_psnr'].val
            fordWarp_psnr = self.val_status['fordWarp_dm_psnr'].val

            ph_src = os.path.join(self.results_folder,
                                  'sample-{}-source-psnr-{:.2f}.png'.format(milestone, backWarp_psnr - fordWarp_psnr))
            ph_tar = os.path.join(self.results_folder,
                                  'sample-{}-target-psnr-{:.2f}.png'.format(milestone, backWarp_psnr - fordWarp_psnr))
            utils.save_image(img1s,
                             ph_src,
                             nrow=int(math.sqrt(self.num_samples)))
            utils.save_image(warp_img2s,
                             ph_tar,
                             nrow=int(math.sqrt(self.num_samples)))
            make_gif(ph_src, ph_tar, milestone, '{:.2f}'.format(
                backWarp_psnr - fordWarp_psnr))

            # import pdb;pdb.set_trace()
        return 0


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def compute_psnr(img1, img2):
    if torch.max(img1) <= 1:
        # print(
        #     f'normalize img1 from [{torch.min(img1)}, {torch.max(img1)}] to [0, 255]')
        img1 = img1 * 255.
    if torch.max(img2) <= 1:
        # print(
        #     f'normalize img2 from [{torch.min(img2)}, {torch.max(img2)}] to [0, 255]')
        img2 = img2 * 255.
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


def data_warp(data, idx):
    img1 = data[:, :3]
    img2 = data[:, 3:6]
    flow = data[:, 6:]

    device = img1.device
    b, _, h, w = img1.shape
    tenMetric = torch.ones(b, 1, h, w).to(device)
    img1_wp = FunctionSoftsplat(
        tenInput=img1, tenFlow=flow, tenMetric=tenMetric, strType='softmax')

    ii = 0.2
    img_diff = (img2 - img1_wp).sum(1, True)
    img_diff[img_diff > ii] = 1
    img_diff[img_diff < -1*ii] = 1
    img_diff[img_diff != 1] = 0
    masked_img1_wp = img1_wp * (1 - img_diff)

    # -- vis debug, 1118
    if False:  # 1:#
        dir_save = '/data/Diffusion/cd_diffusion/Outputs/results_99/debug_99/occ_warped/'
        img1_wp_vis = img1_wp  # [0, :]
        torchvision.utils.save_image(
            img1_wp_vis/img1_wp_vis.max(), dir_save+'img1to2_warp.png')
        torchvision.utils.save_image(
            masked_img1_wp, dir_save+'img1_wp_mask0.2.png')
        import pdb
        pdb.set_trace()

        # thr = [0.1, 0.2, 0.3]  # , 0.4, 0.5, 0.6, 0.8, 1.0
        # while(1):
        #     for ii in thr:
        #         img_diff = (img2 - img1_wp).sum(1, True)
        #         img_diff[img_diff > ii] = 1
        #         img_diff[img_diff < -1*ii] = 1
        #         img_diff[img_diff != 1] = 0
        #         torchvision.utils.save_image(img_diff, dir_save+'%04d_img_diff_%.1f.png' % (idx, ii))

        #     if idx / 10 >= 1:
        #         import pdb;pdb.set_trace()

    # data = torch.cat((img2, masked_img1_wp, flow, img_diff, img1), 1)
    # :3  3:6  6:8  8:9  9:12

    # -- directly use img1_wp, 230627
    data = torch.cat((img2, img1_wp, flow, img_diff, img1), 1)

    return data, [img1, img2, img1_wp]
