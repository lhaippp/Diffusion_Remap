
import cv2
import torch
import argparse

import numpy as np

from denoising_diffusion_pytorch.modules import TanoSave
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=0)
    args = parser.parse_args()

    ts = TanoSave(args.exp_id)

    chnn_dm = 3
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=chnn_dm, num_classes=1, ts=ts)
    # model = Unet(dim=16, dim_mults=(1, 2, 4)).cuda()

    # ph_weight = '/data/users/lihaipeng/diffusion_remap-master/pretrain-models/model-120.pt'  # 
    # pretrained_dict = torch.load(ph_weight)['model']
    # import collections
    # pretrained_alter = collections.OrderedDict()
    # for k, v in pretrained_dict.items():
    #     k = k.replace('model.', '')
    #     pretrained_alter[k] = v
    # pretrained_dict = pretrained_alter

    # if len(model.state_dict().keys()) == len(pretrained_dict.keys()):
    #     model.load_state_dict(pretrained_dict)
    #     ts.print('Restore checkpoint from %s' % ph_weight)
    # else:
    #     new_params = [k for k in model.state_dict().keys() if k not in pretrained_dict.keys()]
    #     ts.print('Num of new params: %d \n %s' % (len(new_params), new_params))
    #     obso_params = [k for k in pretrained_dict.keys() if k not in model.state_dict().keys()]
    #     ts.print('Num of obso params: %d \n %s' % (len(obso_params), obso_params))
    #     model.load_state_dict(pretrained_dict, strict=False)

    sz_img = 256  # 256
    if sz_img != 128:
        ts.print('image_size: %d \n' % sz_img)

    diffusion = GaussianDiffusion(
        model,
        image_size=sz_img,  # 128
        timesteps=1000,  # number of steps
        sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type='l2',  # L1 or L2
        # p2_loss_weight_gamma=1.,
        objective='pred_x0',
        ts=ts,
    )

    bs = 1  # 32
    step_TrEv = 500

    trainer = Trainer(
        diffusion,
        '/data/users/lihaipeng/diffusion_remap-master/benchmark/GHOF_Clean_20230705.npy',
        train_batch_size=bs,  # 4 cards: 32, debug 8
        train_lr=1e-4,
        train_num_steps=60000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder="Outputs/results_"+str(args.exp_id),
        save_and_sample_every=step_TrEv,
        augment_horizontal_flip=False,
        exp_id=args.exp_id,
        num_samples=16,)

    trainer.load('120')
    # input()

    trainer.sampling()


