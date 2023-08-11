import cv2
import torch

import numpy as np
import argparse

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion

from denoising_diffusion_pytorch.modules import TanoSave


def main(args):
    ts = TanoSave(args.exp_id)

    chnn_dm = 3
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=chnn_dm, num_classes=1, ts=ts)  #.cuda()
    # model = Unet(dim=16, dim_mults=(1, 2, 4)).cuda()

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

    bs = 128  # 32
    step_TrEv = 500  # default: 1000, debug: 10  TODO: check bs 32

    trainer = Trainer(
        diffusion,
        '/data/denoising-diffusion-pytorch/work/20221219.ddpm.deepHomo/dataset/Train_List.txt',
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

    # trainer.results_folder = "results"

    # trainer.load("1")  # FS -- to comment

    trainer.train(ts)

    # sampled_images = diffusion.sample(batch_size=4)
    # sampled_images.shape  # (4, 3, 128, 128)
    # sampled_images = sampled_images.detach().cpu().numpy().transpose([0, 2, 3, 1])
    # for i, sample in enumerate(sampled_images):
    #     cv2.imwrite("/data/denoising-diffusion-pytorch/output/RE_frame-{}-diffusion.jpg".format(i), sample * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=0)
    args = parser.parse_args()

    main(args)
