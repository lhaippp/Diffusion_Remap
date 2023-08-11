# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

import sys
sys.path.append('/data/RAFT/core')

from utils import frame_utils
import cv2

from Things.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            # flow = frame_utils.read_gen(self.flow_list[index])

            try:
                flow = frame_utils.read_gen(self.flow_list[index])
            except:
                print(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        if 'clean' in self.image_list[index][0]:
            ph_occ = self.image_list[index][0].replace('clean', 'occlusions')
            img_occ = frame_utils.read_gen(ph_occ)
        elif 'final' in self.image_list[index][0]:
            ph_occ = self.image_list[index][0].replace('final', 'occlusions')
            img_occ = frame_utils.read_gen(ph_occ)

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img_occ = np.array(img_occ).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        img_occ = torch.from_numpy(img_occ).unsqueeze(0).float()

        if not self.sparse:
            return img1, img2, flow, img_occ

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


# ---------- LDM, 230112
class MpiSintel_cd_dm(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/data/FlowDatasets/Sintel', dstype='clean', image_size=128):
        super().__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        image_list_final, extra_info_final, flow_list_final = [], [], []

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

                # -- final
                image_list_final += [ [image_list[i].replace('clean', 'final'), image_list[i+1].replace('clean', 'final')] ]
                extra_info_final += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

        self.image_list += image_list_final
        self.extra_info += extra_info_final
        self.flow_list += self.flow_list

        # -- image size, 0112
        self.image_size = image_size

        # -- for occ, 0113
        self.augmentor = SparseFlowAugmentor(**aug_params)

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            # flow = frame_utils.read_gen(self.flow_list[index])

            try:
                flow = frame_utils.read_gen(self.flow_list[index])
            except:
                print(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        if 'clean' in self.image_list[index][0]:
            ph_occ = self.image_list[index][0].replace('clean', 'occlusions')
            img_occ = frame_utils.read_gen(ph_occ)
        elif 'final' in self.image_list[index][0]:
            ph_occ = self.image_list[index][0].replace('final', 'occlusions')
            img_occ = frame_utils.read_gen(ph_occ)

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img_occ = np.array(img_occ).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # -- resize to target size before crop, 0112
        img1, img2 = cv2.resize(img1, (self.image_size, self.image_size)), cv2.resize(img2, (self.image_size, self.image_size))
        flow = resize_flow(flow, self.image_size)
        img_occ = cv2.resize(img_occ, (self.image_size, self.image_size))

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                # img1, img2, flow = self.augmentor(img1, img2, flow)
                img1, img2, flow, img_occ = self.augmentor(img1, img2, flow, img_occ)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        img_occ = torch.from_numpy(img_occ).unsqueeze(0).float()

        img1, img2 = img1 / 255., img2 / 255.
        img_occ = img_occ / 255.

        if not self.sparse:
            # return img1, img2, flow, img_occ
            
            # print(img1.shape, img2.shape, flow.shape)
            img = torch.cat((img1, img2, flow), dim=0)
            return img, img_occ

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


def resize_flow(flow, size):
    h, w, _ = flow.shape

    res = cv2.resize(flow, (size, size))

    u_scale = (size / w)
    v_scale = (size / h)

    res[:, :, 0] = res[:, :, 0] * u_scale
    res[:, :, 1] = res[:, :, 1] * v_scale
    return res


class MpiSintel_Nori(FlowDataset):
    def __init__(self, aug_params=None, image_size=128):
        super().__init__(aug_params)
        data_clean = SintelNori(data_pass='clean')
        data_final = SintelNori(data_pass='final')
        self.image_list = data_clean.nori_list + data_final.nori_list

        self.func_get_item = SintelNori().get_item

        # -- image size, 0112
        self.image_size = image_size

        # -- for occ, 0113
        # self.augmentor = SparseFlowAugmentor(**aug_params)  # issue(0113): sparse flow

    def __getitem__(self, index):
        sample = self.func_get_item(self.image_list[index])
        img1, img2, flow, img_occ = sample['im1'], sample['im2'], sample['flow'], sample['img_occ']

        # -- resize to target size before crop, 0112
        img1, img2 = cv2.resize(img1, (self.image_size, self.image_size)), cv2.resize(img2, (self.image_size, self.image_size))
        flow = resize_flow(flow, self.image_size)
        img_occ = cv2.resize(img_occ, (self.image_size, self.image_size))

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img_occ = np.expand_dims(img_occ, axis=2)
                img1 = np.concatenate((img1, img_occ), 2)
                img1, img2, flow = self.augmentor(img1, img2, flow)
                img_occ = img1[:,:,3:]
                img1 = img1[:,:,:3]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        img_occ = torch.from_numpy(img_occ).permute(2, 0, 1).float()

        img1, img2 = img1 / 255., img2 / 255.
        img_occ = img_occ / 255.

        if not self.sparse:
            # return img1, img2, flow, img_occ
            
            # print(img1.shape, img2.shape, flow.shape)
            img = torch.cat((img1, img2, flow), dim=0)
            return img, img_occ

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


class SintelNori():
    def __init__(self, data_pass='clean', **kwargs):
        super().__init__()
        self.data_pass = data_pass
        # -- to speedup: nori speedup s3://tanodata/datasets/sintel_nori/sintel.nori --on --replica 3
        # self.ph_nori_oss = 's3://tanodata/datasets/sintel_nori'
        self.ph_nori_list = '/data/diffusion_remap/nori_list.npy'  # TODO: check path
        self.nori_list = self.__get_nori_list()

    def __get_nori_list(self):
        ph_nori = self.ph_nori_list
        if os.path.exists(ph_nori):
            nori_list = np.load(ph_nori, allow_pickle=True).item()  # for dict
        else:
            print('Error: cannot load nori_list.npy!')
        return nori_list[self.data_pass]

    @staticmethod
    def get_item(sample):
        name = sample['name']
        imgs = nori_fetcher_get(sample['imgs'], sample['shape_imgs'], np.uint8).copy()
        img1 = imgs[:, :, :3]
        img2 = imgs[:, :, 3:6]
        img_occ = imgs[:, :, 6:]
        flow = nori_fetcher_get(sample['flow'], sample['shape_flow'], np.float32).copy()
        valid = None
        data = {'im1': img1, 'im2': img2, 'flow': flow, 'img_occ': img_occ, 'valid': valid, 'name': name}
        return data


import nori2 as nori

def nori_fetcher_get(id, shape=(320, 640, 3), data_type=np.uint8):
    byte_data = nori.Fetcher().get(id)
    data = np.frombuffer(byte_data, data_type)
    data = np.reshape(data, shape)
    return data


import torchvision

if __name__ == '__main__':
    trainset = MpiSintelTrainset()
    len_data = len(trainset)
    print('data length: %d' % len_data)
    for ii in range(len_data):
        image1, image2, flow_gt, img_occ = trainset[ii]

        dir_save = '/data/Diffusion/latent-diffusion-main/outputs/debug/'
        torchvision.utils.save_image(image1/255., dir_save + 'image1.png')
        torchvision.utils.save_image(image2/255., dir_save + 'image2.png')
        torchvision.utils.save_image(img_occ/255., dir_save + 'img_occ.png')
        exit(0)
