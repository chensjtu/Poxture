# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os.path
import torchvision.transforms as transforms
import numpy as np
import glob
from tqdm import tqdm

from .dataset import DatasetBase
from .transforms import ToTensor, ImageTransformer

from iPERCore.tools.utils.filesio import cv_utils
from iPERCore.tools.utils.filesio.persistence import load_pickle_file, load_json_file

__all__ = ["FashionVideoDataset", "SeqFashionVideoDataset"]


class FashionVideoDataset(DatasetBase):

    def __init__(self, opt, is_for_train):
        super(FashionVideoDataset, self).__init__(opt, is_for_train)
        self._name = "FashionVideoDataset"

        self._intervals = opt.intervals

        # read dataset
        self._read_dataset_paths()

    def __getitem__(self, index):
        # assert (index < self._dataset_size)

        # start_time = time.time()
        # get sample data
        v_info = self._vids_info[index % self._num_videos]
        images, smpls, masks, offsets = self._load_pairs(v_info)

        # pack data
        sample = {
            "images": images,
            "smpls": smpls,
            "masks": masks,
            "offsets": offsets
        }

        sample = self._transform(sample)
        # print(time.time() - start_time)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        self._root = self._opt.fashion_dir
        self._processed_dir = os.path.join(self._root, "processed")

        # read video list
        self._num_videos = 0
        self._dataset_size = 0
        use_ids_filename = "train" if self._is_for_train else "val"
        train_val_dir = os.path.join(self._processed_dir, use_ids_filename)
        self._vids_info = self._read_vids_info(train_val_dir)

    def _read_vids_info(self, train_val_dir):
        vids_info = []

        for line in tqdm(os.listdir(train_val_dir)):
            images_path = glob.glob(os.path.join(train_val_dir, line, "images", "*"))
            images_path.sort()

            alphas_path = glob.glob(os.path.join(train_val_dir, line, "parse", "*_alpha.png"))
            alphas_path.sort()

            smpl_data = load_pickle_file(os.path.join(train_val_dir, line, "pose_shape.pkl"))
            cams = smpl_data["cams"]
            thetas = smpl_data["pose"]

            length = len(images_path)
            assert length == len(cams), "{} != {}".format(length, len(cams))
            assert length == len(alphas_path), "{} != {}".format(length, len(alphas_path))

            if thetas.shape[-1] == 96:
                thetas = cv_utils.quat2rotvec(thetas)

            betas = smpl_data["shape"]
            smpls = np.concatenate([cams, thetas, betas], axis=1)

            if "offsets" in smpl_data:
                offsets = smpl_data["offsets"]
            else:
                offsets = np.zeros((1, 6890, 3), dtype=np.float32)

            front_file = os.path.join(train_val_dir, line, "front_path.json")
            front_info = load_json_file(front_file)

            info = {
                "name": line,
                "length": len(images_path),
                "images": images_path,
                "alphas": alphas_path,
                "smpls": smpls,
                "offsets": offsets,
                "ft_ids": front_info["ft"]["ids"],
                "bk_ids": front_info["bk"]["ids"]
            }
            vids_info.append(info)
            self._dataset_size += info['length'] // self._intervals
            self._num_videos += 1

        return vids_info

    @property
    def video_info(self):
        return self._vids_info

    def _load_pairs(self, vid_info):
        length = vid_info['length']

        start = np.random.randint(0, 15)
        end = np.random.randint(0, length)
        pair_ids = np.array([start, end], dtype=np.int32)

        smpls = vid_info["smpls"][pair_ids]

        images = []
        masks = []
        images_paths = vid_info["images"]
        alphas_paths = vid_info["alphas"]
        for t in pair_ids:
            image_path = images_paths[t]
            image = cv_utils.read_cv2_img(image_path)

            images.append(image)

            mask = cv_utils.read_mask(alphas_paths[t], self._opt.image_size)

            # front is 0, and background is 1
            mask = 1.0 - mask
            masks.append(mask)

        return images, smpls, masks, vid_info["offsets"]

    def _create_transform(self):
        transform_list = [
            ImageTransformer(output_size=self._opt.image_size),
            ToTensor()]
        self._transform = transforms.Compose(transform_list)


class SeqFashionVideoDataset(FashionVideoDataset):

    def __init__(self, opt, is_for_train):
        super(SeqFashionVideoDataset, self).__init__(opt, is_for_train)
        self._name = 'SeqFashionVideoDataset'

    def _load_pairs(self, vid_info):
        length = vid_info["length"]
        ft_ids = vid_info["ft_ids"]
        ns = self._opt.num_source

        replace = ns >= len(ft_ids)
        src_ids = list(np.random.choice(ft_ids, ns, replace=replace))
        src_ids[0] = ft_ids[0]

        tsf_ids = list(np.random.choice(length, self._opt.time_step, replace=False))
        tsf_ids.sort()

        pair_ids = src_ids + tsf_ids

        smpls = vid_info["smpls"][pair_ids]

        images = []
        masks = []
        images_paths = vid_info["images"]
        alphas_paths = vid_info["alphas"]
        for t in pair_ids:
            image = cv_utils.read_cv2_img(images_paths[t])

            images.append(image)

            mask = cv_utils.read_mask(alphas_paths[t], self._opt.image_size)

            # front is 0, and background is 1
            mask = 1.0 - mask
            masks.append(mask)

        return images, smpls, masks, vid_info["offsets"]


