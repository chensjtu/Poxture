# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import cv2
import numpy as np
from tqdm import tqdm

from iPERCore.tools.utils.filesio import cv_utils
from iPERCore.tools.utils.filesio.persistence import load_pickle_file
from iPERCore.services.options.process_info import read_src_infos

from .dataset import VideoDataset


class ProcessedVideoDataset(VideoDataset):

    def __init__(self, opt, is_for_train):
        super(ProcessedVideoDataset, self).__init__(opt, is_for_train)
        self._dir_lenth = [] 
        self._read_vids_info()

    def _read_single_dataset(self, data_dir, txt_path):
        with open(txt_path, "r") as reader:

            for line in tqdm(reader.readlines()):

                vid_name = line.rstrip()
                vid_info_path = os.path.join(data_dir, "primitives", vid_name, "processed", "vid_info.pkl")

                # print(vid_info_path)
                vid_info = load_pickle_file(vid_info_path)
                vid_info = read_src_infos(vid_info, num_source=self._opt.num_source, ignore_bg=True)

                self._vids_info.append(vid_info)
                self._num_videos += 1
                self._dataset_size += vid_info["length"]
                self._dir_lenth.append(self._dataset_size) 
                # if self._dataset_size == 579:
                #     break

    def _read_vids_info(self):

        data_dir_list = self._opt.dataset_dirs
        for data_dir in data_dir_list:
            if self._opt.for_desired_txt == "":
                if self._is_for_train:
                    if self._opt.debug:
                        txt_path = '/date/dataset_share/iper/iper/debug.txt' # this is for debug
                    else:
                        txt_path = os.path.join(data_dir, "train.txt")
                    # 
                else:
                    if self._opt.debug:
                        txt_path = '/date/dataset_share/iper/iper/debug_val.txt' # this is for debug
                    else:
                        txt_path = os.path.join(data_dir, "val.txt")
            else:
                txt_path = self._opt.for_desired_txt
                print("this project will use %s"%self._opt.for_desired_txt)
            self._read_single_dataset(data_dir, txt_path)

    def _load_pairs(self, vid_info):
        '''
        this is for pint train.
        '''
        ns = 1

        length = vid_info["length"]
        ft_ids = vid_info["ft_ids"]

        replace = ns >= len(ft_ids)
        src_ids = list(np.random.choice(ft_ids, ns, replace=replace))
        src_ids[0] = ft_ids[0]

        tsf_ids = list(np.random.choice(length, self._opt.time_step, replace=False))
        # tsf_ids.sort()

        # take the source and target ids
        pair_ids = src_ids + tsf_ids
        # smpls = vid_info["smpls"][pair_ids]

        # fixed in 3.30, here only load the data need for train
        image_dir = vid_info["img_dir"]
        # image_dir = image_dir.replace('/media/Diske/Datasets/', '/date/dataset_share/iper/')
        images_names = vid_info["images"]
        alphas_paths = vid_info["alpha_paths"]
        
        # for t in pair_ids:
        #     image_path = os.path.join(image_dir, images_names[t])
        #     image = cv_utils.read_cv2_img(image_path)
        #     images.append(image)
            # extra_data_path = os.path.join(image_dir[:-6], 'extra_data/'+images_names[t][:-4]+'.pkl')
            # extra_data = load_pickle_file(extra_data_path)
            
            # Ts.append(extra_data['Tuv2src'])
            # head_bboxs.append(extra_data['head_bbox'])

            # uv_img_path = os.path.join(image_dir[:-6], 'uv_img/'+images_names[t][:-4]+'.jpg')
            # uv_img = cv_utils.read_cv2_img(uv_img_path)
            # uv_imgs.append(uv_img)

            # ref_uv_path = os.path.join(image_dir[:-6], 'ref_uv/'+images_names[t][:-4]+'.jpg')
            # ref_uv = cv_utils.read_cv2_img(ref_uv_path)
            # ref_uvs.append(ref_uv)

            # mask = cv_utils.read_mask(alphas_paths[t], self._opt.image_size)
            # front is 0, and background is 1
            # mask = 1.0 - mask
            # masks.append(mask)
        # as mentioned above, in this program
        # we need tsf's images, masks, Ts, head_bbox, G_tsf
        # src's uv_imgs
        images = cv_utils.read_cv2_img(os.path.join(image_dir, images_names[pair_ids[1]]))
        uv_imgs = cv_utils.read_cv2_img(os.path.join(image_dir[:-6], 'uv_img/'+images_names[pair_ids[0]][:-4]+'.jpg'))
        masks = cv_utils.read_mask(alphas_paths[pair_ids[1]], self._opt.image_size)
        ref_uvs = cv_utils.read_cv2_img(os.path.join(image_dir[:-6], 'ref_uv/'+images_names[pair_ids[1]][:-4]+'.jpg'))
        extra_data_path = os.path.join(image_dir[:-6], 'extra_data/'+images_names[pair_ids[1]][:-4]+'.pkl')
        extra_data = load_pickle_file(extra_data_path)
        Ts = extra_data['Tuv2src']
        head_bboxs = extra_data['head_bbox']
        # uv_index = (cv2.imread(os.path.join(image_dir[:-6], 'uv_indx/'+images_names[pair_ids[1]][:-4]+'.jpg'), 0)[None]).astype(np.int16)
        # uv_index = ((np.sign(uv_index-127)+1)/2).astype(np.uint8).repeat(3,axis=0)
        G_tsf = cv2.imread(os.path.join(image_dir[:-6], 'input_G_tsf/'+images_names[pair_ids[1]][:-4]+'.jpg'))
        # uv_index = cv_utils.read_cv2_img(os.path.join(image_dir[:-6], 'uv_indx/'+images_names[pair_ids[1]][:-4]+'.jpg'))
        return images, masks, uv_imgs, Ts, head_bboxs,G_tsf,ref_uvs


    def __getitem__(self, index):
        
        vid_info = self._vids_info[index % self._num_videos]

        # images, masks, uv_imgs, ref_uvs, Ts, head_bboxs, uv_index, G_tsf= self._load_pairs(vid_info)
        images, masks, uv_imgs, Ts, head_bboxs,G_tsf,ref_uvs= self._load_pairs(vid_info)
        # pack data
        sample = {
            "images": images,
            "masks": masks,
            "uv_imgs":uv_imgs,
            "ref_uvs":ref_uvs,
            "Ts":Ts,
            "head_bboxs":head_bboxs,
            # "uv_index":uv_index,
            "G_tsf":G_tsf,
        }

        sample = self._transform(sample)
        # print(sample)
        return sample

class ProcessedVideoDataset_for_mkdata(VideoDataset):

    def __init__(self, opt, is_for_train):
        super(ProcessedVideoDataset_for_mkdata, self).__init__(opt, is_for_train)
        self._dir_lenth = [] 
        self._read_vids_info()

    def _read_single_dataset(self, data_dir, txt_path):
        with open(txt_path, "r") as reader:

            for line in tqdm(reader.readlines()):

                vid_name = line.rstrip()
                vid_info_path = os.path.join(data_dir, "primitives", vid_name, "processed", "vid_info.pkl")

                # print(vid_info_path)
                vid_info = load_pickle_file(vid_info_path)
                vid_info = read_src_infos(vid_info, num_source=self._opt.num_source, ignore_bg=True)

                self._vids_info.append(vid_info)
                self._num_videos += 1
                self._dataset_size += vid_info["length"]
                self._dir_lenth.append(self._dataset_size) 
                # if self._dataset_size == 579:
                #     break

    def _read_vids_info(self):

        data_dir_list = self._opt.dataset_dirs
        for data_dir in data_dir_list:

            if self._is_for_train:
                if self._opt.debug:
                    txt_path = '/date/dataset_share/iper/iper/debug.txt' # this is for debug
                else:
                    txt_path = os.path.join(data_dir, "all.txt")
                # 
            else:
                if self._opt.debug:
                    txt_path = '/date/dataset_share/iper/iper/debug_val.txt' # this is for debug
                else:
                    txt_path = os.path.join(data_dir, "val.txt")

            self._read_single_dataset(data_dir, txt_path)

    def load_imgs(self, vid_info, dir_index):
        images = []
        masks = []
        ids = [0]+[dir_index]
        # take the source and target ids

        smpls = vid_info["smpls"][ids]
        image_dir = vid_info["img_dir"]
        image_dir = image_dir.replace('Diske', 'Diskf')
        # print(image_dir)
        images_names = vid_info["images"]
        alphas_paths = vid_info["alpha_paths"]
        images_path = os.path.join(image_dir, images_names[dir_index])
        images.append(cv_utils.read_cv2_img(os.path.join(image_dir, images_names[0])))
        images.append(cv_utils.read_cv2_img(images_path))
        # front is 0, and background is 1
        mask0 = cv_utils.read_mask(alphas_paths[0], self._opt.image_size)
        mask0 = 1.0 - mask0
        mask1 = cv_utils.read_mask(alphas_paths[dir_index], self._opt.image_size)
        mask1 = 1.0 - mask1
        masks.append(mask0)
        masks.append(mask1)
        return images, smpls, masks, image_dir, images_names[dir_index]

    def __getitem__(self, index):
        for i in range(self._num_videos):
            if index < self._dir_lenth[0]:
                num = 0
                dir_index = index
                break
            elif index < self._dir_lenth[i]:
                num = i
                dir_index = index-self._dir_lenth[i-1]
                break

        vid_info = self._vids_info[num]

        images, smpls, masks, image_dir, image_name = self.load_imgs(vid_info, dir_index)

        # pack data
        sample = {
            "images": images,
            "smpls": smpls,
            "masks": masks,
            "img_dir": image_dir,
            "image_name":image_name,
        }

        sample = self._transform(sample)
        return sample

    # def __getitem__(self, index):
    #     """

    #     Args:
    #         index (int): the sample index of self._dataset_size.

    #     Returns:
    #         sample (dict): the data sample, it contains the following informations:
    #             --images (torch.Tensor): (ns + nt, 3, h, w), here `ns` and `nt` are the number of source and targets;
    #             --masks (torch.Tensor): (ns + nt, 1, h, w);
    #             --smpls (torch.Tensor): (ns + nt, 85);

    #     """

    #         mask = 1.0 - mask
    #         masks.append(mask)


    #     images, smpls, masks = self._load_pairs(vid_info)

    #     # pack data
    #     sample = {
    #         "images": images,
    #         "smpls": smpls,
    #         "masks": masks
    #     }

    #     sample = self._transform(sample)
    #     # print(sample)
    #     return sample

    # def __getitem__(self, index):
    #     for i in range(self._num_videos):
    #         if index < self._dir_lenth[0]:
    #             num = 0
    #             dir_index = index
    #             break
    #         elif index < self._dir_lenth[i]:
    #             num = i
    #             dir_index = index-self._dir_lenth[i-1]
    #             break

    #     vid_info = self._vids_info[num]

    #     images, smpls, masks, image_dir, image_name = self.load_imgs(vid_info, dir_index)

    #     # pack data
    #     sample = {
    #         "images": images,
    #         "smpls": smpls,
    #         "masks": masks,
    #         "img_dir": image_dir,
    #         "image_name":image_name,
    #     }

    #     sample = self._transform(sample)
    #     return sample

class ProcessedVideoDataset_for_test(VideoDataset):

    def __init__(self, opt, is_for_train):
        super(ProcessedVideoDataset_for_test, self).__init__(opt, is_for_train)
        self._dir_lenth = [] 
        self._read_vids_info()

    def _read_single_dataset(self, data_dir, txt_path):
        with open(txt_path, "r") as reader:

            for line in tqdm(reader.readlines()):

                vid_name = line.rstrip()
                vid_info_path = os.path.join(data_dir, "primitives", vid_name, "processed", "vid_info.pkl")

                # print(vid_info_path)
                vid_info = load_pickle_file(vid_info_path)
                vid_info = read_src_infos(vid_info, num_source=self._opt.num_source, ignore_bg=True)

                self._vids_info.append(vid_info)
                self._num_videos += 1
                self._dataset_size += vid_info["length"]
                self._dir_lenth.append(self._dataset_size) 
                # if self._dataset_size == 579:
                #     break

    def _read_vids_info(self):

        data_dir_list = self._opt.dataset_dirs
        for data_dir in data_dir_list:

            if self._is_for_train:
                if self._opt.debug:
                    txt_path = '/date/dataset_share/iper/iper/debug.txt' # this is for debug
                else:
                    txt_path = os.path.join(data_dir, "train.txt")
                # 
            else:
                if self._opt.debug:
                    txt_path = '/date/dataset_share/iper/iper/debug_val.txt' # this is for debug
                else:
                    txt_path = os.path.join(data_dir, "val.txt")

            txt_path = '/date/dataset_share/iper/iper/debug_val.txt' # this is for debug
            self._read_single_dataset(data_dir, txt_path)

    def _load_pairs(self, vid_info, index):
        '''
        this is for pint train.
        '''
        ns = 1

        length = vid_info["length"]
        ft_ids = vid_info["ft_ids"]

        replace = ns >= len(ft_ids)
        src_ids = list(np.random.choice(ft_ids, ns, replace=replace))
        src_ids[0] = ft_ids[0]

        # tsf_ids = list(np.random.choice(length, self._opt.time_step, replace=False))
        # # tsf_ids.sort()
        tsf_ids = [index]

        # take the source and target ids
        pair_ids = src_ids + tsf_ids
        # smpls = vid_info["smpls"][pair_ids]

        # fixed in 3.30, here only load the data need for train
        image_dir = vid_info["img_dir"]
        # image_dir = image_dir.replace('/media/Diske/Datasets/', '/date/dataset_share/iper/')
        images_names = vid_info["images"]
        alphas_paths = vid_info["alpha_paths"]
        
        # for t in pair_ids:
        #     image_path = os.path.join(image_dir, images_names[t])
        #     image = cv_utils.read_cv2_img(image_path)
        #     images.append(image)
            # extra_data_path = os.path.join(image_dir[:-6], 'extra_data/'+images_names[t][:-4]+'.pkl')
            # extra_data = load_pickle_file(extra_data_path)
            
            # Ts.append(extra_data['Tuv2src'])
            # head_bboxs.append(extra_data['head_bbox'])

            # uv_img_path = os.path.join(image_dir[:-6], 'uv_img/'+images_names[t][:-4]+'.jpg')
            # uv_img = cv_utils.read_cv2_img(uv_img_path)
            # uv_imgs.append(uv_img)

            # ref_uv_path = os.path.join(image_dir[:-6], 'ref_uv/'+images_names[t][:-4]+'.jpg')
            # ref_uv = cv_utils.read_cv2_img(ref_uv_path)
            # ref_uvs.append(ref_uv)

            # mask = cv_utils.read_mask(alphas_paths[t], self._opt.image_size)
            # front is 0, and background is 1
            # mask = 1.0 - mask
            # masks.append(mask)
        # as mentioned above, in this program
        # we need tsf's images, masks, Ts, head_bbox, G_tsf
        # src's uv_imgs
        images = cv_utils.read_cv2_img(os.path.join(image_dir, images_names[pair_ids[1]]))
        # uv_imgs = cv_utils.read_cv2_img(os.path.join(image_dir[:-6], 'uv_img/'+images_names[pair_ids[0]][:-4]+'.jpg'))   

        # use Specify One.
        if os.path.exists(os.path.join(self._opt.source_img_folder, 'uv_img', "frame_00000000.jpg")):
            uv_imgs = cv_utils.read_cv2_img(os.path.join(self._opt.source_img_folder, 'uv_img', "frame_00000000.jpg"))
        elif os.path.exists(os.path.join(self._opt.source_img_folder, 'uv_img', "000.jpg")):
            uv_imgs = cv_utils.read_cv2_img(os.path.join(self._opt.source_img_folder, 'uv_img', "000.jpg"))
        else:
            uv_imgs = cv_utils.read_cv2_img(os.path.join(self._opt.source_img_folder, 'uv_img', "0000.jpg"))
        uv_imgs = cv_utils.read_cv2_img("/date/dataset_share/iper/iper/primitives/001/1/1/processed/uv_img/0139.jpg")

        masks = cv_utils.read_mask(alphas_paths[pair_ids[1]], self._opt.image_size)
        # ref_uvs = cv_utils.read_cv2_img(os.path.join(image_dir[:-6], 'ref_uv/'+images_names[pair_ids[1]][:-4]+'.jpg'))
        extra_data_path = os.path.join(image_dir[:-6], 'extra_data/'+images_names[pair_ids[1]][:-4]+'.pkl')
        extra_data = load_pickle_file(extra_data_path)
        Ts = extra_data['Tuv2src']
        head_bboxs = extra_data['head_bbox']
        # uv_index = (cv2.imread(os.path.join(image_dir[:-6], 'uv_indx/'+images_names[pair_ids[1]][:-4]+'.jpg'), 0)[None]).astype(np.int16)
        # uv_index = ((np.sign(uv_index-127)+1)/2).astype(np.uint8).repeat(3,axis=0)
        G_tsf = cv2.imread(os.path.join(image_dir[:-6], 'input_G_tsf/'+images_names[pair_ids[1]][:-4]+'.jpg'))
        # uv_index = cv_utils.read_cv2_img(os.path.join(image_dir[:-6], 'uv_indx/'+images_names[pair_ids[1]][:-4]+'.jpg'))
        return images, masks, uv_imgs, Ts, head_bboxs,G_tsf


    def __getitem__(self, index):
        
        vid_info = self._vids_info[index % self._num_videos]
        # images, masks, uv_imgs, ref_uvs, Ts, head_bboxs, uv_index, G_tsf= self._load_pairs(vid_info)
        images, masks, uv_imgs, Ts, head_bboxs,G_tsf= self._load_pairs(vid_info, index)
        # pack data
        sample = {
            "images": images,
            "masks": masks,
            "uv_imgs":uv_imgs,
            # "ref_uvs":ref_uvs,
            "Ts":Ts,
            "head_bboxs":head_bboxs,
            # "uv_index":uv_index,
            "G_tsf":G_tsf,
        }

        sample = self._transform(sample)
        # print(sample)
        return sample

class ProcessedVideoDataset_for_brokentexture(VideoDataset):

    def __init__(self, opt, is_for_train):
        super(ProcessedVideoDataset_for_brokentexture, self).__init__(opt, is_for_train)
        self._dir_lenth = [] 
        self._read_vids_info()

    def _read_single_dataset(self, data_dir, txt_path):
        with open(txt_path, "r") as reader:

            for line in tqdm(reader.readlines()):

                vid_name = line.rstrip()
                vid_info_path = os.path.join(data_dir, "primitives", vid_name, "processed", "vid_info.pkl")

                # print(vid_info_path)
                vid_info = load_pickle_file(vid_info_path)
                vid_info = read_src_infos(vid_info, num_source=self._opt.num_source, ignore_bg=True)

                self._vids_info.append(vid_info)
                self._num_videos += 1
                self._dataset_size += vid_info["length"]
                self._dir_lenth.append(self._dataset_size) 
                # if self._dataset_size == 579:
                #     break

    def _read_vids_info(self):

        data_dir_list = self._opt.dataset_dirs
        for data_dir in data_dir_list:

            if self._is_for_train:
                if self._opt.debug:
                    txt_path = '/date/dataset_share/iper/iper/debug.txt' # this is for debug
                else:
                    txt_path = os.path.join(data_dir, "train.txt")
                # 
            else:
                if self._opt.debug:
                    txt_path = '/date/dataset_share/iper/iper/debug_val.txt' # this is for debug
                else:
                    txt_path = os.path.join(data_dir, "val.txt")

            self._read_single_dataset(data_dir, txt_path)

    def _load_pairs(self, vid_info):
        '''
        this is for pint train.
        '''
        ns = 1

        length = vid_info["length"]
        ft_ids = vid_info["ft_ids"]

        # replace = ns >= len(ft_ids)
        # src_ids = list(np.random.choice(ft_ids, ns, replace=replace))
        # src_ids[0] = 0
        # src_ids[0] = ft_ids[0]

        tsf_ids = list(np.random.choice(length, self._opt.time_step, replace=False))
        # tsf_ids.sort()

        # take the source and target ids
        pair_ids = list([0]) + tsf_ids
        # smpls = vid_info["smpls"][pair_ids]

        # fixed in 3.30, here only load the data need for train
        image_dir = vid_info["img_dir"]
        # image_dir = image_dir.replace('/media/Diske/Datasets/', '/date/dataset_share/iper/')
        images_names = vid_info["images"]
        alphas_paths = vid_info["alpha_paths"]

        images = cv_utils.read_cv2_img(os.path.join(image_dir, images_names[pair_ids[1]]))
        # uv_imgs = cv_utils.read_cv2_img(os.path.join(image_dir[:-6], 'uv_img/'+images_names[pair_ids[0]][:-4]+'.jpg'))
        masks = cv_utils.read_mask(alphas_paths[pair_ids[1]], self._opt.image_size)
        ref_uvs = cv_utils.read_cv2_img(os.path.join(image_dir[:-6], 'ref_uv/'+images_names[0][:-4]+'.jpg'))
        extra_data_path = os.path.join(image_dir[:-6], 'extra_data/'+images_names[pair_ids[1]][:-4]+'.pkl')
        extra_data = load_pickle_file(extra_data_path)
        Ts = extra_data['Tuv2src']
        head_bboxs = extra_data['head_bbox']
        # uv_index = (cv2.imread(os.path.join(image_dir[:-6], 'uv_indx/'+images_names[pair_ids[1]][:-4]+'.jpg'), 0)[None]).astype(np.int16)
        # uv_index = ((np.sign(uv_index-127)+1)/2).astype(np.uint8).repeat(3,axis=0)
        G_tsf = cv2.imread(os.path.join(image_dir[:-6], 'input_G_tsf/'+images_names[pair_ids[1]][:-4]+'.jpg'))
        # uv_index = cv_utils.read_cv2_img(os.path.join(image_dir[:-6], 'uv_indx/'+images_names[pair_ids[1]][:-4]+'.jpg'))
        return images, masks, Ts, head_bboxs,G_tsf,ref_uvs


    def __getitem__(self, index):
        
        vid_info = self._vids_info[index % self._num_videos]

        # images, masks, uv_imgs, ref_uvs, Ts, head_bboxs, uv_index, G_tsf= self._load_pairs(vid_info)
        images, masks, Ts, head_bboxs,G_tsf,ref_uvs= self._load_pairs(vid_info)
        # pack data
        sample = {
            "images": images,
            "masks": masks,
            # "uv_imgs":uv_imgs,
            "ref_uvs":ref_uvs,
            "Ts":Ts,
            "head_bboxs":head_bboxs,
            # "uv_index":uv_index,
            "G_tsf":G_tsf,
        }

        sample = self._transform(sample)
        # print(sample)
        return sample

class ProcessedVideoDataset_for_cross_imitation(VideoDataset):

    def __init__(self, opt, is_for_train):
        super(ProcessedVideoDataset_for_cross_imitation, self).__init__(opt, is_for_train)
        self._dir_lenth = [] 
        self._read_vids_info()

    def _read_single_dataset(self, data_dir, txt_path):
        with open(txt_path, "r") as reader:

            for line in tqdm(reader.readlines()):

                vid_name = line.rstrip()
                vid_info_path = os.path.join(data_dir, "primitives", vid_name, "processed", "vid_info.pkl")

                # print(vid_info_path)
                vid_info = load_pickle_file(vid_info_path)
                vid_info = read_src_infos(vid_info, num_source=self._opt.num_source, ignore_bg=True)

                self._vids_info.append(vid_info)
                self._num_videos += 1
                self._dataset_size += vid_info["length"]
                self._dir_lenth.append(self._dataset_size) 
                # if self._dataset_size == 579:
                #     break

    def _read_vids_info(self):

        data_dir_list = self._opt.dataset_dirs
        for data_dir in data_dir_list:

            if self._is_for_train:
                if self._opt.debug:
                    txt_path = '/date/dataset_share/iper/iper/debug.txt' # this is for debug
                else:
                    txt_path = os.path.join(data_dir, "all.txt")
                # 
            else:
                if self._opt.debug:
                    txt_path = '/date/dataset_share/iper/iper/debug_val.txt' # this is for debug
                else:
                    txt_path = os.path.join(data_dir, "val.txt")

            self._read_single_dataset(data_dir, txt_path)

    def load_imgs(self, vid_info, dir_index):
        # some annotations for smpls
        # here smpls contains 85, of which first 3 is camera, later 72 is pose, final 10 is shape
        images = []
        masks = []
        smpls = []
        # for source image get
        svid_info_path = os.path.join(self._opt.source_img_folder, 'vid_info.pkl')
        svid_info = load_pickle_file(svid_info_path)
        svid_info = read_src_infos(svid_info, num_source=self._opt.num_source, ignore_bg=True)

        smpl_source = svid_info["smpls"][self._opt.source_img_id]
        mask_source = 1.0 - cv_utils.read_mask(svid_info["alpha_paths"][self._opt.source_img_id], self._opt.image_size)
        source_image = cv_utils.read_cv2_img(os.path.join(svid_info["img_dir"], svid_info["images"][self._opt.source_img_id]))
        #
        smpls.append(smpl_source)
        images.append(source_image)
        masks.append(mask_source)
        # for target
        smpl_tgt = vid_info["smpls"][dir_index]
        smpls.append(smpl_tgt)
        image_dir = vid_info["img_dir"]
        images_names = vid_info["images"]
        alphas_paths = vid_info["alpha_paths"]
        images_path = os.path.join(image_dir, images_names[dir_index])
        # images.append(cv_utils.read_cv2_img(os.path.join(image_dir, images_names[0])))
        images.append(cv_utils.read_cv2_img(images_path))
        # front is 0, and background is 1
        # mask0 = cv_utils.read_mask(alphas_paths[0], self._opt.image_size)
        # mask0 = 1.0 - mask0
        mask1 = cv_utils.read_mask(alphas_paths[dir_index], self._opt.image_size)
        mask1 = 1.0 - mask1
        # masks.append(mask0)
        masks.append(mask1)

        # here we need the corresponding uv image
        # print(os.path.join(self._opt.source_img_folder, 'uv_img/'+svid_info["images"][self._opt.source_img_id][:-4]+'.jpg'))
        # uv_imgs = cv_utils.read_cv2_img("/date/dataset_share/iper/iper/primitives/001/31/1/processed/uv_img/000.jpg")
        uv_imgs = cv_utils.read_cv2_img(os.path.join(self._opt.source_img_folder, 'uv_img/'+svid_info["images"][self._opt.source_img_id][:-4]+'.jpg'))
        return images, smpls, masks, uv_imgs

    def __getitem__(self, index):
        for i in range(self._num_videos):
            if index < self._dir_lenth[0]:
                num = 0
                dir_index = index
                break
            elif index < self._dir_lenth[i]:
                num = i
                dir_index = index-self._dir_lenth[i-1]
                break

        vid_info = self._vids_info[num]

        images, smpls, masks, uv_imgs = self.load_imgs(vid_info, dir_index)

        # pack data
        sample = {
            "images": images,
            "smpls": smpls,
            "masks": masks,
            "uv_imgs": uv_imgs
        }

        sample = self._transform(sample)
        return sample

class ProcessedVideoDataset_for_cross_imitation_shuangmian(VideoDataset):

    def __init__(self, opt, is_for_train):
        super(ProcessedVideoDataset_for_cross_imitation_shuangmian, self).__init__(opt, is_for_train)
        self._dir_lenth = [] 
        self._read_vids_info()

    def _read_single_dataset(self, data_dir, txt_path):
        with open(txt_path, "r") as reader:

            for line in tqdm(reader.readlines()):

                vid_name = line.rstrip()
                vid_info_path = os.path.join(data_dir, "primitives", vid_name, "processed", "vid_info.pkl")

                # print(vid_info_path)
                vid_info = load_pickle_file(vid_info_path)
                vid_info = read_src_infos(vid_info, num_source=self._opt.num_source, ignore_bg=True)

                self._vids_info.append(vid_info)
                self._num_videos += 1
                self._dataset_size += vid_info["length"]
                self._dir_lenth.append(self._dataset_size) 
                # if self._dataset_size == 579:
                #     break

    def _read_vids_info(self):

        data_dir_list = self._opt.dataset_dirs
        for data_dir in data_dir_list:

            if self._is_for_train:
                if self._opt.debug:
                    txt_path = '/date/dataset_share/iper/iper/debug.txt' # this is for debug
                else:
                    txt_path = os.path.join(data_dir, "all.txt")
                # 
            else:
                if self._opt.debug:
                    txt_path = '/date/dataset_share/iper/iper/debug_val.txt' # this is for debug
                else:
                    txt_path = os.path.join(data_dir, "val.txt")

            self._read_single_dataset(data_dir, txt_path)

    def load_imgs(self, vid_info, dir_index):
        # some annotations for smpls
        # here smpls contains 85, of which first 3 is camera, later 72 is pose, final 10 is shape
        images = []
        masks = []
        smpls = []
        # for source image get
        svid_info_path = os.path.join(self._opt.source_img_folder, 'vid_info.pkl')
        svid_info = load_pickle_file(svid_info_path)
        svid_info = read_src_infos(svid_info, num_source=self._opt.num_source, ignore_bg=True)

        smpl_source = svid_info["smpls"][self._opt.source_img_id]
        mask_source = 1.0 - cv_utils.read_mask(svid_info["alpha_paths"][self._opt.source_img_id], self._opt.image_size)
        source_image = cv_utils.read_cv2_img(os.path.join(svid_info["img_dir"], svid_info["images"][self._opt.source_img_id]))
        #
        smpls.append(smpl_source)
        images.append(source_image)
        masks.append(mask_source)
        # for target
        smpl_tgt = vid_info["smpls"][dir_index]
        smpls.append(smpl_tgt)
        image_dir = vid_info["img_dir"]
        images_names = vid_info["images"]
        alphas_paths = vid_info["alpha_paths"]
        images_path = os.path.join(image_dir, images_names[dir_index])
        # images.append(cv_utils.read_cv2_img(os.path.join(image_dir, images_names[0])))
        images.append(cv_utils.read_cv2_img(images_path))
        # front is 0, and background is 1
        # mask0 = cv_utils.read_mask(alphas_paths[0], self._opt.image_size)
        # mask0 = 1.0 - mask0
        mask1 = cv_utils.read_mask(alphas_paths[dir_index], self._opt.image_size)
        mask1 = 1.0 - mask1
        # masks.append(mask0)
        masks.append(mask1)

        # here we need the corresponding uv image
        # print(os.path.join(self._opt.source_img_folder, 'uv_img/'+svid_info["images"][self._opt.source_img_id][:-4]+'.jpg'))
        # uv_imgs = cv_utils.read_cv2_img("/date/dataset_share/iper/iper/primitives/001/31/1/processed/uv_img/000.jpg")
        # uv_imgs = cv_utils.read_cv2_img(os.path.join(self._opt.source_img_folder, 'ref_uv/'+svid_info["images"][self._opt.source_img_id][:-4]+'.jpg'))
        uv_imgs = cv_utils.read_cv2_img(os.path.join(self._opt.source_img_folder, 'uv_img/'+svid_info["images"][self._opt.source_img_id][:-4]+'.jpg'))
        return images, smpls, masks, uv_imgs

    def __getitem__(self, index):
        for i in range(self._num_videos):
            if index < self._dir_lenth[0]:
                num = 0
                dir_index = index
                break
            elif index < self._dir_lenth[i]:
                num = i
                dir_index = index-self._dir_lenth[i-1]
                break

        vid_info = self._vids_info[num]

        images, smpls, masks, uv_imgs = self.load_imgs(vid_info, dir_index)

        # pack data
        sample = {
            "images": images,
            "smpls": smpls,
            "masks": masks,
            "uv_imgs": uv_imgs
        }

        sample = self._transform(sample)
        return sample

