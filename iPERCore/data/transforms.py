# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF


class ImageTransformer(object):
    """
    Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        """
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is matched to output_size.
                            If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def process_img(self, images):

        if isinstance(images, list):
            resized_images = []
            for image in images:
                height, width = image.shape[0:2]

                if height != self.output_size or width != self.output_size:
                    image = cv2.resize(image, (self.output_size, self.output_size))

                image = image.astype(np.float32)
                image /= 255.0
                image = image * 2 - 1

                image = np.transpose(image, (2, 0, 1))

                resized_images.append(image)

            resized_images = np.stack(resized_images, axis=0)
            return resized_images
        else:
            height, width = images.shape[0:2]
            if height != self.output_size or width != self.output_size:
                images = cv2.resize(images, (self.output_size, self.output_size))

            image = images.astype(np.float32)
            image /= 255.0
            image = image * 2 - 1
            image = np.transpose(image, (2, 0, 1))
            return image

    def __call__(self, sample):
        sample["images"] = self.process_img(sample["images"])
        if "G_tsf" in sample.keys():
            sample["G_tsf"] = self.process_img(sample["G_tsf"])
        if "uv_imgs" in sample.keys():
            sample["uv_imgs"] = self.process_img(sample['uv_imgs'])
        if "ref_uvs" in sample.keys():
            sample["ref_uvs"] = self.process_img(sample["ref_uvs"])
        # if "uv_index" in sample.keys():
        #     uv_index = np.transpose(sample["uv_index"]/255, (2, 0, 1))
        #     sample["uv_index"] = uv_index.astype(np.uint8)
        return sample


class ImageNormalizeToTensor(object):
    """
    Rescale the image in a sample to a given size.
    """

    def __call__(self, image):
        # image = F.to_tensor(image)
        image = TF.to_tensor(image)
        image.mul_(2.0)
        image.sub_(1.0)
        return image


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        sample["images"] = torch.from_numpy(sample["images"]).float()
        # sample["smpls"] = torch.from_numpy(sample["smpls"]).float()
        # sample["Ts"] = torch.from_numpy(sample["Ts"]).float()
        # sample["head_bboxs"] = torch.from_numpy(sample["head_bboxs"])
        # sample["masks"] = torch.from_numpy(sample["masks"]).float()
        if "masks" in sample:
            sample["masks"] = torch.tensor(sample["masks"]).float()
        if "head_bboxs" in sample:
            sample["head_bboxs"] = torch.from_numpy(sample["head_bboxs"])
        if "Ts" in sample:
            sample["Ts"] = torch.from_numpy(sample["Ts"]).float()
        if "smpls" in sample:
            sample["smpls"] = torch.tensor(sample["smpls"]).float()
        if "uv_imgs" in sample:
            sample["uv_imgs"] = torch.from_numpy(sample["uv_imgs"]).float()
        if "ref_uvs" in sample:
            sample["ref_uvs"] = torch.from_numpy(sample["ref_uvs"]).float()
        if "G_tsf" in sample:
            sample["G_tsf"] = torch.from_numpy(sample["G_tsf"]).float()
        return sample

