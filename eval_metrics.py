import torch
import numpy as np
import unittest
import cv2

from metrics import register_metrics


DEVICE = torch.device("cuda:0")


def load_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32, copy=False)
    image /= 255
    image = torch.as_tensor(image)
    return image


class MetricTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.paired_metric_dict = register_metrics(types=("ssim", "psnr", "lps"), device=DEVICE)
        # cls.unpaired_metric_dict = register_metrics(
        #     types=("is", "fid", "PCB-CS-reid", "PCB-freid", "OS-CS-reid", "OS-freid"),
        #     device=DEVICE
        # )

        cls.unpaired_metric_dict = register_metrics(
            types=("is", "fid", "SSPE", "OS-CS-reid", "OS-freid"),
            device=DEVICE
        )

        cls.face_metric_dict = register_metrics(
            types=("face-CS", ),
            device=DEVICE
        )

    # def test_01_paired_metrics(self):
    #     bs = 5
    #     image_size = 512
    #     preds_imgs = np.random.rand(bs, 3, image_size, image_size)
    #     preds_imgs *= 255
    #     preds_imgs = preds_imgs.astype(np.uint8)
    #     ref_imgs = np.copy(preds_imgs)
    #
    #     preds_imgs = torch.tensor(preds_imgs).float()
    #     ref_imgs = torch.tensor(ref_imgs).float()
    #     ssim_score = self.paired_metric_dict["ssim"].calculate_score(preds_imgs, ref_imgs)
    #     psnr_score = self.paired_metric_dict["psnr"].calculate_score(preds_imgs, ref_imgs)
    #     lps_score = self.paired_metric_dict["lps"].calculate_score(preds_imgs, ref_imgs)
    #
    #     print("ssim score = {}".format(ssim_score))
    #     print("psnr score = {}".format(psnr_score))
    #     print("lps score = {}".format(lps_score))
    #
    #     self.assertEqual(ssim_score, 1.0)
    #     self.assertEqual(psnr_score, np.inf)
    #     self.assertEqual(lps_score, 0.0)
    #
    # def test_02_unpaired_metrics(self):
    #     bs = 5
    #     image_size = 512
    #     preds_imgs = np.random.rand(bs, 3, image_size, image_size)
    #     preds_imgs *= 255
    #     preds_imgs = preds_imgs.astype(np.uint8)
    #
    #     ref_imgs = np.random.rand(bs, 3, image_size, image_size)
    #     ref_imgs *= 255
    #     ref_imgs = ref_imgs.astype(np.uint8)
    #
    #     preds_imgs = torch.tensor(preds_imgs).float()
    #     ref_imgs = torch.tensor(ref_imgs).float()
    #
    #     inception_score = self.unpaired_metric_dict["is"].calculate_score(preds_imgs)
    #     fid_score = self.unpaired_metric_dict["fid"].calculate_score(preds_imgs, ref_imgs)
    #     sspe = self.unpaired_metric_dict["SSPE"].calculate_score(preds_imgs, ref_imgs)
    #     os_cs_reid = self.unpaired_metric_dict["OS-CS-reid"].calculate_score(preds_imgs, ref_imgs)
    #     os_freid = self.unpaired_metric_dict["OS-freid"].calculate_score(preds_imgs, ref_imgs)
    #
    #     # pcb_cs_reid = self.unpaired_metric_dict["PCB-CS-reid"].calculate_score(preds_imgs, ref_imgs)
    #     # pcb_freid = self.unpaired_metric_dict["PCB-freid"].calculate_score(preds_imgs, ref_imgs)
    #
    #     print("inception score = {}".format(inception_score))
    #     print("fid score = {}".format(fid_score))
    #     print("ssp error = {}".format(sspe))
    #     print("OS-Cosine Similarity = {}".format(os_cs_reid))
    #     print("OS-freid = {}".format(os_freid))
    #
    #     # print("PCB-Cosine Similarity = {}".format(pcb_cs_reid))
    #     # print("PCB-freid = {}".format(pcb_freid))

    def test_03_face_metric_all_have_face(self):
        pred_img_list = [
            "./data/pred_00000000.jpg",
            "./data/pred_00000114.jpg",
            "./data/pred_00000423.jpg",
            "./data/pred_00000175.jpg"
        ]

        ref_img_list = [
            "./data/pred_00000423.jpg",
            "./data/pred_00000114.jpg",
            "./data/pred_00000175.jpg",
            "./data/pred_00000000.jpg"
        ]

        pred_imgs = []
        for img_path in pred_img_list:
            img = load_image(img_path)
            pred_imgs.append(img)
        pred_imgs = torch.stack(pred_imgs)

        ref_imgs = []
        for img_path in ref_img_list:
            img = load_image(img_path)
            ref_imgs.append(img)
        ref_imgs = torch.stack(ref_imgs)

        face_cs = self.face_metric_dict["face-CS"].calculate_score(pred_imgs, ref_imgs)

        print("face-cs", face_cs)


if __name__ == '__main__':
    unittest.main()
