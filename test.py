import numpy as np
import cv2
from iPERCore.tools.utils.visualizers.visdom_visualizer import VisdomVisualizer

visualizer = VisdomVisualizer(
    env="test_train",
    ip="http://localhost",  # need to be replaced.
    port=8097  # need to be replaced.
)

mask0 = cv2.imread("/media/Diskf/Datasets/iper/primitives/006/1/1/processed/uv_indx/000.jpg", 0)

mask1 = cv2.imread("/media/Diskf/Datasets/iper/primitives/006/1/1/processed/uv_indx/000.jpg")

cv2.imshow("mask0",mask0)
cv2.waitKey(3000)

cv2.imshow("mask1",mask1)
cv2.waitKey(3000)

visualizer.vis_named_img("ppp", mask0[None, None]/255)
