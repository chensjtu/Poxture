# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

"""

Assume that we put the iPER data into $FashionVideo_root_dir

1. Download the FashionVideo dataset, https://vision.cs.ubc.ca/datasets/fashion/
    1.1 download the `fashion_train.txt` into $FashionVideo_root_dir:
        https://vision.cs.ubc.ca/datasets/fashion/resources/fashion_dataset/fashion_train.txt

    1.2 download the `fashion_test.txt` into $FashionVideo_root_dir:
        https://vision.cs.ubc.ca/datasets/fashion/resources/fashion_dataset/fashion_test.txt

    1.3 crawl each video in `fashion_train.txt`, as well as `fashion_test.txt` and
        save them into $FashionVideo_root_dir/videos

   The file structure of $FashionVideo_root_dir will be:

   $FashionVideo_root_dir:
        --fashion_train.txt
        --fashion_test.txt
        --videos:


2. Preprocess all videos in $FashionVideo_root_dir/videos.


3. Reorganize the processed data for evaluations, https://github.com/iPERDance/his_evaluators


"""

import os
import subprocess
import argparse
import glob
from tqdm import tqdm
import requests
import warnings

from iPERCore.services.options.options_setup import setup
from iPERCore.services.options.process_info import ProcessInfo
from iPERCore.tools.utils.filesio.persistence import mkdir
from iPERCore.services.preprocess import human_estimate, digital_deform


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="/date/dataset_share/iper/deepfashion", help="the root directory of iPER dataset.")
parser.add_argument("--gpu_ids", type=str, default="1", help="the gpu ids.")
parser.add_argument("--image_size", type=int, default=512, help="the image size.")
parser.add_argument("--model_id", type=str, default="FashionVideo_Preprocess", help="the renamed model.")
parser.add_argument("--Preprocess.Cropper.src_crop_factor", type=float, default=0, help="directly resize on iPER.")
parser.add_argument("--ref_path", type=str, default="", help="set this empty when preprocessing training dataset.")

args = parser.parse_args()
args.cfg_path = os.path.join("./assets", "configs", "deploy.toml")

FashionVideo_root_dir = mkdir(args.output_dir)
FashionVideo_video_dir = mkdir(os.path.join(FashionVideo_root_dir, "videos"))
FashionVideo_train_url_txt = os.path.join(FashionVideo_root_dir, "fashion_train.txt")
FashionVideo_test_url_txt = os.path.join(FashionVideo_root_dir, "fashion_test.txt")
TRAIN_list_txt = os.path.join(FashionVideo_root_dir, "train.txt")
TEST_list_txt = os.path.join(FashionVideo_root_dir, "val.txt")

TRAIN_URL = "https://vision.cs.ubc.ca/datasets/fashion/resources/fashion_dataset/fashion_train.txt"
TEST_URL = "https://vision.cs.ubc.ca/datasets/fashion/resources/fashion_dataset/fashion_test.txt"


def download_train_test_url_txt():
    global FashionVideo_root_dir, FashionVideo_train_url_txt, FashionVideo_test_url_txt, TRAIN_URL, TEST_URL

    r = requests.get(TRAIN_URL)
    with open(FashionVideo_train_url_txt, "wb") as f:
        f.write(r.content)

    print(f"1.1 download {TRAIN_URL} to {FashionVideo_train_url_txt}")

    r = requests.get(TEST_URL)
    with open(FashionVideo_test_url_txt, "wb") as f:
        f.write(r.content)

    print(f"1.2 download {TEST_URL} to {FashionVideo_test_url_txt}")


def crawl_videos(url_txt_file):
    """

    Args:
        url_txt_file (str): the txt file contains all video urls.

    Returns:

    """
    global FashionVideo_video_dir

    video_urls = []
    with open(url_txt_file, "r") as reader:

        # TODO, convert this to multi-thread or multi-process?
        for vid_url in tqdm(reader):
            vid_url = vid_url.rstrip()
            file_name = os.path.split(vid_url)[-1]

            video_path = os.path.join(FashionVideo_video_dir, file_name)
            if not os.path.exists(video_path):
                r = requests.get(vid_url)
                with open(video_path, "wb") as f:
                    f.write(r.content)

            print(f"crawl {vid_url}")
            video_urls.append(vid_url)

    return video_urls


def extract_one_video(video_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # os.system('ffmpeg -i %s -start_number 0 %s/frame%%08d.png > /dev/null 2>&1' % (video_path, save_dir))

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-start_number", "0",
        "{save_dir}/frame_%08d.png".format(save_dir=save_dir),
        "-loglevel", "quiet"
    ]

    print(" ".join(cmd))
    subprocess.run(cmd)


def get_video_dirs(txt_file):
    vid_names = []
    with open(txt_file, "r") as reader:
        for line in reader:
            line = line.rstrip()
            vid_names.append(line)

    return vid_names


def prepare_src_path():
    global FashionVideo_video_dir

    template_path = "path?={path},name?={name}"

    src_paths = []
    for vid_name in os.listdir(FashionVideo_video_dir):

        # if vid_name != "A19FRhRmb-S.mp4":
        # if vid_name != "91+uwOT1POS.mp4":
        if vid_name != "A1wwPTTzVGS.mp4":
            continue

        vid_path = os.path.join(FashionVideo_video_dir, vid_name)
        assert os.path.exists(vid_path)

        path = template_path.format(path=vid_path, name=vid_name)
        src_paths.append(path)

        print(path)

    return src_paths


def get_video_names(video_urls):
    global FashionVideo_video_dir

    video_names = []
    for vid_url in video_urls:
        file_name = os.path.split(vid_url)[-1]
        vid_path = os.path.join(FashionVideo_video_dir, file_name)
        video_names.append(file_name)

        assert os.path.exists(vid_path), f"download {vid_url} failed."

    return video_names


def download():
    global FashionVideo_train_url_txt, FashionVideo_test_url_txt, TRAIN_list_txt, TEST_list_txt

    download_train_test_url_txt()

    train_urls = crawl_videos(FashionVideo_train_url_txt)
    test_urls = crawl_videos(FashionVideo_test_url_txt)

    train_names = get_video_names(train_urls)
    test_names = get_video_names(test_urls)

    same_set = set(train_names) & set(test_names)
    print(same_set)

    with open(TRAIN_list_txt, "w") as writer:
        train_lines = "\n".join(train_names)
        writer.writelines(train_lines)

    with open(TEST_list_txt, "w") as writer:
        test_lines = "\n".join(test_names)
        writer.writelines(test_lines)


def process_data():
    # 1. preprocess
    src_paths = prepare_src_path()

    args.src_path = "|".join(src_paths)

    print(args.src_path)

    # set this as empty when preprocessing the training dataset.
    args.ref_path = ""

    cfg = setup(args)

    # 1. human estimation, including 2D pose, tracking, 3D pose, parsing, and front estimation.
    human_estimate(opt=cfg)

    # 2. digital deformation.
    digital_deform(opt=cfg)

    # 3. check
    meta_src_proc = cfg.meta_data["meta_src"]
    invalid_meta_process = []
    for meta_proc in meta_src_proc:
        process_info = ProcessInfo(meta_proc)
        process_info.deserialize()

        # check it has been processed successfully
        if not process_info.check_has_been_processed(process_info.vid_infos, verbose=False):
            invalid_meta_process.append(meta_proc)

    num_invalid = len(invalid_meta_process)
    if num_invalid > 0:
        for meta_proc in invalid_meta_process:
            print(f"invalid meta proc {meta_proc}")

    else:
        print(f"process successfully.")


def reorganize():
    # TODO
    pass


if __name__ == "__main__":
    download()
    process_data()
    # reorganize()
