# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import os.path as osp
import platform
import argparse
import subprocess
import sys
import time
import ipdb

from iPERCore.services.options.options_setup import setup
from iPERCore.services.run_imitator import run_imitator


###############################################################################################
##                   Setting
###############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_ids", type=str, default="0", help="the gpu ids.")
parser.add_argument("--image_size", type=int, default=512, help="the image size.")
parser.add_argument("--num_source", type=int, default=2, help="the number of sources.")
parser.add_argument("--output_dir", type=str, default="./results", help="the output directory.")
parser.add_argument("--assets_dir", type=str, default="./assets",
                    help="the assets directory. This is very important, and there are the configurations and "
                         "the all pre-trained checkpoints")

src_path_format = """the source input information. All source paths and it supports multiple paths,
    uses "|" as the separator between all paths.
    The format is "src_path_1|src_path_2|src_path_3".
    Each src_input is "path?=path1,name?=name1,bg_path?=bg_path1".
    It must contain "path". If "name" and "bg_path" are empty, they will be ignored.

    The "path" could be an image path, a path of a directory contains source images, and a video path.

    The "name" is the rename of this source input, if it is empty, we will ignore it,
    and use the filename of the path.

    The "bg_path" is the actual background path if provided, otherwise we will ignore it.

    There are several examples of formated source paths,

    1. "path?=path1,name?=name1,bg_path?=bg_path1|path?=path2,name?=name2,bg_path?=bg_path2",
    this input will be parsed as [{path: path1, name: name1, bg_path:bg_path1},
    {path: path2, name: name2, bg_path: bg_path2}];

    2. "path?=path1,name?=name1|path?=path2,name?=name2", this input will be parsed as
    [{path: path1, name:name1}, {path: path2, name: name2}];

    3. "path?=path1", this input will be parsed as [{path: path1}].

    4. "path1", this will be parsed as [{path: path1}].
"""

parser.add_argument("--model_id", type=str, default=f"model_{str(time.time())}", help="the renamed model.")
parser.add_argument("--src_path", type=str, required=True, help=src_path_format)

ref_path_format = """the reference input information. All reference paths. It supports multiple paths,
    and uses "|" as the separator between all paths.
    The format is "ref_path_1|ref_path_2|ref_path_3".
    Each ref_path is "path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=150".
    It must contain "path", and others could be empty, and they will be ignored.
    
    The "path" could be an image path, a path of a directory contains source images, and a video path.
    
    The "name" is the rename of this source input, if it is empty, we will ignore it,
    and use the filename of the path.
    
    The "audio" is the audio path, if it is empty, we will ignore it. If the "path" is a video,
    you can ignore this, and we will firstly extract the audio information of this video (if it has audio channel).
    
    The "fps" is fps of the final outputs, if it is empty, we will set it as the default fps 25.
    
    The "pose_fc" is the smooth factor of the temporal poses. The smaller of this value, the smoother of the
    temporal poses. If it is empty, we will set it as the default 300. In the most cases, using the default
    300 is enough, and if you find the poses of the outputs are not stable, you can decrease this value.
    Otherwise, if you find the poses of the outputs are over stable, you can increase this value.
    
    The "cam_fc" is the smooth factor of the temporal cameras (locations in the image space). The smaller of
    this value, the smoother of the locations in sequences. If it is empty, we will set it as the default 150.
    In the most cases, the default 150 is enough.
    
    1. "path?=path1,name?=name1,audio?=audio_path1,fps?=30,pose_fc?=300,cam_fc?=150|
        path?=path2,name?=name2,audio?=audio_path2,fps?=25,pose_fc?=450,cam_fc?=200",
        this input will be parsed as
        [{path: path1, name: name1, audio: audio_path1, fps: 30, pose_fc: 300, cam_fc: 150},
         {path: path2, name: name2, audio: audio_path2, fps: 25, pose_fc: 450, cam_fc: 200}]
    
    2. "path?=path1,name?=name1, pose_fc?=450|path?=path2,name?=name2", this input will be parsed as
    [{path: path1, name: name1, fps: 25, pose_fc: 450, cam_fc: 150},
     {path: path2, name: name2, fps: 25, pose_fc: 300, cam_fc: 150}].
    
    3. "path?=path1|path?=path2", this input will be parsed as
    [{path: path1, fps:25, pose_fc: 300, cam_fc: 150}, {path: path2, fps: 25, pose_fc: 300, cam_fc: 150}].
    
    4. "path1|path2", this input will be parsed as
    [{path: path1, fps:25, pose_fc: 300, cam_fc: 150}, {path: path2, fps: 25, pose_fc: 300, cam_fc: 150}].
    
    5. "path1", this will be parsed as [{path: path1, fps: 25, pose_fc: 300, cam_fc: 150}].
"""
parser.add_argument("--ref_path", type=str, default="", help=ref_path_format)

# args = parser.parse_args()
args, extra_args = parser.parse_known_args()

# symlink from the actual assets directory to this current directory
work_asserts_dir = os.path.join("./assets")
if not os.path.exists(work_asserts_dir):
    os.symlink(osp.abspath(args.assets_dir), osp.abspath(work_asserts_dir),
               target_is_directory=(platform.system() == "Windows"))

args.cfg_path = osp.join(work_asserts_dir, "configs", "deploy.toml")


if __name__ == "__main__":
    # # run imitator
    # cfg = setup(args)
    # run_imitator(cfg)

    # or use the system call wrapper
    cmd = [
        sys.executable, "-m", "iPERCore.services.run_imitator",
        "--cfg_path", args.cfg_path,
        "--gpu_ids", args.gpu_ids,
        "--image_size", str(args.image_size),
        "--num_source", str(args.num_source),
        "--output_dir", args.output_dir,
        "--model_id", args.model_id,
        "--src_path", args.src_path,
        "--ref_path", args.ref_path
    ]

    cmd += extra_args
    subprocess.call(cmd)
