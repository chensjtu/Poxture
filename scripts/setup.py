# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
from os.path import exists
import platform
from setuptools import find_packages, setup
import subprocess
import sys
import re

TORCH_DIST = "https://download.pytorch.org/whl/torch_stable.html"
MMCV_DIST = "https://download.openmmlab.com/mmcv/dist/index.html"

PIP_VERSION = "20.2.4"

PRECOMPILED_TORCH_CUDA_PAIRS = {
    "1.7.0+cu110": {
        "torch": "1.7.0+cu110",
        "torchvision": "0.8.1+cu110",
        "mmcv-full": "1.2.0+torch1.7.0+cu110"
    },
    "1.7.0+cu102": {
        "torch": "1.7.0",
        "torchvision": "0.8.1",
        "mmcv-full": "1.2.0+torch1.7.0+cu102"
    },
    "1.7.0+cu101": {
        "torch": "1.7.0+cu101",
        "torchvision": "0.8.1+cu101",
        "mmcv-full": "1.2.0+torch1.7.0+cu101"
    },
    "1.6.0+cu102": {
        "torch": "1.6.0",
        "torchvision": "0.7.0",
        "mmcv-full": "1.1.5+torch1.6.0+cu102"
    },
    "1.6.0+cu101": {
        "torch": "1.6.0+cu101",
        "torchvision": "0.7.0+cu101",
        "mmcv-full": "1.1.5+torch1.6.0+cu101"
    },
    "1.5.0+cu102": {
        "torch": "1.5.0",
        "torchvision": "0.6.0",
        "mmcv-full": "1.2.0+torch1.5.0+cu102"
    },
    "1.5.0+cu101": {
        "torch": "1.5.0+cu101",
        "torchvision": "0.6.0+cu101",
        "mmcv-full": "1.2.0+torch1.5.0+cu101"
    }
}

WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS = {
    "1.6.0+cu102": {
        "torch": "https://download.pytorch.org/whl/cu102/torch-1.6.0-cp{PYTHON_VERSION}-cp{PYTHON_ABI_VERSION}-win_amd64.whl",
        "torchvision": "https://download.pytorch.org/whl/cu102/torchvision-0.7.0-cp{PYTHON_VERSION}-cp{PYTHON_ABI_VERSION}-win_amd64.whl",
        "mmcv-full": "1.1.5+torch1.6.0+cu102"
    },
    "1.6.0+cu101": {
        "torch": "https://download.pytorch.org/whl/cu101/torch-1.6.0%2Bcu101-cp{PYTHON_VERSION}-cp{PYTHON_ABI_VERSION}-win_amd64.whl",
        "torchvision": "https://download.pytorch.org/whl/cu101/torchvision-0.7.0%2Bcu101-cp{PYTHON_VERSION}-cp{PYTHON_ABI_VERSION}-win_amd64.whl",
        "mmcv-full": "1.1.5+torch1.6.0+cu101"
    }
}


def get_cuda_version() -> float:
    """
        Get the cuda version of the system, make sure that the $CUDA_HOME or the $CUDA_PATH has been added into the system path.
        Here the cuda version is aaa.b.ccc, and only return the version with aaa.b
    Returns:
        version (float): the cuda version.

    """

    # In windows, the cuda_install.exe will set the `CUDA_PATH` to the system environmental variables.
    if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]

    assert "CUDA_HOME" in os.environ, r"Cannot find the $CUDA_HOME in the environments. Please manually install the " \
                                      r"CUDA >= 10.1, and set the $CUDA_HOME environment variable."

    # e.g. "CUDA Version 10.1.243", "CUDA Version 10.0.130"
    cuda_version_file = os.path.join(os.environ["CUDA_HOME"], "version.txt")

    if not os.path.exists(cuda_version_file):
        raise FileNotFoundError(f"Cannot read cuda version file {cuda_version_file}")

    with open(cuda_version_file) as f:
        version_str = f.readline().replace("\n", "").replace("\r", "")

    # "CUDA Version 10.1.243" -> ["CUDA", "Version", "10.1.243"] -> "10.1.243"
    version = version_str.split(" ")[2]

    # "10.1.243" -> "10.1" -> 10.1
    version = float(".".join(version.split(".")[0:2]))

    assert version >= 10.1, f"CUDA Version {version} <= 10.1. Please manually install the CUDA >= 10.1"

    return version


def get_python_version() -> str:
    """
        Get the python version. The python version is aaa.b.c, and it only returns aaa.b
    Returns:
        version (str):
    """

    version = str(platform.python_version())
    version = "".join(version.split(".")[0:2])

    assert "36" <= version <= "38", f"Currently, it only support the python version with 3.6.+, 3.7.+, 3.8.+"

    return version


def platform_dependencies():
    """Parse the pre-complied consistent versions of torch, torchvision, mmcv, and CUDA.
    The torch version must >= 1.6.0, and the CUDA version must >= 10.1.
    Currently, it only supports Linux and Windows.
    If the platform is Linux, we will use torch 1.7.0 + CUDA Version.
    Otherwise if the platform is windows, we will use torch 1.6.0 + CUDA VERSION.

    Returns:
        List[List[str]]: list of setup requirements items.

    """
    global TORCH_DIST, MMCV_DIST, PRECOMPILED_TORCH_CUDA_PAIRS, WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS

    cuda_version = get_cuda_version()
    cuda_version_str = str(cuda_version).replace(".", "")

    packages = []

    if platform.system().lower() == "windows":
        python_version = get_python_version()
        python_abi_version = python_version
        if python_version != "38":
            python_abi_version += "m"

        torch_cuda_version = f"1.6.0+cu{cuda_version_str}"
        numpy_version = "numpy==1.19.3"

        assert torch_cuda_version in WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS, \
            f"There is no pre-complied pytorch 1.6.0 with CUDA {cuda_version}, " \
            f"and you might need to install pytorch 1.6.0 with CUDA {cuda_version} from source."

        torch_link = WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["torch"] \
            .format(PYTHON_VERSION=python_version, PYTHON_ABI_VERSION=python_abi_version)
        torchvision_link = WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["torchvision"] \
            .format(PYTHON_VERSION=python_version, PYTHON_ABI_VERSION=python_abi_version)

        mmcv_version = WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["mmcv-full"]

        packages.append([f"{torch_link}", "-f", TORCH_DIST])
        packages.append([f"{torchvision_link}", "-f", TORCH_DIST])
        packages.append([f"mmcv-full=={mmcv_version}", "-f", MMCV_DIST])
        packages.append(numpy_version)

    elif platform.system().lower() == "linux":
        torch_cuda_version = f"1.7.0+cu{cuda_version_str}"
        numpy_version = "numpy>=1.19.3"

        assert torch_cuda_version in PRECOMPILED_TORCH_CUDA_PAIRS, \
            f"There is no pre-complied pytorch 1.7.0 with CUDA {cuda_version}, " \
            f"and you might need to install pytorch 1.7.0 with CUDA {cuda_version} from source."

        torch_version = PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["torch"]
        torchvision_version = PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["torchvision"]
        mmcv_version = PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["mmcv-full"]

        packages.append([f"torch=={torch_version}", "-f", TORCH_DIST])
        packages.append([f"torchvision=={torchvision_version}", "-f", TORCH_DIST])
        packages.append([f"mmcv-full=={mmcv_version}", "-f", MMCV_DIST])
        packages.append(numpy_version)

    else:
        raise ValueError(f"Currently it only supports 'windows' and 'linux'.")

    return packages


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.
    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())

    return packages


# 1. install torch, torchvision, and mmcv firstly.
torch_torchvision_mmcv = platform_dependencies()

# 2. other installed requires
install_requires = parse_requirements("requirements/runtime.txt")
# 3. build requires
build_requiers = parse_requirements("requirements/build.txt")

# 4. pip install all of them
all_requires = [[f"pip=={PIP_VERSION}"]] + torch_torchvision_mmcv + install_requires + build_requiers

pip_executable = [sys.executable, "-m", "pip", "install"]

for package_line in all_requires:
    if isinstance(package_line, str):
        package_line = [package_line]

    pip_install_line = pip_executable + package_line

    print(" ".join(pip_install_line))
    subprocess.run(pip_install_line)

# 5. setup iPERCore
setup(
    name="iPERCore",
    version="0.1.1",
    author="Wen Liu, and Zhixin Piao",
    author_email="liuwen@shanghaitech.edu.cn",
    url="https://github.com/iPERDance/iPERCore",
    description="The core of impersonator++.",
    packages=find_packages(exclude=("assets",)),
    python_requires=">=3.6, <=3.8",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "run_imitator = iPERCore.services.run_imitator:run_imitator",
        ]
    }
)
