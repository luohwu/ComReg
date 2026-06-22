# CUDA 12.8 base: required for Blackwell (RTX 5090, sm_120) — both the
# prebuilt PyTorch kernels and the nvcc that compiles the local extensions.
# Still fully supports Ada (RTX 4090, sm_89) and the cluster's Ampere/Hopper GPUs.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

WORKDIR /workspace

# Arch list drives how the local CUDA extensions (chamfer_dist, pointnet2_ops_lib)
# are compiled. 8.0 A100, 8.6 A40/3090, 8.9 RTX 4090, 9.0 H100, 12.0 RTX 5090.
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps in one layer + cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgl1-mesa-glx \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (cached downloads)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# Python deps in fewer layers (cached downloads).
# NOTE: spconv has no cu128 wheel; spconv-cu126 imports fine on CUDA 12.8 and JITs
# PTX on Blackwell. If sparse-conv ops error on the 5090 ("no kernel image is
# available"), build spconv from source with CUMM_CUDA_ARCH_LIST="8.0;8.9;9.0;12.0".
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
      pyhocon open3d scipy pandas trimesh \
      PyMCubes \
      libigl \
      opencv-python \
      python-igraph \
      spconv-cu126 \
      yacs h5py tensorboardX \
      comet_ml \
      py-cpuinfo \
      timm \
      einops \
      kornia \
      "pytorch-lightning==2.6.0" \
      "segmentation-models-pytorch==0.4.0" \
      "monai==1.4.0"\
      "scikit-image==0.25.0"

# Git installs (kept separate so they only rebuild if these lines change)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
      "git+https://github.com/cong-yi/DualMesh-UDF" \
      "git+https://github.com/sbarratt/torch_interpolations.git"

# Local extensions last (avoid invalidating earlier layers)
COPY extensions ./extensions

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation ./extensions/chamfer_dist \
 && pip install --no-build-isolation ./extensions/pointnet2_ops_lib
