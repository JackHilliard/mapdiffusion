# MapDiffusion (IROS'25) — https://github.com/SonyResearch/MapDiffusion
#
# Diverges from README.md's install steps (conda py3.8, torch 1.9.0+cu111,
# mmcv-full 1.6.0): that stack predates CUDA 11.8, the first CUDA release
# with real Hopper (H100, sm_90) support. Bumped to torch 2.1.0/CUDA 11.8 +
# mmcv-full 1.7.2, following the same fix already applied to the sibling
# MapTRv2 and GeMap codebases' Dockerfiles. mmdet/mmsegmentation/mmdet3d
# keep the versions the README already asks for (2.28.2 / 0.30.0 /
# v1.0.0rc6), which all cover mmcv 1.7.x.
ARG PYTORCH="2.1.0"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# 8.6 covers Ampere (e.g. RTX 3070/3090); 9.0+PTX covers Hopper (H100) and,
# via PTX forward-compat JIT, anything newer released after this image (e.g.
# Blackwell RTX 50-series).
ENV TORCH_CUDA_ARCH_LIST="8.6 9.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ninja-build \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# mmcv-full is the ONLY compiled CUDA dependency in this stack, which makes
# it the whole H100 story here: mmdetection3d v1.0.0rc6 ships no CUDA
# extensions of its own (its setup.py has no ext_modules -- every op in
# mmdet3d/ops/__init__.py is re-exported from mmcv.ops, including the
# spconv used by sparse_block), and MapDiffusion itself has no custom ops
# at all. Every CUDA kernel this repo executes outside torch's own -- the
# deformable attention that BEVFormer's encoder and the map decoder are
# built on (ms_deform_attn), sigmoid_focal_loss, nms, voxelization -- comes
# out of this one build.
#
# So it MUST be built from source, not installed from OpenMMLab's prebuilt
# wheel index. Their prebuilt cu118/torch2.1.0 wheel was compiled with
# OpenMMLab's own (unknown, non-configurable) TORCH_CUDA_ARCH_LIST, which
# does not include Hopper (sm_90) and has no +PTX fallback baked in -- this
# is invisible on Ampere and only surfaces on an actual H100 as
# `RuntimeError: CUDA error: no kernel image is available for execution on
# the device` the first time one of those ops runs. Building from source
# makes mmcv's setup.py pick up this Dockerfile's own TORCH_CUDA_ARCH_LIST
# (set above, "8.6 9.0+PTX").
#
# 1.7.2 rather than the README's 1.6.0: 1.7.x is the first mmcv 1.x line
# that builds and runs against torch 2.x, and it is the version the sibling
# MapTRv2/GeMap images are already proven on. mmdet 2.28.2 and
# mmsegmentation 0.30.0 (both as per the README) declare an mmcv upper
# bound of 1.8.0, so both cover it; mmdet3d is patched below.
RUN MMCV_WITH_OPS=1 pip install --no-cache-dir --no-binary mmcv-full "mmcv-full==1.7.2"
# mmcv 1.7.2's single-GPU MMDataParallel path (mmcv.parallel.Scatter.forward,
# used by tools/test.py, tools/benchmark.py and tools/visualization) calls
# PyTorch's private torch.nn.parallel._functions._get_stream() with a raw
# int device id; torch>=2.x's version of that function requires a
# torch.device object instead (`AttributeError: 'int' object has no
# attribute 'type'`). Patch the installed file to wrap the device id.
RUN sed -i \
    "s/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device('cuda', device) if isinstance(device, int) else device) for device in target_gpus]/" \
    /opt/conda/lib/python3.10/site-packages/mmcv/parallel/_functions.py
RUN pip install --no-cache-dir mmdet==2.28.2 mmsegmentation==0.30.0

# mmdetection3d v1.0.0rc6, the version README.md pins. Not vendored in this
# repo, so it is cloned here and installed editable into the image
# alongside it.
RUN git clone --depth 1 -b v1.0.0rc6 \
        https://github.com/open-mmlab/mmdetection3d.git /workspace/mmdetection3d

# Three pins in mmdetection3d's own runtime.txt predate Python 3.10 and are
# unrelated to anything MapDiffusion uses (numba only backs the KITTI/voxel
# numpy paths, trimesh and networkx only the mesh visualizers), but
# `pip install -e .` reads that file as install_requires, so an
# uninstallable pin there fails the build outright:
#   * numba==0.53.0 has no Python 3.10 wheels and does not build from
#     source on 3.10 (it caps at 3.9). 0.56.4 is the first release with
#     cp310 wheels that still keeps numpy<1.24, i.e. the same numpy the
#     np.float/np.int aliases in mmdet/mmcv-full need (pinned below).
#   * networkx>=2.2,<2.3 and trimesh>=2.35.39,<2.35.40 are 2018-era sdists
#     with no wheels; unpin both.
RUN cd /workspace/mmdetection3d \
    && sed -i -e 's/^numba==0.53.0$/numba==0.56.4/' \
              -e 's/^networkx>=2.2,<2.3$/networkx/' \
              -e 's/^trimesh>=2.35.39,<2.35.40$/trimesh/' \
        requirements/runtime.txt
# mmdet3d v1.0.0rc6 asserts mmcv<=1.7.0 at import time (mmdet3d/__init__.py),
# which the 1.7.2 built above trips -- `AssertionError: MMCV==1.7.2 is used
# but incompatible`. 1.7.2 is a patch release over 1.7.0 with no API change
# mmdet3d touches (it is the torch-2.x build-fix line), so raise the bound
# rather than downgrade mmcv, which would give up the working torch 2.1
# build.
RUN sed -i "s/mmcv_maximum_version = '1.7.0'/mmcv_maximum_version = '1.7.2'/" \
        /workspace/mmdetection3d/mmdet3d/__init__.py
RUN cd /workspace/mmdetection3d && pip install --no-cache-dir -e .

RUN conda clean --all

WORKDIR /workspace/mapdiffusion
COPY . .

# av2 + nuscenes-devkit
RUN pip install --no-cache-dir -r requirements.txt

# Imported at module scope by the plugin but listed nowhere: einops
# (heads/MapDetectorHeadDiffuse.py) and prettytable (the vector/raster
# evaluators). IPython -- `from IPython import embed` sits at the top of
# several plugin/tools modules -- is already in the base image.
RUN pip install --no-cache-dir einops prettytable

# av2 imports numpy.typing (added in numpy 1.20) at module load time, as do
# several of this repo's own modules. Pin below 1.24 to keep the
# np.float/np.int aliases mmdet/mmcv-full may still use (removed in numpy
# 1.24), and to stay inside numba 0.56.4's numpy<1.24 bound.
RUN pip install --no-cache-dir "numpy==1.23.5"

# nuscenes-devkit pulls in shapely 2.x, which breaks nuScenes GT
# extraction: shapely 2.0 changed STRtree.query() to return integer indices
# instead of geometry objects, and
# plugin/datasets/map_utils/nuscmap_extractor.py's ped-crossing merge does
# `for o in tree.query(pgeom): index_by_id[id(o)]` -- against 2.x that
# raises KeyError on every lookup. Re-pin as the last install so it sticks.
#
# pip then warns "nuscenes-devkit 1.2.0 requires Shapely~=2.0.3, but you
# have shapely 1.8.5.post1". That warning is safe here and downgrading the
# devkit is NOT the fix: diffing 1.2.0's map_expansion/map_api.py against
# 1.1.11 (the last release that pins Shapely<2.0.0) shows its only
# shapely-related change is `for poly in polygons` -> `for poly in
# polygons.geoms`, which is the forward-compatible spelling shapely 1.8
# already supports -- i.e. the ~=2.0.3 pin is conservative metadata, not a
# real API dependency. Its other change, `plt.style.use('seaborn-whitegrid')`
# -> `'seaborn-v0_8-whitegrid'`, is why 1.1.11 is not an option: that style
# name was removed in matplotlib 3.8, so the older devkit would fail on
# import against the matplotlib mmdet installs here.
RUN pip install --no-cache-dir "shapely==1.8.5.post1"

# utils/dataset_viewer.py is designed to run outside the container (it
# imports no torch/mmdet3d and needs no GPU), but this lets it also run
# from inside, which is usually more convenient since the data is already
# mounted there. numpy is present already and matplotlib comes in with
# mmdet; only flask is actually missing.
RUN pip install --no-cache-dir flask

# mmcv-full/mmdet/nuscenes-devkit/av2 all transitively pull in non-headless
# opencv-python, which links its GUI backend against libGL.so.1. Swap to the
# headless build -- same OpenCV, no GL/X11 linkage at all -- so cv2 never
# needs a system libGL.so.1 (e.g. on Singularity/Apptainer clusters with
# --nv where the host's newer libGL.so.1 can get bind-mounted over the
# container's own and fail to load against this image's glibc).
RUN OPENCV_VERSION=$(pip show opencv-python | sed -n 's/^Version: //p') \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir "opencv-python-headless==${OPENCV_VERSION}"

RUN mkdir -p datasets ckpts work_dirs

# The configs use relative paths (data_root='./datasets/nuScenes'), so
# mount the dataset at that path inside the workdir, e.g.:
#   docker run --gpus all -it --shm-size=16g \
#     -v /path/to/nuScenes:/workspace/mapdiffusion/datasets/nuScenes \
#     -v /path/to/work_dirs:/workspace/mapdiffusion/work_dirs \
#     mapdiffusion:latest
# --shm-size matters: the default 64 MB is not enough for the dataloader
# workers this repo's configs use.
CMD ["/bin/bash"]
