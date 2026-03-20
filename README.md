<div align="center">
<h1>
Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction
</h1>

[Jiaxin Huang](https://jaceyhuang.github.io/), [Yuanbo Yang](https://github.com/freemty), [Bangbang Yang](https://ybbbbt.com/), [Lin Ma](https://scholar.google.com/citations?user=S4HGIIUAAAAJ&hl=en),  [Yuewen Ma](https://scholar.google.com/citations?user=VG_cdLAAAAAJ), [Yiyi Liao](https://yiyiliao.github.io/)

<a href="https://xdimlab.github.io/Gen3R/"><img src="https://img.shields.io/badge/Project_Page-yellowgreen" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2601.04090"><img src="https://img.shields.io/badge/arXiv-2601.04090-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/JaceyH919/Gen3R"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

<p align="center">
  <a href="">
    <img src="./assets/teaser.png" alt="Logo" width="100%">
  </a>
</p>

<p align="left">
<strong>TL;DR</strong>: Gen3R creates multi-quantity geometry with RGB from images via a unified latent space that aligns geometry and appearance.
</p>

</div>

## 🛠️ Setup
We train and test our model under the following environment: 
- Debian GNU/Linux 12 (bookworm)
- NVIDIA H20 (96G)
- CUDA 12.4
- Python 3.11
- Pytorch 2.5.1+cu124


1. Clone this repository
```bash
git clone https://github.com/JaceyHuang/Gen3R
cd Gen3R
```
2. Install packages
```bash
conda create -n gen3r python=3.11.2 -y
conda activate gen3r
pip install -r requirements.txt
```
3. (**Important**) Download pretrained Gen3R checkpoint from [HuggingFace](https://huggingface.co/JaceyH919/Gen3R) to ./checkpoints
```bash
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/JaceyH919/Gen3R ./checkpoints
```
- **Note:** At present, direct loading weights from HuggingFace via `from_pretrained("JaceyH919/Gen3R")` is not supported due to module naming errors. Please download the model checkpoint **locally** and load it using `from_pretrained("./checkpoints")`.

## 🚀 Inference
Run the python script `infer.py` as follows to test the examples
```bash
python infer.py \
    --pretrained_model_name_or_path ./checkpoints \
    --task 2view \
    --prompts examples/2-view/colosseum/prompts.txt \
    --frame_path examples/2-view/colosseum/first.png examples/2-view/colosseum/last.png \
    --cameras free \
    --output_dir ./results \
    --remove_far_points
```
Some important inference settings below:
- `--task`: `1view` for `First Frame to 3D`, `2view` for `First-last Frames to 3D`, and `allview` for `3D Reconstruction`.
- `--prompts`: the text prompt string or the path to the text prompt file.
- `--frame_path`: the path to the conditional images/video. For the `allview` task, this can be either the path to a folder containing all frames or the path to the conditional video. For the other two tasks, it should be the path to the conditional image(s).
- `--cameras`: the path to the conditional camera extrinsics and intrinsics. We also provide basic trajectories by specifying this argument as `zoom_in`, `zoom_out`, `arc_left`, `arc_right`, `translate_up` or `translate down`. In this way, we will first use [VGGT](https://github.com/facebookresearch/vggt) to estimate the initial camera intrinsics and scene scale. To disable camera conditioning, set this argument to `free`.

Note that the default resolution of our model is 560×560. If the resolution of the conditioning images or videos differs from this, we first apply resizing followed by center cropping to match the required resolution.

### More examples
<details>
<summary>Click to expand</summary>

- **First Frame to 3D**

```bash
python infer.py \
    --pretrained_model_name_or_path ./checkpoints \
    --task 1view \
    --prompts examples/1-view/prompts.txt \
    --frame_path examples/1-view/knossos.png \
    --cameras zoom_out \
    --output_dir ./results
```

- **First-last Frames to 3D**

```bash
python infer.py \
    --pretrained_model_name_or_path ./checkpoints \
    --task 2view \
    --prompts examples/2-view/bedroom/prompts.txt \
    --frame_path examples/2-view/bedroom/first.png examples/2-view/bedroom/last.png\
    --cameras examples/2-view/bedroom/cameras.json \
    --output_dir ./results
```

- **3D Reconstruction**, note that `--cameras` are ignored in this task.

```bash
python infer.py \
    --pretrained_model_name_or_path ./checkpoints \
    --task allview \
    --prompts examples/all-view/prompts.txt \
    --frame_path examples/all-view/garden.mp4 \
    --output_dir ./results
```
</details>

## 🚗 Training
By default, we train our models using 24 H20 GPUs (96 GB VRAM each). However, the models can also be trained on other GPU configurations by appropriately adjusting the batch size. Our training is divided into two parts.

#### 0. Data Preparation
Our dataloader naturally supports multiple datasets; however, as each dataset has a different format, they must be standardized before training. The required data format is as follows:
```
|- /path/to/dataset
    |-RealEstate10K
        |-scene_0
            |-images
                |-frame00000.png
                |-frame00001.png
                ...
            |-captions.txt
            |-transforms.json
        |-scene_1
            |-images
                |-frame00000.png
                |-frame00001.png
                ...
            |-captions.txt
            |-transforms.json
        ...
        |-train_cameras_paths.txt
        |-train_captions_paths.txt
        |-train_videos_dirs.txt
    ...
    |-Co3Dv2
        |-scene_0
            |-images
                |-frame00000.png
                |-frame00001.png
                ...
            |-captions.txt
            |-transforms.json
        ...
        |-train_cameras_paths.txt
        |-train_captions_paths.txt
        |-train_videos_dirs.txt
    |-test_cameras_paths.txt
    |-test_captions_paths.txt
    |-test_videos_dirs.txt
    |-train_cameras_paths.txt
    |-train_captions_paths.txt
    |-train_videos_dirs.txt
```
For each scene, `captions.txt` stores the text prompt, and `transforms.json` follows [`nerfstudio`](https://docs.nerf.studio/quickstart/data_conventions.html#dataset-format) format, containing per-frame metadata (e.g. `w`, `h`, `file_path`, `transform_matrix`), with the only difference being that our `transform_matrix` uses OpenCV convention. 

For each dataset, `train_cameras_paths.txt`, `train_captions_paths.txt` and `train_videos_dirs.txt` specify the paths to `transforms.json`, `captions.txt` and the `images` directory for the training scenes, respectively. 

Additionally, under the root path, the files `test_cameras_paths.txt`, `test_captions_paths.txt` and `test_videos_dirs.txt` store the corresponding paths for all test scenes across datasets; while `train_cameras_paths.txt`, `train_captions_paths.txt` and `train_videos_dirs.txt` store those for all training scenes.

#### 1. Geometry Adapter
Run the bash script `train_geo_adapter_pl.sh` to start training the geometry adapter
```bash
./train_geo_adapter_pl.sh
```
Beforehand, please update the environment variables in the script to the correct paths for the models, datasets and outputs (e.g. `VGGT_PATH`, `WAN_VAE_PATH`, `DATASET_PATH` and `OUTPUT_DIR`).

#### 2. Diffusion Model
Run the bash script `train_dit.sh` to start training the diffusion model
```bash
./train_dit.sh
```
Before that, please update the environment variables in the script to the correct paths for the models, datasets and outputs (e.g. `WAN_PATH`, `VGGT_PATH`, `GEO_ADAPTER_PATH`, `DATASET_PATH` and `OUTPUT_DIR`).

## ✅ TODO
- [x] Release inference code and checkpoints
- [x] Release data preparation & training code

## 🎓 Citation
Please cite our paper if you find this repository useful:

```bibtex
@misc{huang2026gen3r3dscenegeneration,
      title={Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction}, 
      author={Jiaxin Huang and Yuanbo Yang and Bangbang Yang and Lin Ma and Yuewen Ma and Yiyi Liao},
      year={2026},
      eprint={2601.04090},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.04090}, 
}
```