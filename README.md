# Localized Audio Visual DeepFake Dataset (LAV-DF)

<div align="center">
    <img src="assets/overview.svg">
    <p></p>
</div>

<div align="center">
    <a href="https://github.com/ControlNet/LAV-DF/issues">
        <img src="https://img.shields.io/github/issues/ControlNet/LAV-DF?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/LAV-DF/network/members">
        <img src="https://img.shields.io/github/forks/ControlNet/LAV-DF?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/LAV-DF/stargazers">
        <img src="https://img.shields.io/github/stars/ControlNet/LAV-DF?style=flat-square">
    </a>
    <a href="https://github.com/ControlNet/LAV-DF/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-97ca00?style=flat-square">
    </a>
    <a href="https://arxiv.org/abs/2204.06228">
        <img src="https://img.shields.io/badge/arXiv-2204.06228-b31b1b.svg?style=flat-square">
    </a>
    <a href="https://arxiv.org/abs/2305.01979">
        <img src="https://img.shields.io/badge/arXiv-2305.01979-b31b1b.svg?style=flat-square">
    </a>
    <a href="https://paperswithcode.com/sota/temporal-forgery-localization-on-lav-df?p=glitch-in-the-matrix-a-large-scale-benchmark">
        <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/glitch-in-the-matrix-a-large-scale-benchmark/temporal-forgery-localization-on-lav-df&style=flat-square">
    </a>
</div>

This repo is the official PyTorch implementation for the DICTA paper [Do You Really Mean That? Content Driven Audio-Visual 
Deepfake Dataset and Multimodal Method for Temporal Forgery Localization](https://ieeexplore.ieee.org/document/10034605)
(Best Award), and the journal paper [_Glitch in the Matrix_: A Large Scale Benchmark for Content Driven Audio-Visual 
Forgery Detection and Localization](https://www.sciencedirect.com/science/article/pii/S1077314223001984) accepted by CVIU.

## LAV-DF Dataset

### Download

To use this LAV-DF dataset, you should agree the [terms and conditions](https://github.com/ControlNet/LAV-DF/blob/master/TERMS_AND_CONDITIONS.md).

Download link: [Google Drive](https://drive.google.com/file/d/1-OQ-NDtdEyqHNLaZU1Lt9Upk5wVqfYJw/view?usp=sharing), [HuggingFace](https://huggingface.co/datasets/ControlNet/LAV-DF).

### Baseline Benchmark

| Method  | AP@0.5 | AP@0.75 | AP@0.95 | AR@100 | AR@50 | AR@20 | AR@10 |
|---------|--------|---------|---------|--------|-------|-------|-------|
| BA-TFD  | 79.15  | 38.57   | 00.24   | 67.03  | 64.18 | 60.89 | 58.51 |
| BA-TFD+ | 96.30  | 84.96   | 04.44   | 81.62  | 80.48 | 79.40 | 78.75 |

Please note this result of BA-TFD is slightly better than the one reported in the paper. 
This is because we have used the better hyperparameters in this repository.

## Baseline Models

### Requirements

The main versions are,
- Python >= 3.7, < 3.11
- PyTorch >= 1.13
- torchvision >= 0.14
- pytorch_lightning == 1.7.*

Run the following command to install the required packages.

```bash
pip install -r requirements.txt
```

### Training BA-TFD

Train the BA-TFD introduced in paper [Do You Really Mean That? Content Driven Audio-Visual 
Deepfake Dataset and Multimodal Method for Temporal Forgery Localization](https://ieeexplore.ieee.org/document/10034605) with default hyperparameter on LAV-DF dataset.

```bash
python train.py \
  --config ./config/batfd_default.toml \
  --data_root <DATASET_PATH> \
  --batch_size 4 --num_workers 8 --gpus 1 --precision 16
```

The checkpoint will be saved in `ckpt` directory, and the tensorboard log will be saved in `lighntning_logs` directory.

### Training BA-TFD+

Train the BA-TFD+ introduced in paper [_Glitch in the Matrix_: A Large Scale Benchmark for Content Driven Audio-Visual Forgery Detection and Localization](https://www.sciencedirect.com/science/article/pii/S1077314223001984) with default hyperparameter on LAV-DF dataset.

```bash
python train.py \
  --config ./config/batfd_plus_default.toml \
  --data_root <DATASET_PATH> \
  --batch_size 4 --num_workers 8 --gpus 2 --precision 32
```

Please use `FP32` for training BA-TFD+ as `FP16` will cause inf and nan.

The checkpoint will be saved in `ckpt` directory, and the tensorboard log will be saved in `lighntning_logs` directory.


### Evaluation

Please run the following command to evaluate the model with the checkpoint saved in `ckpt` directory.

Besides, you can also download the [BA-TFD](https://github.com/ControlNet/LAV-DF/releases/download/pretrained_model/baftd_default.ckpt) and [BA-TFD+](https://github.com/ControlNet/LAV-DF/releases/download/pretrained_model_v2/baftd_plus_default.ckpt) pretrained models.

```bash
python evaluate.py \
  --config ./config/default.toml \
  --data_root <DATASET_PATH> \
  --checkpoint <CHECKPOINT_PATH>
```

In the script, there will be a temporal inference results generated in `output` directory, and the AP and AR scores will
be printed in the console.

Note please make sure only one GPU is visible to the evaluation script.

## References

If you find this work useful in your research, please cite them.

The conference paper,
```bibtex
@inproceedings{cai2022you,
  title = {Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization},
  author = {Cai, Zhixi and Stefanov, Kalin and Dhall, Abhinav and Hayat, Munawar},
  booktitle = {2022 International Conference on Digital Image Computing: Techniques and Applications (DICTA)},
  year = {2022},
  doi = {10.1109/DICTA56598.2022.10034605},
  pages = {1--10},
  address = {Sydney, Australia},
}
```

The extended journal version is accepted by CVIU,
```bibtex
@article{cai2023glitch,
  title = {Glitch in the Matrix: A Large Scale Benchmark for Content Driven Audio-Visual Forgery Detection and Localization},
  author = {Cai, Zhixi and Ghosh, Shreya and Dhall, Abhinav and Gedeon, Tom and Stefanov, Kalin and Hayat, Munawar},
  journal = {Computer Vision and Image Understanding},
  year = {2023},
  volume = {236},
  pages = {103818},
  issn = {1077-3142},
  doi = {10.1016/j.cviu.2023.103818},
}
```

## License

This project is under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## Acknowledgements

Some code related to boundary matching mechanism is borrowed from 
[JJBOY/BMN-Boundary-Matching-Network](https://github.com/JJBOY/BMN-Boundary-Matching-Network) and 
[xxcheng0708/BSNPlusPlus-boundary-sensitive-network](https://github.com/xxcheng0708/BSNPlusPlus-boundary-sensitive-network).
