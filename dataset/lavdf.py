import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable, Any, Union, Tuple

import einops
import numpy as np
import scipy as sp
import torch
import torchaudio
from einops import rearrange
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.nn import functional as F, Identity
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset

from utils import read_json, read_video, padding_video, padding_audio, resize_video, iou_with_anchors, ioa_with_anchors


@dataclass
class Metadata:
    file: str
    n_fakes: int
    fake_periods: List[List[int]]
    duration: float
    original: Optional[str]
    modify_video: bool
    modify_audio: bool
    split: str
    video_frames: int
    audio_channels: int
    audio_frames: int


T_LABEL = Union[Tensor, Tuple[Tensor, Tensor, Tensor]]


class Lavdf(Dataset):

    def __init__(self, subset: str, root: str = "data", frame_padding: int = 512,
        max_duration: int = 40, fps: int = 25,
        video_transform: Callable[[Tensor], Tensor] = Identity(),
        audio_transform: Callable[[Tensor], Tensor] = Identity(),
        metadata: Optional[List[Metadata]] = None,
        get_meta_attr: Callable[[Metadata, Tensor, Tensor, T_LABEL], List[Any]] = None,
        require_match_scores: bool = False
    ):
        self.subset = subset
        self.root = root
        self.video_padding = frame_padding
        self.audio_padding = int(frame_padding / fps * 16000)
        self.max_duration = max_duration
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.get_meta_attr = get_meta_attr
        self.require_match_scores = require_match_scores

        label_dir = os.path.join(self.root, "label")
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        if metadata is None:
            metadata: List[Metadata] = read_json(os.path.join(self.root, "metadata.min.json"), lambda x: Metadata(**x))
            self.metadata: List[Metadata] = [each for each in metadata if each.split == subset]

        else:
            self.metadata: List[Metadata] = metadata

        if self.require_match_scores:
            temporal_gap = 1 / self.max_duration
            # [-0.05, ..., 0.985]
            self.anchor_x_min = [temporal_gap * (i - 0.5) for i in range(self.video_padding)]
            # [0.05, ..., 0.995]
            self.anchor_x_max = [temporal_gap * (i + 0.5) for i in range(self.video_padding)]
        else:
            self.anchor_x_min = None
            self.anchor_x_max = None
        print(f"Load {len(self.metadata)} data in {subset}.")

    def __getitem__(self, index: int) -> List[Tensor]:
        meta = self.metadata[index]
        video, audio, _ = read_video(os.path.join(self.root, meta.file))
        video = padding_video(video, target=self.video_padding)
        audio = padding_audio(audio, target=self.audio_padding)

        video = self.video_transform(video)
        audio = self.audio_transform(audio)

        video = rearrange(resize_video(video, (96, 96)), "t c h w -> c t h w")
        audio = self._get_log_mel_spectrogram(audio)

        if not self.require_match_scores:
            label = self.get_label(meta)
            return [video, audio, label] + self.get_meta_attr(meta, video, audio, label)
        else:
            label = self.get_label_with_match_scores(meta)
            return [video, audio, *label] + self.get_meta_attr(meta, video, audio, label)

    def get_label(self, meta: Metadata) -> Tensor:
        file_name = meta.file.split("/")[-1].split(".")[0] + ".npy"
        path = os.path.join(self.root, "label", file_name)
        if os.path.exists(path):
            try:
                arr = np.load(path)
            except ValueError:
                pass
            else:
                return torch.tensor(arr)

        label = self._get_train_label(meta.video_frames, meta.fake_periods, meta.video_frames).numpy()
        # cache label
        np.save(path, label)
        return torch.tensor(label)

    def get_label_with_match_scores(self, meta: Metadata) -> Tuple[Tensor, Tensor, Tensor]:
        Path(os.path.join(self.root, "label")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.root, "match_scores")).mkdir(parents=True, exist_ok=True)

        boundary_map_file_name = meta.file.split("/")[-1].split(".")[0] + ".npy"
        boundary_map_file_path = os.path.join(self.root, "label", boundary_map_file_name)

        match_scores_file_name = meta.file.split("/")[-1].split(".")[0] + ".npz"
        match_scores_file_path = os.path.join(self.root, "match_scores", match_scores_file_name)

        if os.path.exists(boundary_map_file_path) and os.path.exists(match_scores_file_path):
            try:
                boundary_map = np.load(boundary_map_file_path)
                match_scores = np.load(match_scores_file_path)
            except ValueError:
                pass
            else:
                return (
                    torch.tensor(boundary_map),
                    torch.tensor(match_scores["match_score_start"]),
                    torch.tensor(match_scores["match_score_end"])
                )

        boundary_map, match_score_start, match_score_end = self._get_train_label(
            meta.video_frames, meta.fake_periods, meta.video_frames
        )

        # cache label
        np.save(boundary_map_file_path, boundary_map.numpy())
        np.savez(
            match_scores_file_path,
            match_score_start=match_score_start.numpy(),
            match_score_end=match_score_end.numpy()
        )

        return boundary_map, match_score_start, match_score_end

    def gen_label(self) -> None:
        # manually pre-generate label as npy
        for meta in self.metadata:
            self.get_label(meta)

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def _get_log_mel_spectrogram(audio: Tensor) -> Tensor:
        ms = torchaudio.transforms.MelSpectrogram(n_fft=321, n_mels=64)
        spec = torch.log(ms(audio[:, 0]) + 0.01)
        assert spec.shape == (64, 2048), "Wrong log mel-spectrogram setup in Dataset"
        return spec

    def _get_train_label(self, frames, video_labels, temporal_scale, fps=25) -> T_LABEL:
        corrected_second = frames / fps
        temporal_gap = 1 / temporal_scale

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_start = max(min(1, video_labels[j][0] / corrected_second), 0)
            tmp_end = max(min(1, video_labels[j][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = torch.tensor(gt_bbox)
        if len(gt_bbox) > 0:
            gt_xmins = gt_bbox[:, 0]
            gt_xmaxs = gt_bbox[:, 1]
        else:
            gt_xmins = np.array([])
            gt_xmaxs = np.array([])
        #####################################################################################################

        gt_iou_map = torch.zeros([self.max_duration, temporal_scale])
        if len(gt_bbox) > 0:
            for begin in range(temporal_scale):
                for duration in range(self.max_duration):
                    end = begin + duration
                    if end > temporal_scale:
                        break
                    gt_iou_map[duration, begin] = torch.max(
                        iou_with_anchors(begin * temporal_gap, (end + 1) * temporal_gap, gt_xmins, gt_xmaxs))
                    # [i, j]: Start in i, end in j.

        ##########################################################################################################
        gt_iou_map = F.pad(gt_iou_map.float(), pad=[0, self.video_padding - frames, 0, 0])

        if not self.require_match_scores:
            return gt_iou_map

        gt_len_small = 3 * temporal_gap
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        ##########################################################################################################
        # calculate the ioa for all timestamp
        if len(gt_start_bboxs) > 0:
            match_score_start = []
            for jdx in range(len(self.anchor_x_min)):
                match_score_start.append(np.max(ioa_with_anchors(self.anchor_x_min[jdx], self.anchor_x_max[jdx],
                    gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))

            match_score_end = []
            for jdx in range(len(self.anchor_x_min)):
                match_score_end.append(np.max(ioa_with_anchors(self.anchor_x_min[jdx], self.anchor_x_max[jdx],
                    gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
            match_score_start = torch.Tensor(match_score_start)
            match_score_end = torch.Tensor(match_score_end)
        else:
            match_score_start = torch.zeros(len(self.anchor_x_min))
            match_score_end = torch.zeros(len(self.anchor_x_min))
        ############################################################################################################
        return gt_iou_map, match_score_start, match_score_end


def _default_get_meta_attr(meta: Metadata, video: Tensor, audio: Tensor, label: Tensor) -> List[Any]:
    return [meta.video_frames]


class LavdfDataModule(LightningDataModule):
    train_dataset: Lavdf
    dev_dataset: Lavdf
    test_dataset: Lavdf
    metadata: List[Metadata]

    def __init__(self, root: str = "data", frame_padding=512, max_duration=40,
        require_match_scores: bool = False, feature_types: Tuple[Optional[str], Optional[str]] = (None, None),
        batch_size: int = 1, num_workers: int = 0,
        take_train: int = None, take_dev: int = None, take_test: int = None,
        cond: Optional[Callable[[Metadata], bool]] = None,
        get_meta_attr: Callable[[Metadata, Tensor, Tensor, Tensor], List[Any]] = _default_get_meta_attr
    ):
        super().__init__()
        self.root = root
        self.frame_padding = frame_padding
        self.max_duration = max_duration
        self.require_match_scores = require_match_scores
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.take_train = take_train
        self.take_dev = take_dev
        self.take_test = take_test
        self.cond = cond
        self.get_meta_attr = get_meta_attr
        self.Dataset = feature_type_to_dataset_type[feature_types]

    def setup(self, stage: Optional[str] = None) -> None:
        self.metadata: List[Metadata] = read_json(os.path.join(self.root, "metadata.min.json"), lambda x: Metadata(**x))

        train_metadata = []
        dev_metadata = []
        test_metadata = []

        for meta in self.metadata:
            if self.cond is None or self.cond(meta):
                if meta.split == "train":
                    train_metadata.append(meta)
                elif meta.split == "dev":
                    dev_metadata.append(meta)
                elif meta.split == "test":
                    test_metadata.append(meta)

        if self.take_dev is not None:
            dev_metadata = dev_metadata[:self.take_dev]

        self.train_dataset = self.Dataset("train", self.root, self.frame_padding, self.max_duration,
            metadata=train_metadata, get_meta_attr=self.get_meta_attr,
            require_match_scores=self.require_match_scores
        )
        self.dev_dataset = self.Dataset("dev", self.root, self.frame_padding, self.max_duration,
            metadata=dev_metadata, get_meta_attr=self.get_meta_attr,
            require_match_scores=self.require_match_scores
        )
        self.test_dataset = self.Dataset("test", self.root, self.frame_padding, self.max_duration,
            metadata=test_metadata, get_meta_attr=self.get_meta_attr,
            require_match_scores=self.require_match_scores
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=RandomSampler(self.train_dataset, num_samples=self.take_train, replacement=True)
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)


# The dictionary is used to map the feature type to the dataset type
# The key is a tuple of (visual_feature_type, audio_feature_type), ``None`` means using end-to-end encoder.
feature_type_to_dataset_type = {
    (None, None): Lavdf
}
