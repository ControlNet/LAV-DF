import os.path
from pathlib import Path
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule, Trainer, Callback
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import LavdfDataModule
from dataset.lavdf import Metadata


class SaveToCsvCallback(Callback):

    def __init__(self, max_duration: int, metadata: List[Metadata], model_name: str, model_type: str,
        modalities: List[Literal["fusion", "visual", "audio"]]
    ):
        super().__init__()
        self.max_duration = max_duration
        self.metadata = metadata
        self.model_name = model_name
        self.model_type = model_type
        self.save_fusion = "fusion" in modalities
        self.save_visual = "visual" in modalities
        self.save_audio = "audio" in modalities

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.model_type == "batfd":
            fusion_bm_map, v_bm_map, a_bm_map = outputs.cpu().numpy()[0]
            video_name = self.metadata[batch_idx].file
            n_frames = batch[3][0]
            if self.save_fusion:
                self.gen_df_for_batfd(fusion_bm_map, n_frames, os.path.join(
                    "output", "results", self.model_name, video_name.split('/')[-1].replace(".mp4", ".csv")
                ))
            if self.save_visual:
                self.gen_df_for_batfd(v_bm_map, n_frames, os.path.join(
                    "output", "results", f"{self.model_name}_v", video_name.split('/')[-1].replace(".mp4", ".csv")
                ))
            if self.save_audio:
                self.gen_df_for_batfd(a_bm_map, n_frames, os.path.join(
                    "output", "results", f"{self.model_name}_a", video_name.split('/')[-1].replace(".mp4", ".csv")
                ))
        elif self.model_type == "batfd_plus":
            fusion_bm_map, fusion_start, fusion_end, v_bm_map, v_start, v_end, a_bm_map, a_start, a_end = outputs
            video_name = self.metadata[batch_idx].file
            n_frames = batch[5][0]
            if self.save_fusion:
                self.gen_df_for_batfd_plus(fusion_bm_map, fusion_start, fusion_end, n_frames, os.path.join(
                    "output", "results", self.model_name, video_name.split('/')[-1].replace(".mp4", ".csv")
                ))
            if self.save_visual:
                self.gen_df_for_batfd_plus(v_bm_map, v_start, v_end, n_frames, os.path.join(
                    "output", "results", f"{self.model_name}_v", video_name.split('/')[-1].replace(".mp4", ".csv")
                ))
            if self.save_audio:
                self.gen_df_for_batfd_plus(a_bm_map, a_start, a_end, n_frames, os.path.join(
                    "output", "results", f"{self.model_name}_a", video_name.split('/')[-1].replace(".mp4", ".csv")
                ))

    def gen_df_for_batfd(self, bm_map: Tensor, n_frames: int, output_file: str):
        # for each boundary proposal in boundary map
        new_props = []
        for i in range(n_frames):
            for j in range(1, self.max_duration):
                # begin frame and end frame
                begin = i
                end = i + j
                if end <= n_frames:
                    new_props.append([begin, end, bm_map[j, i]])

        new_props = np.stack(new_props)
        col_name = ["begin", "end", "score"]
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df["begin"] = new_df["begin"].astype(int)
        new_df["end"] = new_df["end"].astype(int)
        new_df.to_csv(output_file, index=False)

    def gen_df_for_batfd_plus(self, bm_map: Tensor, start: Optional[Tensor], end: Optional[Tensor], n_frames: int,
        output_file: str
    ):
        bm_map = bm_map.cpu().numpy()[0]
        if start is not None and end is not None:
            start = start.cpu().numpy()[0]
            end = end.cpu().numpy()[0]

        # for each boundary proposal in boundary map
        new_props = []
        for i in range(n_frames):
            for j in range(1, self.max_duration):
                # begin frame and end frame
                index_begin = i
                index_end = i + j
                if index_end <= n_frames:
                    if start is not None and end is not None:
                        start_score = start[index_begin]
                        end_score = end[index_end]
                        score = bm_map[j, i] * start_score * end_score
                    else:
                        score = bm_map[j, i]
                    new_props.append([index_begin, index_end, score])

        new_props = np.stack(new_props)
        col_name = ["begin", "end", "score"]
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df["begin"] = new_df["begin"].astype(int)
        new_df["end"] = new_df["end"].astype(int)
        new_df.to_csv(output_file, index=False)


def inference_batfd(model_name: str, model: LightningModule, dm: LavdfDataModule,
    max_duration: int, model_type: str,
    modalities: Optional[List[Literal["fusion", "visual", "audio"]]] = None,
    gpus: int = 1
):
    modalities = modalities or ["fusion"]

    if "fusion" in modalities:
        Path(os.path.join("output", "results", model_name)).mkdir(parents=True, exist_ok=True)
    if "visual" in modalities:
        Path(os.path.join("output", "results", f"{model_name}_v")).mkdir(parents=True, exist_ok=True)
    if "audio" in modalities:
        Path(os.path.join("output", "results", f"{model_name}_a")).mkdir(parents=True, exist_ok=True)

    model.eval()

    test_dataset = dm.test_dataset

    trainer = Trainer(logger=False,
        enable_checkpointing=False, devices=1 if gpus > 1 else None,
        accelerator="gpu" if gpus > 0 else "cpu",
        callbacks=[SaveToCsvCallback(max_duration, test_dataset.metadata, model_name, model_type, modalities)]
    )

    trainer.predict(model, dm.test_dataloader())
