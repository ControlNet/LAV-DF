import os.path
from pathlib import Path
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule, Trainer, Callback
from torch import Tensor

from dataset import LavdfDataModule
from dataset.lavdf import Metadata


def nullable_index(obj, index):
    if obj is None:
        return None
    return obj[index]


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
            fusion_bm_map, v_bm_map, a_bm_map = outputs
            batch_size = fusion_bm_map.shape[0]

            for i in range(batch_size):
                n_frames = batch[3][i]
                video_name = batch[9][i]
                assert isinstance(video_name, str)
                assert video_name == self.metadata[batch_idx * batch_size + i].file
                if self.save_fusion:
                    self.gen_df_for_batfd(fusion_bm_map[i], n_frames, os.path.join(
                        "output", "results", self.model_name, video_name.split('/')[-1].replace(".mp4", ".csv")
                    ))
                if self.save_visual:
                    self.gen_df_for_batfd(v_bm_map[i], n_frames, os.path.join(
                        "output", "results", f"{self.model_name}_v", video_name.split('/')[-1].replace(".mp4", ".csv")
                    ))
                if self.save_audio:
                    self.gen_df_for_batfd(a_bm_map[i], n_frames, os.path.join(
                        "output", "results", f"{self.model_name}_a", video_name.split('/')[-1].replace(".mp4", ".csv")
                    ))
        elif self.model_type == "batfd_plus":
            fusion_bm_map, fusion_start, fusion_end, v_bm_map, v_start, v_end, a_bm_map, a_start, a_end = outputs
            batch_size = fusion_bm_map.shape[0]

            for i in range(batch_size):
                n_frames = batch[5][i]
                video_name = batch[-1][i]
                assert isinstance(video_name, str)

                if self.save_fusion:
                    self.gen_df_for_batfd_plus(fusion_bm_map[i], nullable_index(fusion_start, i), nullable_index(fusion_end, i), 
                        n_frames, os.path.join("output", "results", self.model_name, video_name.split('/')[-1].replace(".mp4", ".csv")
                    ))
                if self.save_visual:
                    self.gen_df_for_batfd_plus(v_bm_map[i], nullable_index(v_start, i), nullable_index(v_end, i), 
                        n_frames, os.path.join("output", "results", f"{self.model_name}_v", video_name.split('/')[-1].replace(".mp4", ".csv")
                    ))
                if self.save_audio:
                    self.gen_df_for_batfd_plus(a_bm_map[i], nullable_index(a_start, i), nullable_index(a_end, i), 
                        n_frames, os.path.join("output", "results", f"{self.model_name}_a", video_name.split('/')[-1].replace(".mp4", ".csv")
                    ))
        else:
            raise ValueError("Invalid model type")

    def gen_df_for_batfd(self, bm_map: Tensor, n_frames: int, output_file: str):
        bm_map = bm_map.cpu().numpy()
        n_frames = n_frames.cpu().item()
        # for each boundary proposal in boundary map
        df = pd.DataFrame(bm_map)
        df = df.stack().reset_index()
        df.columns = ["duration", "begin", "score"]
        df["end"] = df.duration + df.begin
        df = df[(df.duration > 0) & (df.end <= n_frames)]
        df = df.sort_values(["begin", "end"])
        df = df.reset_index()[["begin", "end", "score"]]
        df.to_csv(output_file, index=False)

    def gen_df_for_batfd_plus(self, bm_map: Tensor, start: Optional[Tensor], end: Optional[Tensor], n_frames: int,
        output_file: str
    ):
        bm_map = bm_map.cpu().numpy()
        if start is not None and end is not None:
            start = start.cpu().numpy()
            end = end.cpu().numpy()

        # for each boundary proposal in boundary map
        df = pd.DataFrame(bm_map)
        df = df.stack().reset_index()
        df.columns = ["duration", "begin", "score"]
        df["end"] = df.duration + df.begin
        df = df[(df.duration > 0) & (df.end <= n_frames)]
        df = df.sort_values(["begin", "end"])
        df = df.reset_index()[["begin", "end", "score"]]
        if start is not None and end is not None:
            df["score"] = df["score"] * start[df.begin] * end[df.end]
        df.to_csv(output_file, index=False)


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
