from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from typing import List, Union

import torch
from torch import Tensor
from torch.nn import Module

from dataset.lavdf import Metadata
from utils import iou_with_anchors


class AP(Module):
    """
    Average Precision

    The mean precision in Precision-Recall curve.
    """

    def __init__(self, iou_thresholds: Union[float, List[float]] = 0.5, tqdm_pos: int = 1):
        super().__init__()
        self.iou_thresholds: List[float] = iou_thresholds if type(iou_thresholds) is list else [iou_thresholds]
        self.tqdm_pos = tqdm_pos
        self.n_labels = 0
        self.ap: dict = {}

    def forward(self, metadata: List[Metadata], proposals_dict: dict) -> dict:

        for iou_threshold in self.iou_thresholds:
            values = []
            self.n_labels = 0

            for meta in metadata:
                proposals = torch.tensor(proposals_dict[meta.file])
                labels = torch.tensor(meta.fake_periods)
                values.append(AP.get_values(iou_threshold, proposals, labels, 25.))
                self.n_labels += len(labels)

            # sort proposals
            values = torch.cat(values)
            ind = values[:, 0].sort(stable=True, descending=True).indices
            values = values[ind]

            # accumulate to calculate precision and recall
            curve = self.calculate_curve(values)
            ap = self.calculate_ap(curve)
            self.ap[iou_threshold] = ap

        return self.ap

    def calculate_curve(self, values):
        acc_TP = 0
        acc_FP = 0
        curve = torch.zeros((len(values), 2))
        for i, (confidence, is_TP) in enumerate(values):
            if is_TP == 1:
                acc_TP += 1
            else:
                acc_FP += 1

            precision = acc_TP / (acc_TP + acc_FP)
            recall = acc_TP / self.n_labels
            curve[i] = torch.tensor((recall, precision))

        curve = torch.cat([torch.tensor([[1., 0.]]), torch.flip(curve, dims=(0,))])
        return curve

    def calculate_ap(self, curve):
        y_max = 0.
        ap = 0
        for i in range(len(curve) - 1):
            x1, y1 = curve[i]
            x2, y2 = curve[i + 1]
            if y1 > y_max:
                y_max = y1
            dx = x1 - x2
            ap += dx * y_max
        return ap

    @staticmethod
    def get_values(
        iou_threshold: float,
        proposals: Tensor,
        labels: Tensor,
        fps: float,
    ) -> Tensor:
        n_labels = len(labels)
        ious = torch.zeros((len(proposals), n_labels))
        for i in range(len(labels)):
            ious[:, i] = iou_with_anchors(proposals[:, 1] / fps, proposals[:, 2] / fps, labels[i, 0], labels[i, 1])

        # values: (confidence, is_TP) rows
        n_labels = ious.shape[1]
        detected = torch.full((n_labels,), False)
        confidence = proposals[:, 0]
        potential_TP = ious > iou_threshold

        for i in range(len(proposals)):
            for j in range(n_labels):
                if potential_TP[i, j]:
                    if detected[j]:
                        potential_TP[i, j] = False
                    else:
                        # mark as detected
                        potential_TP[i] = False  # mark others as False
                        potential_TP[i, j] = True  # mark the selected as True
                        detected[j] = True

        is_TP = potential_TP.any(dim=1)
        values = torch.column_stack([confidence, is_TP])
        return values


class AR(Module):
    """
    Average Recall

    Args:
        n_proposals_list: Number of proposals. 100 for AR@100.
        iou_thresholds: IOU threshold samples for the curve. Default: [0.5:0.05:0.95]

    """

    def __init__(self, n_proposals_list: Union[List[int], int] = 100, iou_thresholds: List[float] = None,
        parallel: bool = True
    ):
        super().__init__()
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.n_proposals_list: List[int] = n_proposals_list if type(n_proposals_list) is list else [n_proposals_list]
        self.iou_thresholds = iou_thresholds
        self.parallel = parallel
        self.ar: dict = {}

    def forward(self, metadata: List[Metadata], proposals_dict: dict) -> dict:
        for n_proposals in self.n_proposals_list:
            if self.parallel:
                with ProcessPoolExecutor(cpu_count() // 2 - 1) as executor:
                    futures = []
                    for meta in metadata:
                        proposals = torch.tensor(proposals_dict[meta.file])
                        labels = torch.tensor(meta.fake_periods)
                        futures.append(executor.submit(AR.get_values, n_proposals, self.iou_thresholds,
                            proposals, labels, 25.))

                    values = list(map(lambda x: x.result(), futures))
                    values = torch.stack(values)
            else:
                values = torch.zeros((len(metadata), len(self.iou_thresholds), 2))
                for i, meta in enumerate(metadata):
                    proposals = torch.tensor(proposals_dict[meta.file])
                    labels = torch.tensor(meta.fake_periods)
                    values[i] = AR.get_values(n_proposals, self.iou_thresholds, proposals, labels, 25.)

            values_sum = values.sum(dim=0)

            TP = values_sum[:, 0]
            FN = values_sum[:, 1]
            recall = TP / (TP + FN)
            self.ar[n_proposals] = recall.mean()

        return self.ar

    @staticmethod
    def get_values(
        n_proposals: int,
        iou_thresholds: List[float],
        proposals: Tensor,
        labels: Tensor,
        fps: float,
    ):
        proposals = proposals[:n_proposals]
        n_proposals = proposals.shape[0]
        n_labels = len(labels)
        ious = torch.zeros((n_proposals, n_labels))
        for i in range(len(labels)):
            ious[:, i] = iou_with_anchors(proposals[:, 1] / fps, proposals[:, 2] / fps, labels[i, 0], labels[i, 1])

        n_thresholds = len(iou_thresholds)

        # values: rows of (TP, FN)
        iou_max = ious.max(dim=0)[0]
        values = torch.zeros((n_thresholds, 2))

        for i in range(n_thresholds):
            iou_threshold = iou_thresholds[i]
            TP = (iou_max > iou_threshold).sum()
            FN = n_labels - TP
            values[i] = torch.tensor((TP, FN))

        return values
