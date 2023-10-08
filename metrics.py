from typing import List, Union

import torch
from torch import Tensor
from tqdm.auto import tqdm

from dataset.lavdf import Metadata
from utils import iou_1d


class AP:
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

    def __call__(self, metadata: List[Metadata], proposals_dict: dict) -> dict:

        for iou_threshold in self.iou_thresholds:
            values = []
            self.n_labels = 0

            for meta in tqdm(metadata):
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
        is_TP = values[:, 1]
        acc_TP = torch.cumsum(is_TP, dim=0)
        precision = acc_TP / (torch.arange(len(is_TP)) + 1)
        recall = acc_TP / self.n_labels
        curve = torch.stack([recall, precision]).T
        curve = torch.cat([torch.tensor([[1., 0.]]), torch.flip(curve, dims=(0,))])
        return curve

    @staticmethod
    def calculate_ap(curve):
        x, y = curve.T
        y_max = y.cummax(dim=0).values
        x_diff = x.diff().abs()
        ap = (x_diff * y_max[:-1]).sum()
        return ap

    @staticmethod
    def get_values(
        iou_threshold: float,
        proposals: Tensor,
        labels: Tensor,
        fps: float,
    ) -> Tensor:
        n_labels = len(labels)
        n_proposals = len(proposals)
        if n_labels > 0:
            ious = iou_1d(proposals[:, 1:] / fps, labels)
        else:
            ious = torch.zeros((n_proposals, 0))

        # values: (confidence, is_TP) rows
        n_labels = ious.shape[1]
        detected = torch.full((n_labels,), False)
        confidence = proposals[:, 0]
        potential_TP = ious > iou_threshold

        tp_indexes = []

        for i in range(n_labels):
            potential_TP_index = potential_TP[:, i].nonzero()
            for (j,) in potential_TP_index:
                if j not in tp_indexes:
                    tp_indexes.append(j)
                    break
        
        is_TP = torch.zeros(n_proposals, dtype=torch.bool)
        if len(tp_indexes) > 0:
            tp_indexes = torch.stack(tp_indexes)
            is_TP[tp_indexes] = True
        values = torch.column_stack([confidence, is_TP])
        return values


class AR:
    """
    Average Recall

    Args:
        n_proposals_list: Number of proposals. 100 for AR@100.
        iou_thresholds: IOU threshold samples for the curve. Default: [0.5:0.05:0.95]

    """

    def __init__(self, n_proposals_list: Union[List[int], int] = 100, iou_thresholds: List[float] = None):
        super().__init__()
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.n_proposals_list = n_proposals_list if type(n_proposals_list) is list else [n_proposals_list]
        self.n_proposals_list = torch.tensor(self.n_proposals_list)
        self.iou_thresholds = iou_thresholds
        self.ar: dict = {}

    def __call__(self, metadata: List[Metadata], proposals_dict: dict) -> dict:
        # shape: (n_metadata, n_iou_thresholds, n_proposal_thresholds, 2)
        values = torch.zeros((len(metadata), len(self.iou_thresholds), len(self.n_proposals_list), 2))
        for i, meta in enumerate(tqdm(metadata)):
            proposals = torch.tensor(proposals_dict[meta.file])
            labels = torch.tensor(meta.fake_periods)
            values[i] = self.get_values(self.iou_thresholds, proposals, labels, 25.)

        values_sum = values.sum(dim=0)

        TP = values_sum[:, :, 0]
        FN = values_sum[:, :, 1]
        recall = TP / (TP + FN)  # (n_iou_thresholds, n_proposal_thresholds)
        for i, n_proposals in enumerate(self.n_proposals_list):
            self.ar[n_proposals.item()] = recall[:, i].mean().item()

        return self.ar

    def get_values(
        self,
        iou_thresholds: List[float],
        proposals: Tensor,
        labels: Tensor,
        fps: float,
    ):
        n_proposals_list = self.n_proposals_list
        max_proposals = max(n_proposals_list)
        
        proposals = proposals[:max_proposals]
        n_labels = len(labels)

        if n_labels > 0:
            ious = iou_1d(proposals[:, 1:] / fps, labels)
        else:
            ious = torch.zeros((max_proposals, 0))

        # values: matrix of (TP, FN), shapes (n_iou_thresholds, n_proposal_thresholds, 2)
        iou_max = ious.cummax(0).values[n_proposals_list - 1]  # shape (n_iou_thresholds, n_labels)
        iou_max = iou_max[None]

        iou_thresholds = torch.tensor(iou_thresholds)[:, None, None]
        TP = (iou_max > iou_thresholds).sum(-1)
        FN = n_labels - TP
        values = torch.stack([TP, FN], dim=-1)

        return values
