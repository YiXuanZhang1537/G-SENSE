"""Matching strategies used by the integrated All experiment."""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou


class HungarianMatcher(nn.Module):
    """DETR matching with optional one-to-many Gaussian matching."""

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0,
                 matching_mode="o2m_gaussian", top_k=5, min_iou=0.1,
                 cost_gauss=0.1, sigma=0.3):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.matching_mode = matching_mode
        self.top_k = max(1, int(top_k))
        self.min_iou = float(min_iou)
        self.cost_gauss = float(cost_gauss)
        self.sigma = float(sigma)

    @staticmethod
    def _gaussian_attraction(pred_boxes, target_boxes, sigma):
        distance = torch.cdist(pred_boxes[:, :2], target_boxes[:, :2], p=2)
        return torch.exp(-(distance ** 2) / (2 * sigma ** 2))

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        sizes = [len(target["boxes"]) for target in targets]
        if sum(sizes) == 0:
            empty = torch.empty(0, dtype=torch.int64)
            return [(empty, empty) for _ in range(bs)]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([target["labels"] for target in targets])
        tgt_bbox = torch.cat([target["boxes"] for target in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )
        cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox
        cost = cost + self.cost_giou * cost_giou
        ious = box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )[0]

        if self.matching_mode == "o2m_gaussian" and self.cost_gauss:
            attraction = self._gaussian_attraction(out_bbox, tgt_bbox, self.sigma)
            cost = cost - self.cost_gauss * attraction * (ious >= self.min_iou)

        cost = cost.view(bs, num_queries, -1)
        if self.matching_mode == "o2o":
            chunks = cost.cpu().split(sizes, dim=-1)
            return [
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in (linear_sum_assignment(chunk[b]) for b, chunk in enumerate(chunks))
            ]

        matches = []
        offset = 0
        for batch_idx, target_count in enumerate(sizes):
            if target_count == 0:
                empty = torch.empty(0, dtype=torch.int64)
                matches.append((empty, empty))
                continue
            batch_cost = cost[batch_idx]
            batch_iou = ious[
                batch_idx * num_queries:(batch_idx + 1) * num_queries,
                offset:offset + target_count,
            ]
            query_indices = []
            target_indices = []
            for target_idx in range(target_count):
                valid = torch.where(batch_iou[:, target_idx] >= self.min_iou)[0]
                if valid.numel() == 0:
                    valid = torch.argmin(batch_cost[:, target_idx]).reshape(1)
                order = torch.argsort(batch_cost[valid, target_idx])[:self.top_k]
                query_indices.extend(valid[order].tolist())
                target_indices.extend([target_idx] * len(order))
            matches.append((
                torch.as_tensor(query_indices, dtype=torch.int64),
                torch.as_tensor(target_indices, dtype=torch.int64),
            ))
            offset += target_count
        return matches


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        matching_mode=getattr(args, "matching_mode", "o2m_gaussian"),
        top_k=getattr(args, "k_matches", 5),
        min_iou=getattr(args, "match_iou", 0.1),
        cost_gauss=getattr(args, "set_cost_gauss", 0.1),
        sigma=getattr(args, "gauss_sigma", 0.3),
    )
