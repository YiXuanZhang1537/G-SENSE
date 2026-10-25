"""Training and evaluation for the integrated multi-view experiment."""

import math
import os
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.misc import nested_tensor_from_tensor_list


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _absolute_boxes(boxes, height, width):
    boxes = box_cxcywh_to_xyxy(boxes)
    scale = boxes.new_tensor([width, height, width, height])
    return boxes * scale


def _patches_from_predictions(pred_boxes, target, image, pos_iou, neg_iou, limit):
    """Create positive and negative patches from one view."""
    height, width = target["size"].tolist()
    image = image[:, :height, :width]
    gt_boxes = _absolute_boxes(target["boxes"], height, width)
    pred_boxes = box_cxcywh_to_xyxy(pred_boxes.detach())
    positive, negative = [], []
    for box in pred_boxes:
        if (box < 0).any() or (box > 1).any():
            continue
        absolute = box * box.new_tensor([width, height, width, height])
        if gt_boxes.numel():
            max_iou = box_iou(absolute[None], gt_boxes)[0].max().item()
        else:
            max_iou = 0.0
        x1, y1, x2, y2 = absolute.round().long().tolist()
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = image[:, y1:y2, x1:x2]
        if patch.numel() == 0:
            continue
        patch = TVF.resize(patch.float(), [128, 128], antialias=True)
        if max_iou >= pos_iou and len(positive) < limit:
            positive.append(patch)
        elif max_iou <= neg_iou and len(negative) < limit:
            negative.append(patch)
    return positive, negative


def _extract_features(patches, model, device):
    if not patches:
        return None
    net = _unwrap(model)
    nested = nested_tensor_from_tensor_list(patches).to(device)
    features, positions = net.backbone(nested)
    src, mask = features[-1].decompose()
    _, encoder = net.transformer(
        net.input_proj(src), mask, net.query_embed.weight, positions[-1]
    )
    # Keep the graph: the contrastive term must update the detector backbone.
    return F.normalize(net.linear(torch.flatten(encoder, start_dim=1)), dim=-1)


def _paired_contrastive_loss(model, cc_out, ml_out, cc_samples, ml_samples,
                             cc_targets, ml_targets, device, args):
    losses = []
    for batch_idx in range(len(cc_targets)):
        cc_pos, cc_neg = _patches_from_predictions(
            cc_out["pred_boxes"][batch_idx], cc_targets[batch_idx],
            cc_samples.tensors[batch_idx], args.patch_pos_iou,
            args.patch_neg_iou, args.max_patches,
        )
        ml_pos, ml_neg = _patches_from_predictions(
            ml_out["pred_boxes"][batch_idx], ml_targets[batch_idx],
            ml_samples.tensors[batch_idx], args.patch_pos_iou,
            args.patch_neg_iou, args.max_patches,
        )
        positive_count = min(len(cc_pos), len(ml_pos))
        negative_count = min(len(cc_neg), len(ml_neg))
        if positive_count == 0 or negative_count == 0:
            continue
        cc_features = _extract_features(
            cc_pos[:positive_count] + cc_neg[:negative_count], model, device
        )
        ml_features = _extract_features(
            ml_pos[:positive_count] + ml_neg[:negative_count], model, device
        )
        similarity_cc = cc_features @ ml_features.t() / args.contrast_temperature
        similarity_ml = ml_features @ cc_features.t() / args.contrast_temperature
        positive_logits_cc = torch.logsumexp(similarity_cc[:positive_count, :positive_count], dim=1)
        all_logits_cc = torch.logsumexp(similarity_cc[:positive_count], dim=1)
        positive_logits_ml = torch.logsumexp(similarity_ml[:positive_count, :positive_count], dim=1)
        all_logits_ml = torch.logsumexp(similarity_ml[:positive_count], dim=1)
        losses.append(0.5 * ((all_logits_cc - positive_logits_cc).mean()
                             + (all_logits_ml - positive_logits_ml).mean()))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def _semi_supervised_loss(cc_out, ml_out, cc_labels, ml_labels, device):
    terms = []
    for outputs, labels in ((cc_out, cc_labels), (ml_out, ml_labels)):
        labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
        known = labels >= 0
        if known.any():
            probability = outputs["pred_logits"].softmax(-1)[..., 1].amax(dim=1)
            terms.append(F.binary_cross_entropy(probability[known], labels[known]))
    if not terms:
        return torch.zeros((), device=device)
    return torch.stack(terms).mean()


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = f"Epoch: [{epoch}]"

    for batch in metric_logger.log_every(data_loader, 10, header):
        (cc_samples, ml_samples, cc_targets, ml_targets, _cc_boxes, _ml_boxes,
         _cc_original, _ml_original, _cc_targets_original, _ml_targets_original,
         _cc_roi, _ml_roi, cc_labels, ml_labels) = batch
        cc_samples, ml_samples = cc_samples.to(device), ml_samples.to(device)
        cc_targets = [{key: value.to(device) for key, value in target.items()}
                      for target in cc_targets]
        ml_targets = [{key: value.to(device) for key, value in target.items()}
                      for target in ml_targets]

        cc_out, ml_out = model(cc_samples), model(ml_samples)
        detection_losses = criterion(cc_out, ml_out, cc_targets, ml_targets)
        weights = criterion.weight_dict
        detection_total = sum(
            value * weights[key] for key, value in detection_losses.items()
            if key in weights
        )
        semi_loss = _semi_supervised_loss(cc_out, ml_out, cc_labels, ml_labels, device)
        contrastive_loss = _paired_contrastive_loss(
            model, cc_out, ml_out, cc_samples, ml_samples,
            cc_targets, ml_targets, device, args,
        )
        total_loss = (detection_total + args.semi_loss_coef * semi_loss
                      + args.contrast_loss_coef * contrastive_loss)

        reduced = utils.reduce_dict(detection_losses)
        scaled = {key: value * weights[key] for key, value in reduced.items() if key in weights}
        loss_value = total_loss.detach().item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(reduced)
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(
            loss=loss_value,
            loss_detection=detection_total.detach(),
            loss_semi=semi_loss.detach(),
            loss_contrast=contrastive_loss.detach(),
            **scaled,
            **{f"{key}_unscaled": value for key, value in reduced.items()},
        )
        if "class_error" in reduced:
            metric_logger.update(class_error=reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    """Evaluate the validation view images with the standard COCO evaluator."""
    model.eval()
    iou_types = tuple(key for key in ("segm", "bbox") if key in postprocessors)
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    panoptic_evaluator = None
    if "panoptic" in postprocessors:
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir or ".", "panoptic_eval"),
        )

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        outputs = model(samples)
        sizes = torch.stack([target["orig_size"] for target in targets])
        results = postprocessors["bbox"](outputs, sizes)
        if "segm" in postprocessors:
            target_sizes = torch.stack([target["size"] for target in targets])
            results = postprocessors["segm"](results, outputs, sizes, target_sizes)
        coco_evaluator.update({target["image_id"].item(): result
                               for target, result in zip(targets, results)})

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = {}
    if "bbox" in coco_evaluator.coco_eval:
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
    if "segm" in coco_evaluator.coco_eval:
        stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator
