# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import cv2
import numpy as np
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from PIL import Image
import torchvision.transforms as T  
from util.misc import nested_tensor_from_tensor_list
import torchvision.transforms.functional as F
import random
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import torch.nn.functional as F
# 反归一化
def rescale_bboxes(out_bbox, w, h):
    if torch.is_tensor(out_bbox):
    # 断开计算图，移动到CPU，转为numpy
        b = out_bbox.detach().cpu().numpy()
    # 处理 numpy array
    elif isinstance(out_bbox, np.ndarray):
        b = out_bbox
    # 将 out_bbox 扩展到实际图像坐标系
    b = b * np.array([w, h, w, h], dtype=np.float32)
    b[0] = min(max(b[0],0),w)
    b[1] = min(max(b[1],0),h)
    b[2] = min(max(b[2],0),w)
    b[3] = min(max(b[3],0),h)
    return b

# 计算iou
def cal_iou_xyxy(box1,box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    #计算两个框的面积
    s1 = (y1max - y1min) * (x1max - x1min)
    s2 = (y2max - y2min) * (x2max - x2min)

    #计算相交部分的坐标
    xmin = max(x1min,x2min)
    ymin = max(y1min,y2min)
    xmax = min(x1max,x2max)
    ymax = min(y1max,y2max)

    inter_h = max(ymax - ymin, 0)
    inter_w = max(xmax - xmin, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    #计算iou
    iou = intersection / union
    return iou

def tensor_to_pil(tensor):
    return T.ToPILImage()(tensor)

# 提取patch特征
def extract_patch_features(patches, model, device):
    if patches.numel() == 0 or patches.shape[0] == 0:
        return torch.empty(0, device="cpu")

    # Tensor[K,C,H,W] → list[Tensor(C,H,W)]
    patches = [p for p in patches]
    nt = nested_tensor_from_tensor_list(patches).to(device)

    xs, pos = model.backbone(nt)
    src, mask = xs[-1].decompose()
    _, enc = model.transformer(
        model.input_proj(src),
        mask,
        model.query_embed.weight,
        pos[-1]
    )

    feat = model.linear(torch.flatten(enc, start_dim=1))
    
    # ✅ 直接返回 Tensor
    return feat.detach().cpu()


def center_in_image_is_black(box_xyxy, img, thr=0.05):
    """
    box_xyxy: (xmin, ymin, xmax, ymax) 像素坐标
    img: Tensor, shape = (C, H, W)
    thr: 黑色阈值（0~1），越小越严格
    """
    xmin, ymin, xmax, ymax = box_xyxy
    cx = int(round((xmin + xmax) * 0.5))
    cy = int(round((ymin + ymax) * 0.5))

    _, H, W = img.shape
    cx = max(0, min(cx, W - 1))
    cy = max(0, min(cy, H - 1))

    # 取中心点像素（C,)
    pixel = img[:, cy, cx]

    # 如果已经 Normalize 过，这里需要反过来判断
    # 推荐：在 Normalize 之前用
    return pixel.mean().item() <= thr


def randSelect_tensor(Q, k):
    # 兼容 torch.Tensor / numpy.ndarray（都支持 .shape）
    if Q is None:
        raise ValueError("Q is None")
    if not hasattr(Q, "shape"):
        raise TypeError(f"Q has no .shape, type={type(Q)}")

    K = Q.shape[0]
    if K == 0 or k <= 0:
        return Q[:0]

    k = min(k, K)

    # 如果是 torch.Tensor：用 torch.randperm
    if torch.is_tensor(Q):
        idx = torch.randperm(K, device=Q.device)[:k]
        return Q.index_select(0, idx)

    # 如果不是 torch.Tensor（例如 numpy）：先转 tensor 再采样（可选）
    Q = torch.as_tensor(Q)
    idx = torch.randperm(K, device=Q.device)[:k]
    return Q.index_select(0, idx)


def umcl_loss_paired(cc_sel, ml_sel, M, N, tau: float = 0.07, use_m2_scale: bool = True):
    """
    cc_sel: Tensor [M+N, D]  view1 embeddings（例如 CC）
    ml_sel: Tensor [M+N, D]  view2 embeddings（例如 MLO）
    M: 正样本数量（前 M 个为正）
    tau: temperature
    use_m2_scale: True 时用论文里 1/M^2 的缩放；False 时用 mean（更稳）

    return: scalar loss
    """
    device = cc_sel.device
    assert cc_sel.dim() == 2 and ml_sel.dim() == 2, "cc_sel/ml_sel 必须是 [K,D]"
    assert cc_sel.shape == ml_sel.shape, "cc_sel/ml_sel 形状必须一致"
    K = cc_sel.size(0)
    assert 0 < M <= K, "M 必须在 (0, K]"


    Mv1 = (cc_sel @ ml_sel.t()) / tau          # [K, K]
    Mv2 = (ml_sel @ cc_sel.t()) / tau          # [K, K]

    def one_direction_loss(Mv):
        Mv = Mv[:M, :]                 # [M, K] 其中 K=M+N
        log_pos = torch.logsumexp(Mv[:, :M], dim=1)   # log(∑_{j=1..M} exp)
        log_all = torch.logsumexp(Mv, dim=1)          # log(∑_{j=1..M+N} exp)
        log_ratio = log_pos - log_all                 # [M]

        if use_m2_scale:
            # 论文写法：-(1/M^2) * sum(log_ratio)
            return -(log_ratio.sum() / (M * M))
        else:
            # 更稳写法：-(1/M) * sum(...)
            return -log_ratio.mean()

    loss_v1 = one_direction_loss(Mv1)
    loss_v2 = one_direction_loss(Mv2)

    return 0.5 * (loss_v1 + loss_v2)

def list_to_tensor(patches, device):
    """
    patches: list of Tensor [C,H,W]
    return: Tensor [K,C,H,W] 或 空 Tensor
    """
    if isinstance(patches, list):
        if len(patches) > 0:
            return torch.stack(patches).to(device)
        else:
            # 空 list -> 空 tensor（不炸）
            return torch.empty((0,), device=device)
    elif torch.is_tensor(patches):
        return patches.to(device)
    else:
        raise TypeError(f"Unsupported type: {type(patches)}")
def split_pos_neg(boxes, gts, img, w, h, pos_th, neg_th=None, iou_threshold=0.0):
    if neg_th is None:
        neg_th = pos_th

    pos, neg = [], []
    
    for b in boxes:
        # 先用归一化坐标做边界检查
        if b[0] <= 0 or b[1] <= 0 or b[2] >= 1 or b[3] >= 1:
            continue

        # rescale 只做一次，别在 gt 循环里反复 rescale
        b_abs = rescale_bboxes(b, w, h)

        # 计算该候选框对所有 GT 的最大 IoU
        max_iou = 0.0
        for gt in gts:
            iou = cal_iou_xyxy(b_abs, gt)
            if iou > max_iou:
                max_iou = iou

        # 按 max_iou 单次分配 -> 保证 pos/neg 不会同时收同一个框
        if max_iou > pos_th:
            # 检查该框是否与任何已有的正样本框重叠
            if not any(cal_iou_xyxy(b_abs, pos_b) > iou_threshold for pos_b in pos):
                pos.append(b_abs)
        elif max_iou <= neg_th:
            # 检查负样本框是否与任何正样本框有重叠
            if not any(cal_iou_xyxy(b_abs, pos_b) > iou_threshold for pos_b in pos):
                # 如果框位于图像中心并且没有被判定为正样本，则是负样本
                if center_in_image_is_black(b_abs, img):
                    continue
                neg.append((b_abs, max_iou))  # 记录负样本及其对应的最大IoU
        else:
            # 灰区：既不当正也不当负（对 MLO 这种 0.1~0.3 很重要）
            pass

    # 按照负样本的 IoU 从大到小排序
    neg = sorted(neg, key=lambda x: x[1], reverse=True)  # 按最大IoU排序

    # 将负样本框提取出来
    final_neg = [neg_box for neg_box, _ in neg]

    # 遍历所有已选的正样本，确保没有负样本与它们重叠
    final_neg_filtered = []
    for neg_box in final_neg:
        add_neg = True
        for pos_box in pos:
            if cal_iou_xyxy(neg_box, pos_box) > iou_threshold:
                add_neg = False
                break
        if add_neg:
            final_neg_filtered.append(neg_box)

    return pos, final_neg_filtered



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # print(device)

    # for CC_samples,ML_samples,CC_targets,ML_targets,cc_img_list,ml_img_list,cc_box_list,ml_box_list,CC_samples_o, ML_samples_o,CC_targets_o,ML_targets_o in metric_logger.log_every(data_loader, print_freq, header):
    for CC_samples,ML_samples,CC_targets,ML_targets, cc_box_list,ml_box_list,CC_samples_o, ML_samples_o,CC_targets_o,ML_targets_o, cc_mask, ml_mask in metric_logger.log_every(data_loader, print_freq, header):
        # batch=4
        CC_samples = CC_samples.to(device)
        ML_samples = ML_samples.to(device)
        # CC_samples_o = CC_samples_o.to(device)
        # ML_samples_o = ML_samples_o.to(device)
        CC_targets_o = [{k: v.to(device) for k, v in t.items()} for t in CC_targets_o]
        ML_targets_o=[{k: v.to(device) for k, v in  t.items()} for t in ML_targets_o]  
        # print("CC_targets",CC_targets_o)
        # print("ML_targets",ML_targets_o) 
        CC_targets = [{k: v.to(device) for k, v in t.items()} for t in CC_targets]
        ML_targets=[{k: v.to(device) for k, v in t.items()} for t in ML_targets]

        cc_out=model(CC_samples)
        ml_out=model(ML_samples)

        cc_out_per = []
        ml_out_per = []
        for bbox in cc_out['pred_boxes']:
            cc_out_per.append(bbox)
        for bbox in ml_out['pred_boxes']:
            ml_out_per.append(bbox)

        # 需要保存的列表，每个批次会添加一个字典，其中包含正负类框
        B,_,_,_ = CC_samples.tensors.shape
        L_umcl = torch.zeros((), device=device) 
        len_PG = 0
        # 遍历所有批次
        for idx in range(B):
            
            cc_pos_img = []  # 当前批次中图片的正类框
            ml_pos_img = []  # 当前批次中图片的正类框
            cc_neg_img = []  # 当前批次中图片的负类框
            ml_neg_img = []  # 当前批次中图片的负类框

            # 获取当前批次中第 idx 张图片的框
            cc_boxes = cc_out_per[idx]  # 获取当前图片的 cc_box
            ml_boxes = ml_out_per[idx]  # 获取当前图片的 ml_box          
            gt_cc = CC_targets_o[idx]['boxes']
            gt_ml = ML_targets_o[idx]['boxes']
            # 提取图片基本信息
            img_cc = CC_samples.tensors[idx]
            h1,w1 = img_cc.shape[1], img_cc.shape[2]
            img_ml = ML_samples.tensors[idx]
            h2,w2 = img_ml.shape[1], img_ml.shape[2]

            # 将框从 cxcywh 转换为 xyxy 格式
            cc_boxes = box_cxcywh_to_xyxy(cc_boxes)
            ml_boxes = box_cxcywh_to_xyxy(ml_boxes)
            # gt_cc = [box_cxcywh_to_xyxy(g) for g in gt_cc]
            # gt_ml = [box_cxcywh_to_xyxy(g) for g in gt_ml]
            cc_pos_img, cc_neg_img = split_pos_neg(
                cc_boxes, gt_cc, img_cc, w1, h1, pos_th=0.1, neg_th=0.1
            )

            ml_pos_img, ml_neg_img = split_pos_neg(
                ml_boxes, gt_ml, img_ml, w2, h2, pos_th=0.1, neg_th=0.1
            )

            fe_cc_pos = []
            fe_cc_neg = []
            fe_ml_pos = []
            fe_ml_neg = []
            patches_pos_cc = []
            patches_neg_cc = []
            patches_pos_ml = []
            patches_neg_ml = []
            # 如果正负样本不为空，分别处理它们
            if len(cc_pos_img) != 0:
                for i in range(len(cc_pos_img)):
                    img = CC_samples.tensors[idx]
                    h1, w1 = img.shape[1], img.shape[2]

                    # 选择正样本
                    rand_pos_idx = random.randint(0, len(cc_pos_img) - 1)
                    pos_box = cc_pos_img[rand_pos_idx]
                 
                    de_pos_cc = rescale_bboxes(pos_box, w1, h1)
                
                    fe_cc_pos.append(pos_box)

                    xmin_p, ymin_p, xmax_p, ymax_p = pos_box.astype(int)

                    # 裁剪正样本
                    cc_cropped_pos = img_cc[:, ymin_p:ymax_p, xmin_p:xmax_p]
                    resize = T.Resize((128, 128), antialias=True)
                    cc_cropped_pos = cc_cropped_pos.float()
                    cc_cropped_pos = resize(cc_cropped_pos)
                    patches_pos_cc.append(cc_cropped_pos)

            if len(cc_neg_img) != 0:
                for i in range(len(cc_neg_img)):
                    
                    # 选择负样本
                    rand_neg_idx = random.randint(0, len(cc_neg_img) - 1)
                    neg_box = cc_neg_img[rand_neg_idx]
                    fe_cc_neg.append(neg_box)
                    xmin_n, ymin_n, xmax_n, ymax_n = neg_box.astype(int)
                    if xmax_n <= xmin_n or ymax_n <= ymin_n:
                        continue
                    else:
                    # 裁剪负样本
                        cc_cropped_neg = img_cc[:, ymin_n:ymax_n, xmin_n:xmax_n]
                        cc_cropped_neg = cc_cropped_neg.float()
                        resize = T.Resize((128, 128), antialias=True)
                        cc_cropped_neg = resize(cc_cropped_neg)
                        patches_neg_cc.append(cc_cropped_neg)
                    
            # 输出处理后的patch数量，便于调试
            # 处理正样本
            if len(ml_pos_img) > 0:
                for i in range(len(ml_pos_img)):
                    img = ML_samples.tensors[idx]
                    h2, w2 = img.shape[1], img.shape[2]

                    # 随机选择一个正样本
                    rand_pos_idx = random.randint(0, len(ml_pos_img) - 1)
                    pos_box = ml_pos_img[rand_pos_idx]
                    # print(type(pos_box))
                    de_pos_ml = rescale_bboxes(pos_box, w2, h2)

                    # 添加正样本框
                    fe_ml_pos.append(pos_box)

                    # 获取正样本的坐标
                    xmin_p, ymin_p, xmax_p, ymax_p = pos_box.astype(int)

                    # 裁剪正样本
                    ml_cropped_pos = img_ml[:, ymin_p:ymax_p, xmin_p:xmax_p]
                    resize = T.Resize((128, 128), antialias=True)
                    ml_cropped_pos = ml_cropped_pos.float()
                    ml_cropped_pos = resize(ml_cropped_pos)
                    patches_pos_ml.append(ml_cropped_pos)

            # 处理负样本
            if len(ml_neg_img) > 0:
                # print(len(ml_neg_img))
                for i in range(len(ml_neg_img)):
                    img = ML_samples.tensors[idx]
                    h2, w2 = img.shape[1], img.shape[2]

                    # 随机选择一个负样本
                    rand_neg_idx = random.randint(0, len(ml_neg_img) - 1)
                    neg_box = ml_neg_img[rand_neg_idx]
                    # print(f"选中的框类型: {type(neg_box)}")
                    # de_neg_ml = rescale_bboxes(neg_box, w2, h2)

                    # 添加负样本框
                    fe_ml_neg.append(neg_box)

                    # 获取负样本的坐标
                    xmin_n, ymin_n, xmax_n, ymax_n = neg_box.astype(int)
                    if xmax_n <= xmin_n or ymax_n <= ymin_n:
                        continue
                    else:
                        # 裁剪负样本
                        ml_cropped_neg = img_ml[:, ymin_n:ymax_n, xmin_n:xmax_n]
                        ml_cropped_neg = ml_cropped_neg.float()
                        resize = T.Resize((128, 128), antialias=True)
                        ml_cropped_neg = resize(ml_cropped_neg)
                        patches_neg_ml.append(ml_cropped_neg)

            M,N = min(len(patches_pos_cc), len(patches_pos_ml)),min(len(patches_neg_cc),len(patches_neg_ml))
            len_PG += (M+N)

            if len(patches_pos_cc) == 0 or len(patches_pos_ml) == 0 or len(patches_neg_cc) == 0 or len(patches_neg_ml) == 0:
                loss = torch.tensor(0.0, device=device)
            else:
                pos_cc_sel = randSelect_tensor(list_to_tensor(patches_pos_cc, device), M)
                pos_ml_sel = randSelect_tensor(list_to_tensor(patches_pos_ml, device), M)
                neg_cc_sel = list_to_tensor(patches_neg_cc, device)[:N]  
                neg_ml_sel = list_to_tensor(patches_neg_ml, device)[:N] 
                patches_pos_cc = extract_patch_features(pos_cc_sel, model, device)
                patches_pos_ml = extract_patch_features(pos_ml_sel, model, device)
                patches_neg_cc = extract_patch_features(neg_cc_sel, model, device)
                patches_neg_ml = extract_patch_features(neg_ml_sel, model, device)
                cc_sel = torch.concat((patches_pos_cc, patches_neg_cc), dim=0)
                ml_sel = torch.concat((patches_pos_ml, patches_neg_ml), dim=0)
                loss = umcl_loss_paired(cc_sel, ml_sel, M, N, tau=0.07)
            L_umcl += loss

        if len_PG > 0:
            L_umcl = L_umcl / len_PG
        else:
            L_umcl = torch.zeros((), device=device)
        # print("L_umcl",L_umcl)
        loss_dict= criterion(cc_out,ml_out,CC_targets,ML_targets)


        # cc_loss_dict=criterion(cc_out, CC_targets)
        # ml_loss_dict=criterion(ml_out,ML_targets)

        # loss_dict={}
        # for key in cc_loss_dict:
        #     # 确保两个字典都有这个键
        #     if key in ml_loss_dict:
        #         # 取平均值
        #         loss_dict[key] = (cc_loss_dict[key] + ml_loss_dict[key]) / 2
        # loss_dict = criterion(cc_out,cc_merge_text,cc_encoder_fea,ml_out,ml_merge_text,ml_encoder_fea,CC_targets,ML_targets,cc_crop_fea,ml_crop_fea)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)+L_umcl*0.1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        # 检查梯度
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is None:
        #             print(f"警告：参数 {name} 的梯度为 None（未参与计算）")
        #         else:
        #             print(f"参数 {name} 的梯度已计算（范数：{param.grad.norm().item():.4f}）")

        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
