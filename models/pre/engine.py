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


import torch
import numpy as np

def rescale_bboxes(out_bbox, w, h):
    # 先使用 .detach() 断开与计算图的联系，然后调用 .cpu() 和 .numpy()
    out_bbox_cpu = out_bbox.detach().cpu().numpy()
    
    # 将 out_bbox 扩展到实际图像坐标系
    b = out_bbox_cpu * np.array([w, h, w, h], dtype=np.float32)
    b[0] = min(max(b[0],0),w)
    b[1] = min(max(b[1],0),h)
    b[2] = min(max(b[2],0),w)
    b[3] = min(max(b[3],0),h)
    return b


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

def extract_patch_features(patches, model, device):
    if not patches:           # 空列表直接返回
        return []
    nt = nested_tensor_from_tensor_list(patches).to(device)
    
    xs, pos = model.backbone(nt)
    src, mask = xs[-1].decompose()
    _, enc = model.transformer(model.input_proj(src), mask, model.query_embed.weight, pos[-1])
    feat = model.linear(torch.flatten(enc, start_dim=1))
    return [f.detach().cpu() for f in feat]

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

    for CC_samples,ML_samples,CC_targets,ML_targets,cc_img_list,ml_img_list,cc_box_list,ml_box_list,CC_samples_o, ML_samples_o,CC_targets_o,ML_targets_o in metric_logger.log_every(data_loader, print_freq, header):
        # 一次进来四个
        CC_samples = CC_samples.to(device)
        ML_samples = ML_samples.to(device)
        CC_samples_o = CC_samples_o.to(device)
        ML_samples_o = ML_samples_o.to(device)
        CC_targets_o = [{k: v.to(device) for k, v in t.items()} for t in CC_targets_o]
        ML_targets_o=[{k: v.to(device) for k, v in  t.items()} for t in ML_targets_o]   
        # print(CC_samples_o.tensors.shape)
        # print(ML_samples_o.tensors.shape)
        # print(CC_targets_o)
        # print(ML_targets_o)
        # print("*",cc_box_list)
        # print("*",ml_box_list)
        CC_targets = [{k: v.to(device) for k, v in t.items()} for t in CC_targets]
        ML_targets=[{k: v.to(device) for k, v in t.items()} for t in ML_targets]
        # text_embedding =torch.stack(text_embedding).to(device).squeeze(1)
        # know_embedding=torch.stack(know_embedding).to(device).squeeze(1)
        cc_img_list=cc_img_list.to(device)
        ml_img_list=ml_img_list.to(device)

        cc_crop_xs,cc_crop_pos=model.backbone(cc_img_list)
        ml_crop_xs,ml_crop_pos=model.backbone(ml_img_list)
        cc_src, cc_mask = cc_crop_xs[-1].decompose()
        ml_src, ml_mask = ml_crop_xs[-1].decompose()
        _, cc_crop_encoder = model.transformer(model.input_proj(cc_src), cc_mask, model.query_embed.weight, cc_crop_pos[-1])
        _, ml_crop_encoder = model.transformer(model.input_proj(ml_src), ml_mask, model.query_embed.weight, ml_crop_pos[-1])
        cc_crop_fea= model.linear(torch.flatten(cc_crop_encoder, start_dim=1))
        ml_crop_fea= model.linear(torch.flatten(ml_crop_encoder, start_dim=1))


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
        # 初始化存储所有正负类框的列表
        cc_pos = []  # 存储所有的 CC 正类框
        ml_pos = []  # 存储所有的 ML 正类框
        cc_neg = []  # 存储所有的 CC 负类框
        ml_neg = []  # 存储所有的 ML 负类框

        all_patches_pos_cc = []
        all_patches_neg_cc = []
        all_patches_pos_ml = []
        all_patches_neg_ml = []

        # 遍历所有批次
        for batch_idx in range(B):
            cc_pos_img = []  # 当前批次中图片的正类框
            ml_pos_img = []  # 当前批次中图片的正类框
            cc_neg_img = []  # 当前批次中图片的负类框
            ml_neg_img = []  # 当前批次中图片的负类框

            # 获取当前批次中第 batch_idx 张图片的框
            cc_boxes = cc_out_per[batch_idx]  # 获取当前图片的 cc_box
            ml_boxes = ml_out_per[batch_idx]  # 获取当前图片的 ml_box
            # print("cc_boxes",cc_boxes[0])
            # print("ml_boxes",ml_boxes[0])
            
            gt_cc = cc_box_list[batch_idx]
            gt_ml = ml_box_list[batch_idx]
            # print("gt_cc",gt_cc)
            # print("gt_ml",gt_ml)

            # 将框从 cxcywh 转换为 xyxy 格式
            cc_boxes = box_cxcywh_to_xyxy(cc_boxes)
            ml_boxes = box_cxcywh_to_xyxy(ml_boxes)
            gt_cc = [box_cxcywh_to_xyxy(g) for g in gt_cc]
            gt_ml = [box_cxcywh_to_xyxy(g) for g in gt_ml]

            # 对比当前图片的所有目标框
            for bbox_cc in cc_boxes:
                for gt_bbox in gt_cc:
                    iou = cal_iou_xyxy(bbox_cc, gt_bbox)
                    if iou >= 0.1:
                        cc_pos_img.append(bbox_cc)  # 正类
                    else:
                        if bbox_cc[0] < 0 or bbox_cc[1] < 0 or bbox_cc[2] > 1 or bbox_cc[3] > 1:
                            continue
                        else:
                            cc_neg_img.append(bbox_cc)  # 负类

            for bbox_mlo in ml_boxes:
                for gt_bbox in gt_ml:
                    iou = cal_iou_xyxy(bbox_mlo, gt_bbox)
                    if iou >= 0.1:
                        ml_pos_img.append(bbox_mlo)  # 正类
                    else:
                        if bbox_mlo[0] < 0 or bbox_mlo[1] < 0 or bbox_mlo[2] > 1 or bbox_mlo[3] > 1:
                            continue
                        else:
                            ml_neg_img.append(bbox_mlo)  # 负类

            # 输出每个批次中每张图片的正负类框数量
            # print(f"Batch {batch_idx + 1} - CC Positive: {len(cc_pos_img)}, CC Negative: {len(cc_neg_img)}")
            # print(f"Batch {batch_idx + 1} - ML Positive: {len(ml_pos_img)}, ML Negative: {len(ml_neg_img)}")

            fe_cc_pos = []
            fe_cc_neg = []
            fe_ml_pos = []
            fe_ml_neg = []
            patches_pos_cc = []
            patches_neg_cc = []
            patches_pos_ml = []
            patches_neg_ml = []
            # 对每个批次中的每张图片进行正负类框的裁剪处理  

            if len(cc_pos_img) != 0 and len(cc_neg_img) != 0:
                min_len = min(len(cc_pos_img), len(cc_neg_img))

                # 对最小长度进行裁剪处理
                for i in range(min_len):
                    img = CC_samples_o.tensors[batch_idx]
                    h1,w1 = img.shape[1], img.shape[2]
                    # print("w1,h1",w1,h1)
                    # print(cc_neg_img[i])
                    de_pos_cc = rescale_bboxes(cc_pos_img[i], w1, h1)
                    de_neg_cc = rescale_bboxes(cc_neg_img[i], w1, h1)
                    fe_cc_pos.append(de_pos_cc)
                    fe_cc_neg.append(de_neg_cc)


                    xmin_p, ymin_p, xmax_p, ymax_p = de_pos_cc.astype(int)
                    xmin_n, ymin_n, xmax_n, ymax_n = de_neg_cc.astype(int)
                    # print("cc",xmin_p,ymin_p,xmax_p,ymax_p)
                    # print("cc",xmin_n,ymin_n,xmax_n,ymax_n)

                    # 裁剪正样本
                    cc_cropped_pos = img[:, ymin_p:ymax_p, xmin_p:xmax_p]
                    resize = T.Resize((128, 128), antialias=True)
                    cc_cropped_pos = cc_cropped_pos.float()
                    cc_cropped_pos = resize(cc_cropped_pos)
                    patches_pos_cc.append(cc_cropped_pos)

                    # 裁剪负样本
                    cc_cropped_neg = img[:, ymin_n:ymax_n, xmin_n:xmax_n]
                    resize = T.Resize((128, 128), antialias=True)
                    cc_cropped_neg = cc_cropped_neg.float()
                    cc_cropped_neg = resize(cc_cropped_neg)
                    # de_cc_neg = nested_tensor_from_tensor_list(cc_cropped_neg)
                    # cc_crop_xs_neg, cc_crop_pos_neg = model.backbone(de_cc_neg)
                    # cc_src_neg, cc_mask_neg = cc_crop_xs_neg[-1].decompose()
                    # _, cc_crop_encoder_neg = model.transformer(model.input_proj(cc_src_neg), cc_mask_neg, model.query_embed.weight, cc_crop_pos_neg[-1])
                    # cc_crop_fea_neg = model.linear(torch.flatten(cc_crop_encoder_neg, start_dim=1))

                    patches_neg_cc.append(cc_cropped_neg)
                # print(f"正类patch数: {len(patches_pos_cc)}, 负类patch数: {len(patches_neg_cc)}")
                # print(len(fe_cc_pos))
                # print(len(fe_cc_neg))

            if len(ml_pos_img) != 0 and len(ml_neg_img) != 0:
                min_len = min(len(ml_pos_img), len(ml_neg_img))

                # 对最小长度进行裁剪处理
                for i in range(min_len):
                    img = ML_samples_o.tensors[batch_idx]
                    h2,w2 = img.shape[1], img.shape[2]
                    # print("w2,h2",w2,h2)
                    de_pos_ml = rescale_bboxes(ml_pos_img[i], w2, h2)
                    de_neg_ml = rescale_bboxes(ml_neg_img[i], w2, h2)
                    fe_ml_pos.append(de_pos_ml)
                    fe_ml_neg.append(de_neg_ml)


                    xmin_p, ymin_p, xmax_p, ymax_p = de_pos_ml.astype(int)
                    xmin_n, ymin_n, xmax_n, ymax_n = de_neg_ml.astype(int)
                    # print(xmin_p, ymin_p, xmax_p, ymax_p)
                    # print(xmin_n, ymin_n, xmax_n, ymax_n)

                    # 裁剪正样本
                    ml_cropped_pos = img[:, ymin_p:ymax_p, xmin_p:xmax_p]
                    resize = T.Resize((128, 128), antialias=True)
                    ml_cropped_pos = ml_cropped_pos.float()
                    ml_cropped_pos = resize(ml_cropped_pos)
                    # de_ml_pos = nested_tensor_from_tensor_list(ml_cropped_pos)

                    # ml_crop_xs_pos, ml_crop_pos_pos = model.backbone(de_ml_pos)
                    
                    # ml_src_pos, ml_mask_pos = ml_crop_xs_pos[-1].decompose()
                    
                    # _, ml_crop_encoder_pos = model.transformer(model.input_proj(ml_src_pos), ml_mask_pos, model.query_embed.weight, ml_crop_pos_pos[-1])
                    
                    # ml_crop_fea_pos = model.linear(torch.flatten(ml_crop_encoder_pos, start_dim=1))
                    
                    patches_pos_ml.append(ml_cropped_pos)

                    # 裁剪负样本
                    ml_cropped_neg = img[:, ymin_n:ymax_n, xmin_n:xmax_n]
                    resize = T.Resize((128, 128), antialias=True)
                    ml_cropped_neg = ml_cropped_neg.float()
                    ml_cropped_neg = resize(ml_cropped_pos)
                    # de_ml_neg = nested_tensor_from_tensor_list(ml_cropped_neg)

                    # ml_crop_xs_neg, ml_crop_pos_neg = model.backbone(de_ml_neg)

                    # ml_src_neg, ml_mask_neg = ml_crop_xs_neg[-1].decompose()
                    
                    # _, ml_crop_encoder_neg = model.transformer(model.input_proj(ml_src_neg), ml_mask_neg, model.query_embed.weight, ml_crop_pos_neg[-1])
                    
                    # ml_crop_fea_neg = model.linear(torch.flatten(ml_crop_encoder_neg, start_dim=1))
                    
                    patches_neg_ml.append(ml_cropped_neg)
                # print(f"正类patch数: {len(patches_pos_ml)}, 负类patch数: {len(patches_neg_ml)}")
                # print(len(fe_ml_pos))
                # print(len(fe_ml_neg))
            all_patches_pos_cc += extract_patch_features(patches_pos_cc, model, device)
            all_patches_pos_ml += extract_patch_features(patches_pos_ml, model, device)
            all_patches_neg_cc += extract_patch_features(patches_neg_cc, model, device)
            all_patches_neg_ml += extract_patch_features(patches_neg_ml, model, device)
        # print(f"总共CC正类patch数: {len(all_patches_pos_cc)}")
        # print(f"总共CC负类patch数: {len(all_patches_neg_cc)}")
        # print(f"总共ML正类patch数: {len(all_patches_pos_ml)}")
        # print(f"总共ML负类patch数: {len(all_patches_neg_ml)}")
              
        loss_dict = criterion(cc_out,ml_out,CC_targets,ML_targets,all_patches_pos_cc,all_patches_neg_cc,all_patches_pos_ml,all_patches_neg_ml)

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
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

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
