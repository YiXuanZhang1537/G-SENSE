# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
import numpy as np

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F



class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer,num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.linear = nn.Linear(4096, 1024)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)


        src, mask = features[-1].decompose()
        assert mask is not None
        hs,encoder_hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])


        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def contrastive_loss(positive, negative, temperature=0.07):
    """计算对比损失 (没有 anchor)，仅基于正负样本对"""
    pos_similarity = torch.cosine_similarity(positive, positive, dim=-1) / temperature
    neg_similarity = torch.cosine_similarity(positive, negative, dim=-1) / temperature
    # 获取正负样本的数量
    M = positive.shape[0]
    N = negative.shape[0]

    # 计算正样本相似度
    pos_similarity = torch.cosine_similarity(positive, positive, dim=-1) / temperature

    # 计算负样本相似度
    neg_similarity = torch.cosine_similarity(positive, negative, dim=-1) / temperature

    # 计算损失
    loss = -torch.log(torch.exp(pos_similarity) / (torch.exp(pos_similarity) + torch.exp(neg_similarity)))

    # 归一化，除以 M^2
    loss = loss / (M ** 2)

    return loss.mean()  # 返回损失的均值

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def get_logits(self, image_features, text_features,logit_scale):
        image_features=F.normalize(image_features, dim=-1)
        text_features=F.normalize(text_features,dim=-1)
        logits_per_image = logit_scale *image_features @ text_features.T
        logits_per_text = logit_scale *text_features @ image_features.T
        return logits_per_image, logits_per_text



    def forward(self, cc_out,ml_out,CC_targets,ML_targets, all_patches_pos_cc,all_patches_neg_cc,all_patches_pos_ml,all_patches_neg_ml):
        # ,cc_crop_fea_pos,cc_crop_fea_neg,ml_crop_fea_pos,ml_crop_fea_neg
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # CC loss
        cc_outputs_without_aux = {k: v for k, v in cc_out.items() if k != 'aux_outputs'}
        logits1=self.logit_scale1.exp()
        logits2=self.logit_scale2.exp()

        # Retrieve the matching between the outputs of the last layer and the targets
        cc_indices = self.matcher(cc_outputs_without_aux, CC_targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        cc_num_boxes = sum(len(t["labels"]) for t in CC_targets)
        cc_num_boxes = torch.as_tensor([cc_num_boxes], dtype=torch.float, device=next(iter(cc_out.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(cc_num_boxes)
        cc_num_boxes = torch.clamp(cc_num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        cc_losses = {}
        for loss in self.losses:
            cc_losses.update(self.get_loss(loss, cc_out, CC_targets, cc_indices, cc_num_boxes))
        
        if len(all_patches_pos_cc) > 0 and len(all_patches_neg_cc) > 0:
            all_patches_pos_cc = torch.stack(all_patches_pos_cc)  # 正样本图像 patch
            all_patches_neg_cc = torch.stack(all_patches_neg_cc)  # 负样本图像 patch
            cc_loss_patch = contrastive_loss(all_patches_pos_cc, all_patches_neg_cc)  # 计算对比损失
        else:
            cc_loss_patch = torch.tensor(0.0)  # 或者选择一个默认值

        if len(all_patches_pos_ml) > 0 and len(all_patches_neg_ml) > 0:
            all_patches_pos_ml = torch.stack(all_patches_pos_ml)  # 正样本图像 patch
            all_patches_neg_ml = torch.stack(all_patches_neg_ml)  # 负样本图像 patch
            ml_loss_patch = contrastive_loss(all_patches_pos_ml, all_patches_neg_ml)  # 计算对比损失
        else:
            ml_loss_patch = torch.tensor(0.0)  # 或者选择一个默认值

        # 计算总损失
        loss_patch = (cc_loss_patch + ml_loss_patch) / 2

        # print(f"Total loss: {loss_patch.item()}")
        

        # MLO loss
        ml_outputs_without_aux = {k: v for k, v in ml_out.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        ml_indices = self.matcher(ml_outputs_without_aux, ML_targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        ml_num_boxes = sum(len(t["labels"]) for t in ML_targets)
        ml_num_boxes = torch.as_tensor([ml_num_boxes], dtype=torch.float, device=next(iter(ml_out.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(ml_num_boxes)
        ml_num_boxes = torch.clamp(ml_num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        ml_losses = {}
        for loss in self.losses:
            ml_losses.update(self.get_loss(loss, ml_out, ML_targets, ml_indices, ml_num_boxes))
        
        # ml_train_label = torch.arange(ml_merge_text.shape[0], device=ml_merge_text.device, dtype=torch.long)
        # ml_logits_per_img, ml_logits_per_text = self.get_logits(ml_merge_image.squeeze(1), ml_merge_text.squeeze(1),logits1)
        # ml_loss_cross = (F.cross_entropy(ml_logits_per_img, ml_train_label) + F.cross_entropy(ml_logits_per_text,
        #                                                                                      ml_train_label))/2
        # ml_losses.update({"loss_cross": ml_loss_cross/2})
        # print(ml_losses)

        losses = {}
        for key in cc_losses:
            # 确保两个字典都有这个键
            if key in ml_losses:
                # 取平均值
                losses[key] = (cc_losses[key] + ml_losses[key]) / 2
        losses.update({"loss_patch": loss_patch})
        # print("------------------")
        # print(losses)
        # losses = {key: value / 2 for key, value in losses.items()}

        # logits_per_cc, logits_per_ml = self.get_logits(cc_crop_fea, ml_crop_fea,logits2)
        # train_label = torch.arange(cc_crop_fea.shape[0], device=ml_crop_fea.device, dtype=torch.long)
        # print("train_label",train_label)
        # loss_view = F.cross_entropy(logits_per_cc, train_label) + F.cross_entropy(logits_per_ml,
        #                                                                           train_label)
        # losses.update({"loss_view": loss_view/10})

        # gat = CCPosGAT(in_dim=cc_crop_fea_pos.size(1), hidden_dim=256, out_dim=256, heads=8)
        # cc_crop_fea_pos_u, cc_pos_proto = gat(cc_crop_fea_pos)
        # ml_crop_fea_pos_u, ml_pos_proto = gat(ml_crop_fea_pos)
        # 计算正类和负类之间的对比损失
        # 1. 计算 cc_pos 和 cc_neg 之间的相似度
        # logits_per_cc_pos_neg, logits_per_ml_pos_neg = self.get_logits(cc_crop_fea_pos, cc_crop_fea_neg, logits2)

        # # 2. 计算 ml_pos 和 ml_neg 之间的相似度
        # logits_per_ml_pos_neg, logits_per_cc_pos_neg = self.get_logits(ml_crop_fea_pos, ml_crop_fea_neg, logits2)

        # cc_crop_fea_pos = torch.stack(cc_crop_fea_pos)
        # cc_crop_fea_neg = torch.stack(cc_crop_fea_neg)
        # ml_crop_fea_pos = torch.stack(ml_crop_fea_pos)
        # ml_crop_fea_neg = torch.stack(ml_crop_fea_neg)

        # print(cc_crop_fea_pos)

        # # 假设目标标签是每个样本的索引
        # train_label = torch.arange(cc_crop_fea_pos.size(0), device=logits_per_cc_pos_neg.device, dtype=torch.long)

        # # 计算 cc_pos 和 cc_neg 之间的损失
        # loss_cc_pos_neg = F.cross_entropy(logits_per_cc_pos_neg, train_label)

        # # 计算 ml_pos 和 ml_neg 之间的损失
        # loss_ml_pos_neg = F.cross_entropy(logits_per_ml_pos_neg, train_label)

        # # 最终损失：取正类和负类损失的平均
        # loss_view = (loss_cc_pos_neg + loss_ml_pos_neg) / 2

        # losses.update({"loss_view": loss_view/10})

        
                


        cc_loss_aux = {}
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in cc_out:
            for i, aux_outputs in enumerate(cc_out['aux_outputs']):
                indices = self.matcher(aux_outputs, CC_targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, CC_targets, indices, cc_num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    cc_loss_aux.update(l_dict)
        ml_loss_aux = {}
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in ml_out:
            for i, aux_outputs in enumerate(ml_out['aux_outputs']):
                indices = self.matcher(aux_outputs, ML_targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, ML_targets, indices, ml_num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    ml_loss_aux.update(l_dict)
        losses_aux = {}
        for key in cc_loss_aux:
            # 确保两个字典都有这个键
            if key in ml_loss_aux:
                # 取平均值
                losses_aux[key] = (cc_loss_aux[key] + ml_loss_aux[key]) / 2
        # losses_aux = {key: value / 2 for key, value in losses_aux.items()}
        losses.update(losses_aux)
        # print(losses)

        return losses



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    num_classes=2
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)


    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    weight_dict.update({'loss_cross':1,'loss_view':1})

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
