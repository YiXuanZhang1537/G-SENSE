"""COCO-style CBIS dataset for paired CC/MLO training."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from torchvision.datasets import CocoDetection as TorchCocoDetection

import datasets.transforms as T
from .raw_coco import CocoDetection as PairedCocoDetection


class ValCocoDetection(TorchCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        image, annotations = super().__getitem__(idx)
        target = {"image_id": self.ids[idx], "annotations": annotations}
        image, target = self.prepare(image, target)
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target


class CocoDetection(PairedCocoDetection):
    def __init__(self, img_folder, ann_file, roi_folder=None, mass_list=None,
                 nomass_list=None, mass_unlabel_list=None, transforms=None,
                 return_masks=False):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.roi_folder = Path(roi_folder) if roi_folder else None
        self.mass_list = self._read_list(mass_list)
        self.nomass_list = self._read_list(nomass_list)
        self.mass_unlabel_list = self._read_list(mass_unlabel_list)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _read_list(path):
        if not path:
            return set()
        with open(path, encoding="utf-8") as handle:
            return {line.strip().replace("\\", "/") for line in handle if line.strip()}

    def _label_for(self, file_name):
        normalized = file_name.replace("\\", "/")
        candidates = {normalized, os.path.basename(normalized)}
        if candidates & self.mass_list or candidates & self.mass_unlabel_list:
            return 1
        if candidates & self.nomass_list:
            return 0
        return -1

    def _read_roi(self, image_id, size):
        if self.roi_folder is None:
            return torch.ones((size[1], size[0]), dtype=torch.uint8)
        file_name = self.coco.loadImgs(image_id)[0]["file_name"]
        path = self.roi_folder / file_name
        if not path.exists():
            return torch.ones((size[1], size[0]), dtype=torch.uint8)
        roi = np.asarray(Image.open(path).convert("L")) > 0
        return torch.from_numpy(roi.astype(np.uint8))

    def __getitem__(self, idx):
        cc_img, ml_img, cc_id, ml_id, cc_annotations, ml_annotations, _ = super().__getitem__(idx)
        cc_target = {"image_id": cc_id, "annotations": cc_annotations}
        ml_target = {"image_id": ml_id, "annotations": ml_annotations}
        cc_img, cc_target = self.prepare(cc_img, cc_target)
        ml_img, ml_target = self.prepare(ml_img, ml_target)

        cc_img_original, ml_img_original = cc_img.copy(), ml_img.copy()
        cc_target_original = {key: value.clone() if torch.is_tensor(value) else value
                              for key, value in cc_target.items()}
        ml_target_original = {key: value.clone() if torch.is_tensor(value) else value
                              for key, value in ml_target.items()}
        cc_roi = self._read_roi(cc_id, cc_img.size)
        ml_roi = self._read_roi(ml_id, ml_img.size)

        if self._transforms is not None:
            cc_img, cc_target = self._transforms(cc_img, cc_target)
            ml_img, ml_target = self._transforms(ml_img, ml_target)

        cc_img, cc_target = self.normalize(cc_img, cc_target)
        ml_img, ml_target = self.normalize(ml_img, ml_target)
        cc_img_original = T.ToTensor()(cc_img_original)
        ml_img_original = T.ToTensor()(ml_img_original)

        cc_boxes, ml_boxes = [], []
        for cc_box, cc_box_id in zip(cc_target["boxes"], cc_target["bbox_ids"]):
            for ml_box, ml_box_id in zip(ml_target["boxes"], ml_target["bbox_ids"]):
                if cc_box_id == ml_box_id:
                    cc_boxes.append(cc_box)
                    ml_boxes.append(ml_box)

        cc_name = self.coco.loadImgs(cc_id)[0]["file_name"]
        ml_name = self.coco.loadImgs(ml_id)[0]["file_name"]
        return (
            cc_img, ml_img, cc_target, ml_target,
            cc_boxes, ml_boxes,
            cc_img_original, ml_img_original,
            cc_target_original, ml_target_original,
            cc_roi, ml_roi,
            self._label_for(cc_name), self._label_for(ml_name),
        )


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        masks.append(torch.as_tensor(mask, dtype=torch.uint8).any(dim=2))
    return torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.uint8)


class ConvertCocoPolysToMask:
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        width, height = image.size
        annotations = [a for a in target["annotations"] if a.get("iscrowd", 0) == 0]
        boxes = torch.as_tensor([a["bbox"] for a in annotations], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        labels = torch.as_tensor([a["category_id"] for a in annotations], dtype=torch.int64)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        target = {
            "boxes": boxes[keep],
            "labels": labels[keep],
            "image_id": torch.tensor([target["image_id"]]),
            "area": torch.as_tensor([a.get("area", 0) for a in annotations], dtype=torch.float32)[keep],
            "iscrowd": torch.as_tensor([a.get("iscrowd", 0) for a in annotations], dtype=torch.int64)[keep],
            "orig_size": torch.as_tensor([height, width]),
            "size": torch.as_tensor([height, width]),
            "bbox_ids": torch.as_tensor([a.get("bbox_id", -1) for a in annotations], dtype=torch.int64)[keep],
        }
        if self.return_masks:
            masks = convert_coco_poly_to_mask([a["segmentation"] for a in annotations], height, width)
            target["masks"] = masks[keep]
        return image, target


def make_coco_transforms(image_set):
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ]),
            ),
        ])
    if image_set == "val":
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    raise ValueError(f"unknown image set: {image_set}")


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path does not exist: {root}"
    suffix = "train2017" if image_set == "train" else "val2017"
    image_dir = root / suffix
    ann_file = root / "annotations" / f"instances_{suffix}.json"
    if image_set == "train":
        return CocoDetection(
            image_dir, ann_file,
            roi_folder=getattr(args, "roi_path", None) or root / "roi_masks",
            mass_list=getattr(args, "mass_list", None),
            nomass_list=getattr(args, "nomass_list", None),
            mass_unlabel_list=getattr(args, "mass_unlabel_list", None),
            transforms=make_coco_transforms(image_set), return_masks=args.masks,
        )
    return ValCocoDetection(image_dir, ann_file, make_coco_transforms(image_set), args.masks)
