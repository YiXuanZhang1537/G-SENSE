# All

`All` is the integrated experiment branch for mammography mass detection.
It combines the useful pieces from the repository's earlier experiments:

- paired CC and MLO training on COCO-style CBIS annotations;
- DETR box detection with optional mask prediction;
- one-to-one, one-to-many, and Gaussian one-to-many matching;
- optional image-level mass/no-mass supervision;
- optional cross-view patch contrastive learning;
- COCO-style evaluation, FROC evaluation, batch prediction, and visualisation.

## Dataset layout

The dataset root is expected to contain:

```text
dataset/
  train2017/
  val2017/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  roi_masks/                         # optional
```

Every image entry in the annotation JSON needs `view_id` and `view`. Each
`view_id` must have exactly one `CC` image and one `MLO` image. The annotation
records may include `bbox_id`; equal IDs are used to build paired patches.

Optional image-level lists contain one `file_name` per line:

```text
mass.list       # label 1
nomass.list     # label 0
mass_unlabel.list  # label 1, weak image-level label
```

## Training

Detection, semi-supervised, and contrastive terms are enabled by their loss
weights. When the optional label lists are omitted, the semi-supervised term is
automatically zero. When no valid positive and negative patches exist in a
batch, the contrastive term is zero for that batch.

```bash
python main.py \
  --coco_path /path/to/dataset \
  --output_dir outputs/all \
  --epochs 300 \
  --batch_size 2 \
  --mass_list /path/to/mass.list \
  --nomass_list /path/to/nomass.list
```

Useful ablations:

```bash
# Standard DETR matching
python main.py --coco_path /path/to/dataset --matching_mode o2o

# One-to-many matching without the Gaussian attraction term
python main.py --coco_path /path/to/dataset --matching_mode o2m

# Detection only
python main.py --coco_path /path/to/dataset \
  --semi_loss_coef 0 --contrast_loss_coef 0
```

## Inference and FROC

`predict_mv.py` performs batch inference and writes detection JSON files.
`predict_cam.py` provides visualisation. `froc.py` evaluates saved detections
against ground-truth boxes using sensitivity versus false positives per scan.

## Notes

The integrated training loop is in [engine.py](engine.py). It keeps the
contrastive feature graph connected to the detector, and the semi-supervised
loss uses model probabilities directly so it can contribute gradients.
`engine2.py` is retained as the earlier experimental implementation for
comparison; the main entry point uses `engine.py`.
