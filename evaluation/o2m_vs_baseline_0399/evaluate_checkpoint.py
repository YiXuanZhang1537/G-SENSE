"""Evaluate one checkpoint with the settings used for the paired comparisons."""
import argparse
import json
from pathlib import Path
import compare as evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    evaluation.OUT = args.out_dir.resolve()
    evaluation.OUT.mkdir(parents=True, exist_ok=True)
    evaluation.torch.set_num_threads(4)
    gt = evaluation.load_gt(evaluation.GT)
    checkpoint = args.checkpoint.resolve()
    pred = evaluation.predict('checkpoint', gt, 'cuda', checkpoint)
    assert set(pred) == set(gt)
    stats = [(*evaluation.per_image_stats(evaluation.np.asarray(pred[n], evaluation.np.float32).reshape(-1, 5), gt[n], .1), len(gt[n])) for n in sorted(gt)]
    values = evaluation.vector(stats, list(range(len(stats))))
    ckpt = evaluation.torch.load(checkpoint, map_location='cpu', weights_only=False)
    result = dict(checkpoint=str(checkpoint), epoch=ckpt.get('epoch'), n_images=len(gt),
                  n_gt_boxes=sum(map(len, gt.values())), n_detections=sum(map(len, pred.values())),
                  device='cuda', resize_short_side=800, threshold=0., nms_iou=.1, evaluation_iou=.1,
                  gt=str(evaluation.GT), image_dir=str(evaluation.IMAGES),
                  froc_definition='Repository _compute_froc: mean recall at detection-ranked points with FP/image in [0.125,2].',
                  metrics={k:float(v) for k,v in zip(evaluation.LABELS,values)})
    evaluation.write_json(evaluation.OUT/'results.json', result)
    text = json.dumps(result, indent=2)
    (evaluation.OUT/'summary.txt').write_text(text+'\n')
    print(text, flush=True)


if __name__ == '__main__':
    main()
