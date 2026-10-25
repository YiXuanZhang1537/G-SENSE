"""Reproduce both predictions and compare using paired patient-level resampling."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path('/mnt/sda/Archive/Big-Data')
sys.path.insert(0, str(ROOT / 'baseline+o2m'))
import numpy as np
import torch
import models.backbone as backbone
from models import build_model
from test_cbis_bootstrap import infer_one, load_gt, per_image_stats, metrics_from_pool, FIXED_FPS
from PIL import Image

OUT = Path(__file__).resolve().parent
GT = Path('/mnt/sda/Medical_Dataset/Public/CBIS-DDSM/cbis_test/cbis_boxes_test.json')
IMAGES = GT.parent / 'images2'
LABELS = ['FROC-AUC (repository definition)', 'CPM'] + [f'Sensitivity@{x}FP/image' for x in FIXED_FPS]

def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def predict(label, gt, device, ckpt_path):
    dest = OUT / f'predictions_{label}_{device}.json'
    pred = json.loads(dest.read_text()) if dest.exists() else {}
    if set(pred) == set(gt):
        return pred
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    args.device = device
    args.k_matches = getattr(args, 'k_matches', 5)
    # Full checkpoint supplies every backbone tensor; no pretrained download needed.
    backbone.is_main_process = lambda: False
    model, _, _ = build_model(args)
    if 'linear.weight' not in ckpt['model']:
        # Original DETR has no unused cross-view projection layer.
        del model.linear
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device)
    model.eval()
    start = time.monotonic()
    with torch.inference_mode():
        for i, name in enumerate(sorted(gt)):
            if name in pred:
                continue
            path = IMAGES / (name + '.png')
            if not path.exists():
                matches = list(IMAGES.glob(name + '.*'))
                assert len(matches) == 1, (name, matches)
                path = matches[0]
            with Image.open(path) as im:
                pred[name] = infer_one(model, im.convert('RGB'), torch.device(device), 0., .1, size=800).tolist()
            if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(gt):
                write_json(dest, pred)
                print(f'{label}: {i+1}/{len(gt)}, elapsed {time.monotonic()-start:.1f}s', flush=True)
    return pred

def vector(stats, idx):
    s = np.concatenate([stats[i][0] for i in idx])
    t = np.concatenate([stats[i][1] for i in idx])
    ng = sum(stats[i][2] for i in idx)
    m = metrics_from_pool(s, t, ng, len(idx))
    return np.array([m['froc_auc'], m['cpm']] + [m['sensitivity'][x] for x in FIXED_FPS])

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--n-boot', type=int, default=5000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--baseline-checkpoint', type=Path, default=ROOT/'baseline/detr-cbis/checkpoint0399.pth')
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = [ROOT/'baseline+o2m/detr-cbis/checkpoint0399.pth', args.baseline_checkpoint.resolve()]
    torch.set_num_threads(args.threads)
    gt = load_gt(GT)
    names = sorted(gt)
    print(f'Device: {args.device}; CUDA available: {torch.cuda.is_available()}', flush=True)
    predictions = [predict(label, gt, args.device, path) for label, path in zip(['baseline+o2m', 'baseline'], checkpoints)]
    stats = [[(*per_image_stats(np.asarray(p[n], np.float32).reshape(-1, 5), gt[n], .1), len(gt[n])) for n in names] for p in predictions]
    groups = {}
    for i, name in enumerate(names):
        patient = re.search(r'P_\d+', name).group()
        groups.setdefault(patient, []).append(i)
    clusters = list(groups.values())
    all_idx = list(range(len(names)))
    point = [vector(s, all_idx) for s in stats]
    diff = point[0] - point[1]
    rng = np.random.default_rng(42)
    boot = np.empty((args.n_boot, len(LABELS)))
    for b in range(args.n_boot):
        idx = [i for c in rng.integers(0, len(clusters), len(clusters)) for i in clusters[c]]
        boot[b] = vector(stats[0], idx) - vector(stats[1], idx)
        if (b+1) % 1000 == 0:
            print(f'Patient bootstrap {b+1}/{args.n_boot}', flush=True)
    # Under the equality null, exchange the two models' predictions for each
    # patient as a whole, preserving correlations between that patient's images.
    rng = np.random.default_rng(43)
    exceed = np.zeros(len(LABELS), dtype=int)
    for b in range(args.n_perm):
        swap = np.zeros(len(names), dtype=bool)
        for c, flag in zip(clusters, rng.integers(0, 2, len(clusters))):
            swap[c] = flag
        sa = [stats[int(swap[i])][i] for i in all_idx]
        sb = [stats[1-int(swap[i])][i] for i in all_idx]
        pd = vector(sa, all_idx) - vector(sb, all_idx)
        exceed += np.abs(pd) >= np.abs(diff) - 1e-12
        if (b+1) % 1000 == 0:
            print(f'Patient permutation {b+1}/{args.n_perm}', flush=True)
    pvalues = (exceed + 1) / (args.n_perm + 1)
    rows = []
    for k, metric in enumerate(LABELS):
        rows.append(dict(metric=metric, o2m=float(point[0][k]), baseline=float(point[1][k]),
                         difference=float(diff[k]), difference_ci95=np.percentile(boot[:,k], [2.5,97.5]).tolist(),
                         p_two_sided=float(pvalues[k])))
    metadata = dict(n_images=len(names), n_patients=len(clusters), n_gt_boxes=sum(map(len,gt.values())),
                    evaluation_iou=.1, nms_iou=.1, threshold=0., resize_short_side=800,
                    resampling_unit='patient', n_permutations=args.n_perm, n_bootstrap=args.n_boot,
                    bootstrap_seed=42, permutation_seed=43, comparison='baseline+o2m minus baseline',
                    primary_metric=LABELS[0], secondary_pvalues_adjusted=False,
                    froc_definition='Repository _compute_froc: mean recall at detection-ranked points with FP/image in [0.125,2]; not a trapezoidal area.',
                    checkpoints=[str(p) for p in checkpoints],
                    checkpoint_epochs=[torch.load(p, map_location='cpu', weights_only=False)['epoch'] for p in checkpoints],
                    gt=str(GT), image_dir=str(IMAGES), torch_version=torch.__version__, device=args.device, metrics=rows)
    write_json(OUT / 'results.json', metadata)
    lines = [f'Comparison: {checkpoints[0]} vs {checkpoints[1]}',
             f"Stored epochs: {metadata['checkpoint_epochs']}",
             f'{len(names)} images, {len(clusters)} patients; patient-level paired permutation (two-sided), {args.n_perm} permutations.',
             f'Patient bootstrap 95% CI, {args.n_boot} resamples; IoU=0.1, NMS=0.1, Resize=800, score threshold=0.',
             'FROC-AUC follows repository definition (mean sensitivity at detection-ranked points in FP/image range [0.125,2]).',
             'Metric | o2m | baseline | difference | 95% CI of difference | p (two-sided, unadjusted)']
    for r in rows:
        lines.append(f"{r['metric']} | {r['o2m']:.6f} | {r['baseline']:.6f} | {r['difference']:+.6f} | {r['difference_ci95']} | {r['p_two_sided']:.6g}")
    (OUT / 'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)

if __name__ == '__main__':
    main()
