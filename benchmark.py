"""Measure checkpoint forward FLOPs (multiply-add = 2) and batch-one FPS."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

import models.backbone as backbone
from models.detr import DETR
from models.transformer import build_transformer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--sizes', nargs='+', default=['800x800', '800x1333'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--output', type=Path, default=Path('evaluation/benchmark/results.json'))
    options = parser.parse_args()
    torch.set_num_threads(options.threads)
    torch.manual_seed(42)
    if options.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu or run on a GPU host.')
    checkpoint = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    # All backbone tensors come from the checkpoint; avoid pretrained downloads.
    backbone.is_main_process = lambda: False
    model = DETR(backbone.build_backbone(args), build_transformer(args),
                 checkpoint['model']['class_embed.weight'].shape[0] - 1,
                 args.num_queries, aux_loss=args.aux_loss)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(options.device).eval().requires_grad_(False)
    cpu_name = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu_name = next((line.split(':', 1)[1].strip() for line in
                         Path('/proc/cpuinfo').read_text().splitlines()
                         if line.startswith('model name')), cpu_name)
    result = dict(checkpoint=str(options.checkpoint.resolve()), epoch=checkpoint.get('epoch'),
                  device=options.device, hardware=torch.cuda.get_device_name() if options.device == 'cuda' else cpu_name,
                  torch_version=torch.__version__, threads=options.threads, precision='float32',
                  batch_size=1, warmup=options.warmup, iterations=options.iterations,
                  parameters=sum(p.numel() for p in model.parameters()),
                  scope='Model forward only; synthetic input; excludes preprocessing, transfer and postprocessing.',
                  flop_convention='multiply-add = 2 FLOPs; PyTorch FlopCounterMode registered operators (primarily convolution and matrix multiplication); elementwise operations excluded.', measurements=[])

    def sync():
        if options.device == 'cuda':
            torch.cuda.synchronize()

    with torch.inference_mode():
        for size in options.sizes:
            height, width = map(int, size.split('x'))
            x = torch.randn(1, 3, height, width, device=options.device)
            with FlopCounterMode(display=False) as counter:
                model(x)
            for _ in range(options.warmup):
                model(x)
            sync()
            elapsed = []
            for _ in range(options.iterations):
                start = time.perf_counter()
                model(x)
                sync()
                elapsed.append(time.perf_counter() - start)
            mean = statistics.mean(elapsed)
            measurement = dict(input_shape=list(x.shape), gflops=counter.get_total_flops() / 1e9,
                               fps=1 / mean, mean_latency_ms=mean * 1000,
                               median_latency_ms=statistics.median(elapsed) * 1000,
                               latency_std_ms=statistics.pstdev(elapsed) * 1000,
                               counted_operations={str(op): value for op, value in counter.get_flop_counts()['Global'].items()})
            result['measurements'].append(measurement)
            print(json.dumps(measurement), flush=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved to {options.output}', flush=True)


if __name__ == '__main__':
    main()
