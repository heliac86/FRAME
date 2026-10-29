#!/usr/bin/env python3
"""Run FRAME reconstruction on one MRI volume using the local nesvor package.

This is a minimal public-facing *revision-runner-based* example, NOT yet a
validated reproduction of the paper's main 4x BraTS configuration.

Place in FRAME/scripts/run_frame_volume.py. No third-party dependencies are
needed for --dry-run; actual reconstruction needs the FRAME training env.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def output_model_path(volume: Path) -> Path:
    name = volume.name
    if name.endswith('.nii.gz'):
        return volume.with_name(name[:-7] + '.pt')
    if name.endswith('.nii'):
        return volume.with_name(name[:-4] + '.pt')
    raise ValueError('--output must end in .nii or .nii.gz')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path,
                        help='Preprocessed, degraded 3D MRI input (.nii/.nii.gz).')
    parser.add_argument('--output', required=True, type=Path,
                        help='Reconstructed 3D MRI output (.nii/.nii.gz).')
    parser.add_argument('--mask', type=Path, default=None,
                        help='Sample mask; defaults to input, as in the provided research runner.')
    parser.add_argument('--model-output', type=Path, default=None,
                        help='Volume-specific model output (.pt); defaults next to --output.')
    parser.add_argument('--device', default='0', help='GPU index passed to FRAME CLI.')
    parser.add_argument('--output-intensity-mean', type=float, default=308.0,
                        help='308 for BraTS in the supplied runner; UPENN used 290.')
    parser.add_argument('--ff-weight', type=float, default=0.0125,
                        help='Revision runner default (paper reports 0.0125).')
    parser.add_argument('--gating-lr-scale', type=float, default=1.0,
                        help='Revision runner default.')
    parser.add_argument('--n-iter', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print command only; do not touch data or invoke FRAME.')
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    mask = (args.mask or args.input).expanduser().resolve()
    model_out = ((args.model_output.expanduser().resolve()) if args.model_output
                 else output_model_path(out))

    if inp == out or inp == model_out or out == model_out:
        parser.error('Input, volume output, and model output must be distinct.')
    if args.ff_weight < 0 or args.gating_lr_scale <= 0:
        parser.error('--ff-weight must be >= 0 and --gating-lr-scale must be > 0.')
    if args.n_iter <= 0 or args.batch_size <= 0:
        parser.error('--n-iter and --batch-size must be positive.')

    cmd = [
        sys.executable, '-u', '-c', 'from nesvor.cli.main import main; main()',
        'reconstruct',
        '--input-stacks', str(inp),
        '--sample-mask', str(mask),
        '--output-volume', str(out),
        '--output-model', str(model_out),
        '--output-resolution', '1.0',
        '--output-intensity-mean', str(args.output_intensity_mean),
        '--no-transformation-optimization', '--registration', 'none',
        '--single-precision',
        '--patch-size', '48', '--n-patches', '8',
        '--n-samples', '256',
        '--weight-image', '1.0', '--delta', '0.05',
        '--diversity-loss-fn', 'variance', '--diversity-loss-space', 'raw',
        '--target-diversity-var', '0.06',
        '--n-iter', str(args.n_iter), '--batch-size', str(args.batch_size),
        '--learning-rate', '5e-3', '--gamma', '0.33',
        '--milestones', '0.5', '0.75', '0.9',
        '--seed', '42',
        '--gating-hidden-dim', '8', '--gating-activation', 'silu',
        '--gating-lr-scale', str(args.gating_lr_scale),
        '--gating-grad-clip', '0.1',
        '--weight-ff-loss', str(args.ff_weight), '--ff-alpha', '2',
        '--spatial-gating', '--device', str(args.device),
    ]
    print('Repository:', repo)
    print('Python:', sys.executable)
    print('Command:', shlex.join(cmd), flush=True)

    if args.dry_run:
        print('DRY RUN: no files created; reconstruction not executed.')
        return 0

    if not (repo / 'nesvor' / 'inr' / 'models.py').is_file():
        parser.error('Cannot find local FRAME package. Place script in FRAME/scripts/.')
    for label, path in [('input', inp), ('mask', mask)]:
        if not path.is_file():
            parser.error(f'{label} file not found: {path}')
    for path in [out, model_out]:
        if path.exists():
            parser.error(f'Output already exists; choose a new path: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    result = subprocess.run(cmd, cwd=str(repo), env=env, check=False)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
