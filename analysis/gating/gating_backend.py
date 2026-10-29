"""Torch/NeSVoR adapter. Imported only for real checkpoint extraction."""
from __future__ import annotations
import copy
import importlib
import inspect
import importlib.util
import sys
from pathlib import Path
import numpy as np
from gating_common import sha_file, sample_indices, validate_weights, array_hash

def resolve_nesvor_source(explicit_root=None):
    """An editable install is optional; an explicit source checkout also works."""
    roots=[]
    if explicit_root:
        p=Path(explicit_root).expanduser().resolve()
        if not (p/'nesvor'/'inr'/'models.py').is_file():
            raise ValueError('--nesvor-root must be the directory CONTAINING nesvor/, '
                             f'not the nesvor directory itself: {p}')
        roots=[p]
    elif importlib.util.find_spec('nesvor') is None:
        # Only bounded nearby directories; do not search a user's home or drive.
        here=Path(__file__).resolve().parent
        roots=[p for p in [here.parent,here.parent.parent,Path.cwd(),Path.cwd().parent]
               if (p/'nesvor'/'inr'/'models.py').is_file()]
        roots=list(dict.fromkeys(roots))
        if len(roots)>1:
            raise RuntimeError('Multiple nearby NeSVoR checkouts found. Select one with --nesvor-root: '
                               +', '.join(map(str,roots)))
    if roots:
        sys.path.insert(0,str(roots[0]))
        importlib.invalidate_caches()
    if importlib.util.find_spec('nesvor') is None:
        raise RuntimeError('Python cannot find nesvor. Interpreter: '+sys.executable+
            '. Use the reconstruction environment and --nesvor-root /path/to/checkout '
            '(this directory must contain nesvor/inr/models.py). No pip install is required for a source checkout.')
    return str(roots[0]) if roots else 'installed/importable package'

def initialize(device, nesvor_root=None):
    source=resolve_nesvor_source(nesvor_root)
    import torch
    from nesvor.inr.models import INR
    from nesvor.image import load_volume
    module = importlib.import_module(INR.__module__)
    if not torch.cuda.is_available() or not str(device).startswith('cuda'):
        raise RuntimeError('Run extraction in the original CUDA/tinycudann environment. No CPU/backend fallback.')
    if getattr(module, 'USE_TORCH', True):
        raise RuntimeError('tinycudann backend is unavailable; refusing to change checkpoint backend')
    torch.cuda.set_device(torch.device(device))
    return torch, INR, load_volume, dict(
        torch_version=str(torch.__version__), cuda_version=str(torch.version.cuda),
        gpu=torch.cuda.get_device_name(torch.device(device)),
        models_path=str(Path(inspect.getfile(INR)).resolve()),
        python_executable=sys.executable, source_selection=source,
        models_sha256=sha_file(inspect.getfile(INR)), backend='tinycudann')

def reference_points(path, load_volume, torch, a, patient, modality):
    # Use the SAME native image API as sample.py; do not guess RAS/LPS conversions.
    volume = load_volume(path, device=torch.device('cpu'))
    image = volume.image.detach().cpu()
    if image.ndim != 3 or not torch.isfinite(image).all():
        raise ValueError(f'GT must be finite and 3D: {path}')
    brain = image > 0
    if not brain.any():
        raise ValueError(f'Empty GT-positive mask: {path}')
    volume.mask = brain.to(volume.mask.device)
    xyz = volume.xyz_masked.detach().cpu().float().contiguous()
    flat_brain = np.flatnonzero(brain.numpy().ravel())
    if len(xyz) != len(flat_brain):
        raise ValueError('Native xyz_masked does not match GT-positive mask')
    chosen = sample_indices(len(xyz), a.points, a.seed, patient, modality)
    points = xyz[torch.as_tensor(chosen, dtype=torch.long)].numpy()
    # A native array-axis-0 central plane; not asserted to be anatomical axial.
    z = image.shape[0]//2
    plane_pick = np.flatnonzero(flat_brain//(image.shape[1]*image.shape[2]) == z)
    plane_xyz = xyz[torch.as_tensor(plane_pick, dtype=torch.long)].numpy()
    plane_flat = flat_brain[plane_pick] % (image.shape[1]*image.shape[2])
    return points, plane_xyz, plane_flat, tuple(image.shape), dict(
        gt_sha256=sha_file(path), n_gt_brain=len(xyz), n_points=len(points),
        coordinates_sha256=array_hash(points), sample_indices_sha256=array_hash(chosen),
        gt_brain_bounding_box=[xyz.amin(dim=0).tolist(),xyz.amax(dim=0).tolist()],
        coordinate_api='nesvor.image.load_volume / GT-positive mask / xyz_masked',
        native_shape=list(image.shape), plane_axis=0, plane_index=z)

def load_checkpoint(path, torch, INR, a):
    # Only trusted checkpoints produced by the user's own experiment are intended.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or not {'args','model'} <= set(ckpt):
        raise ValueError('Expected checkpoint dictionary with args and model')
    args = copy.deepcopy(ckpt['args'])
    sd = ckpt['model']
    if not isinstance(sd, dict) or 'bounding_box' not in sd:
        raise ValueError('Expected exported INR state_dict with bounding_box (no automatic key stripping)')
    bb = sd['bounding_box'].detach().to(device=a.device, dtype=torch.float32).clone()
    if bb.shape != (2,3) or not torch.isfinite(bb).all() or not torch.all(bb[1] > bb[0]):
        raise ValueError('Invalid saved bounding_box')
    if not hasattr(args, 'dtype'):
        raise ValueError('Checkpoint args.dtype missing; do not guess architecture/dtype')
    if a.constructor == 'training':
        # Reproduce train.py's arithmetic, including ceil-sensitive grid construction.
        center = (bb[0]+bb[1])/2
        build_bb = (bb-center)/30.0
        scale = 30.0
    else:
        build_bb, scale = bb.clone(), 1.0
    from nesvor.inr.models import compute_resolution_nlevel
    base, levels = compute_resolution_nlevel(build_bb, args.coarsest_resolution,
                                            args.finest_resolution, args.level_scale, scale)
    model = INR(build_bb.clone(), args, spatial_scaling=scale).to(a.device)
    expected = model.state_dict()
    missing = sorted(set(expected)-set(sd))
    unexpected = sorted(set(sd)-set(expected))
    mismatch = {k: dict(expected=list(expected[k].shape), saved=list(sd[k].shape))
                for k in set(sd)&set(expected) if expected[k].shape != sd[k].shape}
    if missing or unexpected or mismatch:
        raise ValueError(f'STRICT LOAD FAILED: missing={missing}; unexpected={unexpected}; shapes={mismatch}. '
                         'No parameters were removed. Check model source/backend/constructor; '
                         '--constructor physical is an explicit alternative for exported inference construction.')
    model.load_state_dict(sd, strict=True)
    model.eval()
    if not (model.use_gating and model.spatial_gating and model.z_gating):
        raise ValueError('This protocol expects spatial + content-conditioned gating enabled')
    if not torch.equal(model.bounding_box, bb):
        raise ValueError('Physical bounding_box was not restored')
    gate = model.gating_net
    config = dict(n_levels=int(model.n_levels), n_features_per_level=int(model.n_features_per_level),
        n_features_z=int(model.n_features_z), base_resolution=base,
        coarsest_resolution=float(args.coarsest_resolution), finest_resolution=float(args.finest_resolution),
        level_scale=float(args.level_scale), width=int(args.width), depth=int(args.depth),
        dtype=str(args.dtype), gating_mode=str(model.gating_mode),
        gating_hidden_dim=int(gate.net[0].out_features), gating_activation=type(gate.net[1]).__name__,
        temperature=float(gate.temperature), min_weight=float(gate.min_weight),
        weight_ff_loss=float(getattr(args,'weight_ff_loss',0.)),
        ff_alpha=float(getattr(args,'ff_alpha',1.)), training_seed=str(getattr(args,'seed','not_saved')),
        v_mean=float(model.v_mean.item()), bounding_box=bb.detach().cpu().tolist(),
        physical_max_span=float((bb[1]-bb[0]).max().item()),
        constructor=a.constructor, n_state_keys=len(sd), strict_load=True)
    del ckpt, sd, expected
    return model, config

def query(model, points, torch, a):
    if not len(points):
        return np.empty((0, model.n_levels), np.float64)
    bb = model.bounding_box.detach().cpu().numpy()
    # No out-of-bounds clipping or implicit common-intersection cropping.
    outside = ((points < bb[0]-a.bounds_tolerance_mm) | (points > bb[1]+a.bounds_tolerance_mm)).any(1)
    if outside.any():
        raise ValueError(f'{outside.sum()}/{len(points)} GT query points outside model bounding_box. '
                         'No extrapolation/cropping was performed; inspect bounding_box and native coordinates.')
    output = []
    with torch.no_grad():
        for i in range(0, len(points), a.batch_size):
            xyz = torch.as_tensor(points[i:i+a.batch_size], dtype=torch.float32, device=a.device)
            # Exactly the point-query form of sample_points(resolution=0); no PSF random samples.
            xyz = model.sample_batch(xyz, None, 0.0, 0)
            _, w = model(xyz)
            if w is None:
                raise ValueError('Missing weights; no uniform-weight substitution is allowed')
            output.append(w.detach().float().cpu().numpy())
    w = np.concatenate(output)
    validate_weights(w, model.n_levels, len(points), model.gating_net.min_weight)
    return w
