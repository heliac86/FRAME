"""FRAME R1-6: shared discovery, paired inference and reproducibility utilities."""
from __future__ import annotations
import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = '1.2.0'
MODALITIES = ('flair', 't1ce', 't1', 't2')
ID_RE = re.compile(r'brats20_training_(\d+)', re.I)
MOD_RE = re.compile(r'(?:^|[^a-z0-9])(flair|t1ce|t1|t2)(?=$|[^a-z0-9])', re.I)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def patients(path, expected=48):
    with Path(path).open(encoding='utf-8-sig') as f:
        header=f.readline()
    try:
        delimiter=csv.Sniffer().sniff(header,delimiters=',;\t').delimiter
    except csv.Error:
        delimiter=','  # Single-column Brats20ID files are also valid.
    d = pd.read_csv(path, sep=delimiter, dtype=str,
                    keep_default_na=False, encoding='utf-8-sig')
    d.columns = d.columns.str.strip()
    if list(d.columns).count('Brats20ID') != 1:
        raise ValueError('test CSV/TSV needs exactly one Brats20ID column')
    keys = []
    for value in d.Brats20ID:
        m = ID_RE.fullmatch(value.strip())
        if not m:
            raise ValueError(f'Invalid patient ID: {value!r}')
        keys.append(str(int(m.group(1))).zfill(3))
    if len(keys) != expected or len(set(keys)) != len(keys):
        raise ValueError(f'Need {expected} unique patients; found {len(keys)} rows')
    return keys

def index_files(root, keys, extensions, pattern='*', modalities=MODALITIES):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Directory not found: {root}')
    result = {(k, m): [] for k in keys for m in modalities}
    def fail(e):
        raise e
    for directory, _, names in os.walk(root, onerror=fail):
        for name in sorted(names):
            if not name.lower().endswith(extensions) or not fnmatch.fnmatchcase(name.lower(), pattern.lower()):
                continue
            mods = set(x.lower() for x in MOD_RE.findall(name))
            if len(mods) != 1:
                continue
            if next(iter(mods)) not in modalities:
                continue
            p = Path(directory)/name
            ids = {str(int(x)).zfill(3) for x in ID_RE.findall(name)}
            lead = re.match(r'^(\d+)(?=[_.-])', name)
            if lead:
                ids.add(str(int(lead.group(1))).zfill(3))
            for part in p.relative_to(root).parts[:-1]:
                match = ID_RE.fullmatch(part)
                if match:
                    ids.add(str(int(match.group(1))).zfill(3))
                elif re.fullmatch(r'\d{1,3}', part):
                    ids.add(str(int(part)).zfill(3))
            if ids & set(keys) and len(ids) != 1:
                raise ValueError(f'Conflicting patient IDs: {p}')
            if len(ids) == 1 and next(iter(ids)) in keys:
                result[next(iter(ids)), next(iter(mods))].append(str(p.resolve()))
    return result

def discover(args, keys, out):
    modalities=tuple(args.modalities)
    indexes = {
        'gt': index_files(args.gt_root, keys, ('.nii', '.nii.gz'), args.gt_pattern,modalities),
        'nifti': index_files(args.recon_root, keys, ('.nii', '.nii.gz'), args.nii_pattern,modalities),
        'checkpoint': index_files(args.ckpt_root or args.recon_root, keys, ('.pt',), args.ckpt_pattern,modalities)}
    rows, issues = [], []
    for k in keys:
        for mod in modalities:
            row = {'patient': k, 'modality': mod, 'factor': args.factor}
            for role, ix in indexes.items():
                hits = ix[k, mod]
                if len(hits) != 1:
                    issues.append(dict(patient=k, modality=mod, role=role, candidates=hits))
                else:
                    row[role] = hits[0]
            rows.append(row)
    write_json(out/'discovery_issues.json', issues)
    pd.DataFrame(rows).to_csv(out/'file_manifest.csv', index=False)
    if issues:
        raise ValueError(f'{len(issues)} missing/ambiguous entries; see discovery_issues.json. '
                         'Use --ckpt-pattern / --nii-pattern / --gt-pattern to select one run.')
    for role in indexes:
        if len({r[role] for r in rows}) != len(rows):
            raise ValueError(f'A {role} file was selected for multiple entries')
    return rows

def sample_indices(n, count, seed, patient, modality):
    if n <= 0 or count <= 0:
        raise ValueError('Brain mask and --points must be positive')
    # Stable across factors, traversal order, devices and Python hash seeds.
    salt = hashlib.sha256(f'{seed}/{patient}/{modality}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(salt[:8], 'little'))
    return np.sort(rng.choice(n, min(n, count), replace=False))

def validate_weights(w, nlevels, npoints, min_weight=0.):
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (npoints, nlevels) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError(f'Invalid spatial weights shape/values: {w.shape}, expected {(npoints,nlevels)}')
    total = w.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError('Nonpositive weight sum')
    if min_weight == 0 and not np.allclose(total, nlevels, rtol=2e-5, atol=2e-5):
        raise ValueError('Unclamped softmax weights do not sum to n_levels')
    return w, w / total[:, None]

def holm(p):
    p = np.asarray(p, float)
    if not np.isfinite(p).all():
        raise ValueError('Holm inputs must be finite')
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    out = np.empty_like(p)
    out[order] = adjusted
    return out

def paired(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite(a-b).all():
        raise ValueError('Paired analysis needs complete finite patient pairs')
    delta = a-b
    mean, sd, n = float(delta.mean()), float(delta.std(ddof=1)), len(delta)
    tolerance = 16*np.finfo(float).eps*max(1., float(np.abs(a).max()), float(np.abs(b).max()))
    if float(np.ptp(delta)) <= tolerance:
        p = 1. if abs(mean) <= tolerance else None
        low = high = mean
        flag = 'all_zero' if abs(mean) <= tolerance else 'constant_nonzero_difference_no_t_inference'
    else:
        se = sd/np.sqrt(n)
        margin = float(stats.t.ppf(.975, n-1)*se)
        low, high = mean-margin, mean+margin
        p = float(2*stats.t.sf(abs(mean/se), n-1))
        flag = 'ok'
    return dict(n_patients=n, delta=mean, ci95_low=low, ci95_high=high,
                p_raw=p, inference_flag=flag, n_increased=int((delta>0).sum()),
                n_decreased=int((delta<0).sum()), n_equal=int((delta==0).sum()))

def patient_means(raw,modalities=MODALITIES):
    required = ['patient','modality','factor','level','weight_mean','share_mean']
    if any(x not in raw for x in required):
        raise ValueError('Invalid raw CSV schema')
    if raw.duplicated(['patient','modality','factor','level']).any():
        raise ValueError('Duplicate patient/modality/factor/level row')
    if not np.isfinite(raw[['weight_mean','share_mean']].to_numpy(float)).all():
        raise ValueError('Nonfinite weights')
    if set(raw.modality) != set(modalities):
        raise ValueError('Expected exactly these modalities: '+str(tuple(modalities)))
    counts = raw.groupby(['patient','factor','level']).modality.nunique()
    if not counts.eq(len(modalities)).all():
        raise ValueError('Incomplete modality set')
    return raw.groupby(['patient','factor','level'], as_index=False)[['weight_mean','share_mean']].mean()

def prepare_out(path):
    out = Path(path).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output is not empty: {out}. Use a new --out directory.')
    out.mkdir(parents=True, exist_ok=True)
    return out
