#!/usr/bin/env python3
"""Extract four-modal or selected-modal 4x weights in the NeSVoR CUDA environment."""
from __future__ import annotations
import argparse
import datetime as dt
import inspect
from pathlib import Path
import platform
import re
import sys
import traceback
import numpy as np
import pandas as pd
from gating_common import (VERSION, MODALITIES, patients, discover,
    prepare_out, write_json, sha_file, validate_weights, patient_means)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factor', type=int, choices=[4], default=4)
    p.add_argument('--modalities',nargs='+',choices=MODALITIES,default=list(MODALITIES),
                   help='Defaults to all four; use t1 t2 to reuse existing FLAIR/T1ce results')
    p.add_argument('--recon-root', required=True, help='Directory containing reconstructed MRI volumes')
    p.add_argument('--ckpt-root', help='Separate checkpoint directory; defaults to recon-root')
    p.add_argument('--gt-root', required=True, help='Directory containing original reference MRI volumes')
    p.add_argument('--test-csv', required=True, help='CSV listing the patients to analyze')
    p.add_argument('--out', required=True)
    p.add_argument('--expected-patients', type=int, default=48)
    p.add_argument('--ckpt-pattern', default='*')
    p.add_argument('--nii-pattern', default='*')
    p.add_argument('--gt-pattern', default='*')
    p.add_argument('--points', type=int, default=50000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--nesvor-root', help='Source checkout directory CONTAINING nesvor/; no editable install needed')
    p.add_argument('--constructor', choices=['training','physical'], default='training',
                   help='training reproduces centering/scaling arithmetic in supplied train.py')
    p.add_argument('--bounds-tolerance-mm', type=float, default=1e-4)
    p.add_argument('--examples', type=int, default=3, help='First patients in test CSV, not best-performing cases')
    p.add_argument('--save-point-weights', action='store_true')
    p.add_argument('--allow-model-source-difference', action='store_true')
    p.add_argument('--discovery-only', action='store_true', help='File checks only; no checkpoint validation')
    p.add_argument('--dry-run', action='store_true', help='Strict-load ALL checkpoints and query 64 fixed GT points each')
    return p

def condition_summary(raw, out, factor, modalities):
    pm = patient_means(raw,modalities)
    pm.to_csv(out/'patient_means.csv', index=False)
    summaries = []
    scopes = [('_'.join(modalities)+'_patient_mean', pm)] + [(m,raw[raw.modality.eq(m)]) for m in modalities]
    for scope, d in scopes:
        for lv, sub in d.groupby('level'):
            for measure in ['weight_mean','share_mean']:
                v = sub[measure].to_numpy(float)
                summaries.append(dict(scope=scope, level=int(lv), measure=measure,
                    n_patients=len(v), mean=v.mean(), std=v.std(ddof=1), median=np.median(v)))
    pd.DataFrame(summaries).to_csv(out/'level_summary.csv', index=False)
    return pm

def save_spatial_data(weights, indices, shape, out, name, nlevels):
    # Data only; render_gating.py produces PNGs in the analysis environment.
    out.mkdir(exist_ok=True)
    maps = np.full((shape[1]*shape[2],nlevels), np.nan)
    maps[indices] = weights
    maps = maps.reshape(shape[1],shape[2],nlevels)
    np.savez_compressed(out/(name+'.npz'), weights=maps, native_plane_axis=0,
                        native_plane_index=shape[0]//2)

def run(a,out,manifest):
    if len(set(a.modalities))!=len(a.modalities):
        raise ValueError('Duplicate --modalities values')
    manifest['modalities']=list(a.modalities)
    manifest['sampling_protocol']='native_gt_positive_point_query_v1'
    if a.points<1 or a.batch_size<1 or a.examples<0 or a.bounds_tolerance_mm<0:
        raise ValueError('Invalid points/batch-size/examples/tolerance')
    keys=patients(a.test_csv,a.expected_patients)
    manifest.update(patient_ids=keys, test_csv_sha256=sha_file(a.test_csv))
    rows=discover(a,keys,out)
    for r in rows:
        tokens=re.findall(r'(?:^|_)([246])x(?:5)?(?=_|\.)',Path(r['nifti']).name)
        if tokens and set(map(int,tokens))!={a.factor}:
            raise ValueError(f'NIfTI filename factor conflicts with --factor: {r["nifti"]}')
    if a.discovery_only:
        manifest['status']='DISCOVERY_ONLY'
        return
    from gating_backend import initialize, reference_points, load_checkpoint, query
    torch,INR,load_volume,runtime=initialize(a.device,a.nesvor_root)
    manifest['runtime']=runtime
    reference=Path(__file__).parent/'reference_sources'/'models.py'
    if reference.exists():
        runtime['reference_models_sha256']=sha_file(reference)
        runtime['matches_supplied_models']=runtime['models_sha256']==runtime['reference_models_sha256']
        if not runtime['matches_supplied_models'] and not a.allow_model_source_difference:
            raise ValueError(f'Imported models.py differs from the supplied reference: {runtime["models_path"]}. '
                             'Check PYTHONPATH. --allow-model-source-difference records an explicit reviewed exception.')
    raw_rows=[];audit=[];common_nlevels=None
    audit_path=out/'checkpoint_audit.json'
    for i,r in enumerate(rows,1):
        manifest['current_entry']=r
        print(f'[{i}/{len(rows)}] {r["patient"]} {r["modality"]}',flush=True)
        points,plane_xyz,plane_indices,shape,roi=reference_points(
            r['gt'],load_volume,torch,a,r['patient'],r['modality'])
        model,config=load_checkpoint(r['checkpoint'],torch,INR,a)
        nlevels=config['n_levels']
        if common_nlevels is None:
            common_nlevels=nlevels
        if nlevels != common_nlevels:
            raise ValueError('n_levels differs within condition; pooled level profiles require matching level sets')
        # Verify containment of the entire GT brain, not just its random sample.
        bb=np.array(config['bounding_box'])
        rb=np.array(roi['gt_brain_bounding_box'])
        if np.any(rb[0]<bb[0]-a.bounds_tolerance_mm) or np.any(rb[1]>bb[1]+a.bounds_tolerance_mm):
            raise ValueError(f'Full GT brain is not contained in checkpoint bounding_box: {r["checkpoint"]}. '
                             f'GT bounds={rb.tolist()}, model bounds={bb.tolist()}. No implicit ROI truncation.')
        evaluated=points[:64] if a.dry_run else points
        w=query(model,evaluated,torch,a)
        w,q=validate_weights(w,nlevels,len(evaluated),config['min_weight'])
        entry=dict(patient=r['patient'],modality=r['modality'],factor=a.factor,
            checkpoint=r['checkpoint'],checkpoint_sha256=sha_file(r['checkpoint']),
            nifti=r['nifti'],nifti_sha256=sha_file(r['nifti']),roi=roi,model=config,
            weight_sum_min=float(w.sum(1).min()),weight_sum_max=float(w.sum(1).max()),
            n_evaluated=len(evaluated))
        audit.append(entry);write_json(audit_path,audit)
        if not a.dry_run:
            maxima=np.isclose(w,w.max(axis=1,keepdims=True),rtol=1e-6,atol=1e-8)
            top_fraction=(maxima/maxima.sum(axis=1,keepdims=True)).mean(axis=0)
            for j in range(nlevels):
                raw_rows.append(dict(patient=r['patient'],modality=r['modality'],factor=a.factor,
                    level=j+1,n_points=len(w),weight_mean=float(w[:,j].mean()),
                    weight_spatial_std=float(w[:,j].std(ddof=1)) if len(w)>1 else 0.,
                    share_mean=float(q[:,j].mean()),
                    top_level_fraction=float(top_fraction[j])))
            if a.save_point_weights:
                folder=out/'point_weights';folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{r["patient"]}_{r["modality"]}.npz',
                                    coordinates=points,weights=w.astype(np.float32))
            if r['patient'] in keys[:a.examples] and len(plane_xyz):
                pw=query(model,plane_xyz,torch,a)
                save_spatial_data(pw,plane_indices,shape,out/'spatial',
                             f'{r["patient"]}_{r["modality"]}_{a.factor}x',nlevels)
        del model
        torch.cuda.empty_cache()
        if raw_rows:
            pd.DataFrame(raw_rows).to_csv(out/'partial_gating_weights_raw.csv',index=False)
    manifest['n_levels']=common_nlevels
    manifest.pop('current_entry',None)
    manifest['n_checkpoints']=len(audit)
    manifest['checkpoint_audit_sha256']=sha_file(audit_path)
    if a.dry_run:
        manifest['status']='PREFLIGHT_OK'
        return
    raw=pd.DataFrame(raw_rows)
    raw.to_csv(out/'gating_weights_raw.csv',index=False)
    manifest['raw_csv_sha256']=sha_file(out/'gating_weights_raw.csv')
    condition_summary(raw,out,a.factor,tuple(a.modalities))
    report=f'''# FRAME gating analysis: {a.factor}x

COMPLETE. Patients={len(keys)}; modalities={','.join(a.modalities)}; checkpoints={len(audit)}; levels={common_nlevels}.
Each model was restored strictly; no missing parameters or artificial uniform weights.
GT-positive brain coordinates use the native NeSVoR image API, sampled deterministically.
All GT-brain points were required to lie inside each model bounding box.
Point queries use resolution=0 (no PSF averaging), not training-time PSF-group weights.
The primary comparison variable is normalized weight share (pointwise w/sum(w), then spatial mean).
Raw weights are also retained. The combined profile averages modalities within patients first.
Extraction saves CSV/JSON/NPZ only; it does not import matplotlib or SciPy.
Render/analyze in the analysis environment with analyze_4x_modalities.py.
Spatial examples are the first {a.examples} CSV patients; raw maps are saved as NPZ.
No hypothesis test is performed here. This output is for the four-modal 4x analysis.
These are feature-weight patterns, not disjoint frequency bands or evidence of frequency recovery.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    manifest['status']='COMPLETE'

def main():
    a=parser().parse_args()
    if a.discovery_only and a.dry_run:
        raise SystemExit('Choose either --discovery-only or --dry-run')
    out=prepare_out(a.out)
    manifest=dict(protocol_version=VERSION,analysis_role='exploratory R1-6 condition-dependent weight patterns',
        status='RUNNING',started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        configuration=vars(a),python=platform.python_version(),
        code_sha256={n:sha_file(Path(__file__).parent/n) for n in
                     ['extract_4x_gating.py','gating_common.py','gating_backend.py']})
    try:
        run(a,out,manifest)
    except Exception:
        manifest['status']='FAILED'
        (out/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(traceback.format_exc(),file=sys.stderr)
    finally:
        manifest['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(out/'run_manifest.json',manifest)
    print(f'[{manifest["status"]}] {out}',flush=True)
    return 1 if manifest['status']=='FAILED' else 0

if __name__=='__main__':
    raise SystemExit(main())
