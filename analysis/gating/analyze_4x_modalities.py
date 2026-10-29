#!/usr/bin/env python3
"""Analyze four-modal 4x weights; analysis environment only, no NeSVoR/Torch."""
from __future__ import annotations
import argparse
import datetime as dt
from itertools import combinations
from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd
from scipy import stats
from gating_common import (VERSION,MODALITIES,read_json,write_json,sha_file,
                           patient_means,paired,holm,prepare_out)

MODEL_KEYS=('n_levels','n_features_per_level','n_features_z','coarsest_resolution',
    'finest_resolution','level_scale','width','depth','dtype','gating_mode',
    'gating_hidden_dim','gating_activation','temperature','min_weight',
    'weight_ff_loss','ff_alpha','constructor')

def read_extraction(folder):
    folder=Path(folder).resolve();m=read_json(folder/'run_manifest.json')
    if m.get('status')!='COMPLETE' or m.get('protocol_version') not in ('1.1.0',VERSION):
        raise ValueError(f'Need COMPLETE v1.1.0/v{VERSION} extraction: {folder}')
    if m['configuration']['factor']!=4:
        raise ValueError('This analysis accepts 4x ONLY')
    for name,key in [('gating_weights_raw.csv','raw_csv_sha256'),
                     ('checkpoint_audit.json','checkpoint_audit_sha256')]:
        if sha_file(folder/name)!=m[key]:
            raise ValueError(f'Extraction output changed after completion: {folder/name}')
    d=pd.read_csv(folder/'gating_weights_raw.csv',dtype={'patient':str})
    audit=read_json(folder/'checkpoint_audit.json')
    required={'patient','modality','factor','level','weight_mean','share_mean'}
    if not required<=set(d.columns):raise ValueError('Raw CSV schema mismatch')
    mods=set(d.modality);keys=m['patient_ids'];levels=range(1,m['n_levels']+1)
    if len(set(keys))!=len(keys) or len(keys)<2:raise ValueError('Need unique patient IDs, at least two')
    if not mods or not mods<=set(MODALITIES):raise ValueError('Unsupported modality set')
    expected={(p,mod,4,lv) for p in keys for mod in mods for lv in levels}
    actual=list(d[['patient','modality','factor','level']].itertuples(index=False,name=None))
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Incomplete/duplicate patient/modality/level rows')
    if not np.isfinite(d[['weight_mean','share_mean']].to_numpy()).all():
        raise ValueError('Nonfinite weights')
    if (d[['weight_mean','share_mean']]<0).any().any() or not np.allclose(
        d.groupby(['patient','modality']).share_mean.sum(),1.,rtol=2e-5,atol=2e-5):
        raise ValueError('Invalid normalized shares')
    expected_audit={(p,mod) for p in keys for mod in mods}
    actual_audit=[(r['patient'],r['modality']) for r in audit]
    if len(actual_audit)!=len(expected_audit) or set(actual_audit)!=expected_audit:
        raise ValueError('Incomplete/duplicate checkpoint audit')
    for r in audit:
        if r['factor']!=4 or not r['model']['strict_load'] or r['model']['n_levels']!=m['n_levels']:
            raise ValueError('Invalid model audit')
        if r['n_evaluated']!=r['roi']['n_points']:
            raise ValueError('Partial point evaluation cannot be analyzed as COMPLETE')
    return dict(folder=folder,manifest=m,raw=d,audit=audit)

def merge_inputs(inputs):
    bundles=[read_extraction(p) for p in inputs]
    first=bundles[0]['manifest'];keys=first['patient_ids']
    first_model=bundles[0]['audit'][0]['model']
    all_pairs=[];all_hashes=[];geometry=[]
    for b in bundles:
        m=b['manifest']
        if set(m['patient_ids'])!=set(keys) or m['test_csv_sha256']!=first['test_csv_sha256']:
            raise ValueError('Patient list/test CSV differs between inputs')
        if m['n_levels']!=first['n_levels']:raise ValueError('n_levels differs')
        for field in ('points','seed','bounds_tolerance_mm','constructor'):
            if m['configuration'][field]!=first['configuration'][field]:
                raise ValueError(f'Extraction protocol differs: {field}')
        if m['runtime']['models_sha256']!=first['runtime']['models_sha256']:
            raise ValueError('Imported NeSVoR model source differs')
        if m['code_sha256']['gating_backend.py']!=first['code_sha256']['gating_backend.py']:
            raise ValueError('Checkpoint/coordinate-query backend differs; do not merge silently')
        if m.get('sampling_protocol','native_gt_positive_point_query_v1')!='native_gt_positive_point_query_v1':
            raise ValueError('Unsupported sampling protocol')
        for r in b['audit']:
            pair=(r['patient'],r['modality']);all_pairs.append(pair)
            all_hashes.append(r['checkpoint_sha256'])
            model=r['model']
            for field in MODEL_KEYS:
                if model[field]!=first_model[field]:
                    raise ValueError(f'Model setting differs: {pair} {field}')
            geometry.append(dict(patient=pair[0],modality=pair[1],n_levels=model['n_levels'],
                base_resolution=model['base_resolution'],physical_max_span=model['physical_max_span'],
                nominal_coarsest_grid_mm=model['physical_max_span']/model['base_resolution'],
                n_gt_brain=r['roi']['n_gt_brain'],n_points=r['roi']['n_points'],
                training_seed=model['training_seed']))
    if len(set(all_pairs))!=len(all_pairs):
        raise ValueError('Overlapping modalities/patients across input folders; select nonoverlapping extractions')
    if len(set(all_hashes))!=len(all_hashes):
        raise ValueError('Same checkpoint was reused for multiple patient/modality entries')
    expected={(p,mod) for p in keys for mod in MODALITIES}
    if set(all_pairs)!=expected:
        raise ValueError(f'Four complete modalities required. Missing: {sorted(expected-set(all_pairs))}')
    raw=pd.concat([b['raw'] for b in bundles],ignore_index=True)
    return bundles,raw,keys,pd.DataFrame(geometry)

def summarize(raw):
    four=patient_means(raw,MODALITIES)
    two=patient_means(raw[raw.modality.isin(['flair','t1ce'])],('flair','t1ce'))
    scopes=[(mod,raw[raw.modality.eq(mod)]) for mod in MODALITIES]+[
        ('four_modality_patient_mean',four),('flair_t1ce_patient_mean',two)]
    rows=[]
    for scope,d in scopes:
        for lv,sub in d.groupby('level'):
            for variable in ('weight_mean','share_mean'):
                v=sub[variable].to_numpy(float);n=len(v);mean=v.mean();sd=v.std(ddof=1)
                margin=stats.t.ppf(.975,n-1)*sd/np.sqrt(n)
                rows.append(dict(scope=scope,level=int(lv),measure=variable,n_patients=n,
                    mean=mean,std=sd,median=np.median(v),cv=sd/mean if mean else np.nan,
                    ci95_low=mean-margin,ci95_high=mean+margin))
    return pd.DataFrame(rows),four,two

def modality_comparisons(raw,keys,levels):
    rows=[]
    for a,b in combinations(MODALITIES,2):
        for lv in levels:
            d=raw[raw.level.eq(lv)].pivot(index='patient',columns='modality',values='share_mean').reindex(keys)
            result=paired(100*d[a].to_numpy(),100*d[b].to_numpy())
            rows.append(dict(contrast=f'{a}_minus_{b}',modality_a=a,modality_b=b,level=int(lv),
                mean_a_pct=100*d[a].mean(),mean_b_pct=100*d[b].mean(),**result))
    results=pd.DataFrame(rows)
    eligible=results.p_raw.notna().to_numpy()
    corrected=holm(results.p_raw.fillna(1.).to_numpy())
    results['p_holm']=np.where(eligible,corrected,np.nan)
    results['significant_holm_05']=np.where(eligible & (corrected<.05),'YES','NO')
    results['holm_family_size']=len(results)
    return results

def figures(raw,summary,keys,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'flair':'#4c72b0','t1ce':'#dd8452','t1':'#55a868','t2':'#c44e52'}
    levels=sorted(raw.level.unique())
    fig,ax=plt.subplots(figsize=(11,4.5))
    for mod in MODALITIES:
        d=summary[(summary.scope==mod)&(summary.measure=='share_mean')]
        ax.plot(d.level,100*d['mean'],color=colors[mod],marker='o',label=mod)
        ax.fill_between(d.level,100*d.ci95_low,100*d.ci95_high,color=colors[mod],alpha=.12)
    ax.axhline(100/len(levels),color='gray',ls='--',label='Uniform share')
    ax.set(xlabel='Hash-grid level (coarse to fine resolution)',ylabel='Mean normalized weight share (%)',
           title='FRAME 4x: four modalities; patient means and pointwise 95% t CIs')
    ax.set_xticks(levels);ax.legend(ncol=5,fontsize=9)
    fig.tight_layout();fig.savefig(out/'four_modality_profiles.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(15,5))
    for j,mod in enumerate(MODALITIES):
        data=[100*raw[(raw.modality==mod)&(raw.level==lv)].share_mean.to_numpy() for lv in levels]
        bp=ax.boxplot(data,positions=np.arange(len(levels))+(j-1.5)*.18,widths=.16,patch_artist=True,
            manage_ticks=False,showfliers=True,medianprops={'color':'black','linewidth':1})
        for box in bp['boxes']:box.set(facecolor=colors[mod],alpha=.65)
        ax.plot([],[],color=colors[mod],lw=7,label=mod)
    ax.set_xticks(np.arange(len(levels)));ax.set_xticklabels([f'L{x}' for x in levels])
    ax.set(ylabel='Normalized weight share (%)',title='Within-modality distributions across patients')
    ax.legend();fig.tight_layout();fig.savefig(out/'four_modality_boxplots.png',dpi=160);plt.close(fig)
    matrix=[]
    for mod in MODALITIES:
        matrix.append(raw[raw.modality.eq(mod)].pivot(index='patient',columns='level',values='share_mean').reindex(keys)[levels].to_numpy()*100)
    matrix=np.vstack(matrix)
    fig,ax=plt.subplots(figsize=(12,10))
    im=ax.imshow(matrix,aspect='auto',cmap='viridis',vmin=0,vmax=float(matrix.max()))
    for j,mod in enumerate(MODALITIES):
        if j:ax.axhline(j*len(keys)-.5,color='white',lw=2)
    ax.set_yticks([(j+.5)*len(keys)-.5 for j in range(4)]);ax.set_yticklabels(MODALITIES)
    ax.set_xticks(np.arange(len(levels)));ax.set_xticklabels([f'L{x}' for x in levels])
    ax.set(xlabel='Hash-grid level',title='FRAME 4x: all patients, grouped by modality')
    fig.colorbar(im,ax=ax,label='Normalized weight share (%)')
    fig.tight_layout();fig.savefig(out/'four_modality_heatmap.png',dpi=150);plt.close(fig)

def render_spatial(bundles,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outputs=set()
    for bundle in bundles:
        for path in sorted((bundle['folder']/'spatial').glob('*.npz')):
            if path.stem in outputs:raise ValueError('Duplicate spatial map filename across inputs')
            outputs.add(path.stem)
            with np.load(path,allow_pickle=False) as f:maps=f['weights']
            if maps.ndim!=3:raise ValueError('Invalid spatial map shape')
            n=maps.shape[-1];cols=4;rows=int(np.ceil(n/cols))
            fig,axes=plt.subplots(rows,cols,figsize=(12,rows*3),squeeze=False,constrained_layout=True)
            im=None
            for i,ax in enumerate(axes.ravel()):
                if i>=n:ax.axis('off');continue
                im=ax.imshow(maps[...,i],origin='lower',cmap='magma',vmin=0,vmax=n)
                ax.set_title(f'L{i+1}');ax.axis('off')
            fig.colorbar(im,ax=axes.ravel().tolist(),label='Raw gating weight (fixed 0..n_levels)')
            fig.suptitle(path.stem+' | GT native array-axis-0 mid-plane')
            folder=out/'spatial';folder.mkdir(exist_ok=True)
            fig.savefig(folder/(path.stem+'.png'),dpi=120);plt.close(fig)

def run(a,out,manifest):
    bundles,raw,keys,geometry=merge_inputs(a.inputs)
    if len(keys)<2:raise ValueError('Analysis requires at least two patients')
    nlevels=bundles[0]['manifest']['n_levels'];levels=range(1,nlevels+1)
    summary,four,two=summarize(raw)
    tests=modality_comparisons(raw,keys,levels)
    raw.to_csv(out/'combined_volume_weights_4mod.csv',index=False)
    summary.to_csv(out/'modality_level_summary.csv',index=False)
    four.to_csv(out/'patient_means_4mod.csv',index=False)
    two.to_csv(out/'patient_means_flair_t1ce.csv',index=False)
    geometry.to_csv(out/'grid_geometry_audit.csv',index=False)
    tests.to_csv(out/'paired_modality_comparisons.csv',index=False)
    figures(raw,summary,keys,out)
    render_spatial(bundles,out)
    manifest.update(n_patients=len(keys),patient_ids=keys,n_levels=nlevels,modalities=list(MODALITIES),
        n_checkpoints=len(geometry),inputs=[dict(folder=str(b['folder']),
        protocol_version=b['manifest']['protocol_version'],
        run_manifest_sha256=sha_file(b['folder']/'run_manifest.json')) for b in bundles])
    lines=['# FRAME 4x: four-modality gating analysis','',
        f'Patients: {len(keys)}; modalities: flair, t1ce, t1, t2; checkpoints: {len(geometry)}; levels: {nlevels}.',
        'Complete, nonoverlapping input extractions and matching sampling/model settings were required.',
        'Legacy v1.1.0 FLAIR/T1ce outputs can be reused without re-querying their checkpoints.',
        'GT-positive masks and sampled points are modality-specific, consistent with the original extraction protocol.',
        'Modality differences describe trained models and their evaluation domains, not a causal effect of contrast alone.',
        '', '## Planned modality comparisons',
        f'Six modality pairs × {nlevels} levels = {6*nlevels} two-sided paired t-tests in ONE Holm family.',
        'Unit: percentage points of normalized weight share. Positive means A has more share, not better reconstruction.',
        'Each comparison uses one value per patient per modality; four modalities are repeated observations.',
        '95% difference CIs are pointwise, not multiplicity-adjusted. SD/CV describe across-patient variability.',
        'Nonzero constant differences have no t inference and retain a reserved hypothesis in the Holm family.',
        'The family is separate from prior cross-degradation analyses and does not imply study-wide error control.',
        '', '## Mean normalized weight shares (%)','',
        '|Level|FLAIR|T1ce|T1|T2|','|---|---:|---:|---:|---:|']
    means=summary[summary.measure.eq('share_mean')].pivot(index='level',columns='scope',values='mean')
    for lv in levels:lines.append('|L'+str(lv)+'|'+ '|'.join(f'{100*means.loc[lv,mod]:.4f}' for mod in MODALITIES)+'|')
    lines+=['','## Modality-pair summary after the full Holm correction','',
            '|Contrast|Holm-significant levels|','|---|---|']
    for contrast,sub in tests.groupby('contrast',sort=False):
        selected=sub.loc[sub.significant_holm_05.eq('YES'),'level'].astype(int).tolist()
        labels=', '.join('L'+str(lv) for lv in selected) if selected else 'none'
        lines.append('|'+contrast+'|'+labels+'|')
    lines+=['','This list describes weight differences, not an aggregate performance score.',
        '', '## Output interpretation',
        'The four-modality patient mean is distinct from the historical FLAIR/T1ce patient mean.',
        'Use only FLAIR/T1ce for 2x/4x/6x comparisons; do not replace their 4x mean with a four-modality mean.',
        'Level IDs are not disjoint frequency bands. Weight magnitude is not causal importance or reconstruction accuracy.',
        'Neither low CV nor modality differences prove recovery of a missing frequency band.',
        'Inspect grid_geometry_audit.csv before assigning physical-scale interpretations.',
        'Non-significance is not equivalence. Conclusions condition on these trained checkpoints and fixed seeds.',
        '', '## Files',
        'modality_level_summary.csv: raw weights/shares, patient SD/CV/median and pointwise mean CIs.',
        'paired_modality_comparisons.csv: all planned comparisons, unadjusted and Holm-adjusted p values.',
        'four_modality_profiles.png: means and pointwise patient-level 95% CIs; overlap is not a paired test.',
        'four_modality_boxplots.png and four_modality_heatmap.png: patient variability.',
        'spatial/: rendered native mid-plane maps, if present in source extractions; same raw color scale.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest['status']='COMPLETE'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',nargs='+',required=True,help='One complete 4-modality directory, or nonoverlapping 2-modality directories')
    p.add_argument('--out',required=True)
    a=p.parse_args();out=prepare_out(a.out)
    manifest=dict(protocol_version=VERSION,status='RUNNING',analysis_role='exploratory 4x modality analysis',
        configuration=vars(a),started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        code_sha256={n:sha_file(Path(__file__).parent/n) for n in ['analyze_4x_modalities.py','gating_common.py']})
    try:run(a,out,manifest)
    except Exception:
        manifest['status']='FAILED';(out/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(traceback.format_exc(),file=sys.stderr)
    finally:
        manifest['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat();write_json(out/'run_manifest.json',manifest)
    print(f'[{manifest["status"]}] {out}')
    return int(manifest['status']=='FAILED')

if __name__=='__main__':raise SystemExit(main())
