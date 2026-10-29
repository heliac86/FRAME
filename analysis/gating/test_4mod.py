"""CPU regression checks; real checkpoint querying still requires server preflight."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
from gating_common import (MODALITIES,VERSION,index_files,patient_means,write_json,sha_file,holm)
from analyze_4x_modalities import (merge_inputs,modality_comparisons,summarize,MODEL_KEYS)

HERE=Path(__file__).resolve().parent

def fixture(folder,mods=MODALITIES,version=VERSION,factor=4):
    folder.mkdir();keys=[f'{x:03d}' for x in range(1,9)];rows=[];audit=[]
    base={'flair':[.2,.5,.3],'t1ce':[.25,.4,.35],'t1':[.15,.6,.25],'t2':[.3,.3,.4]}
    for i,p in enumerate(keys):
        for mod in mods:
            j=MODALITIES.index(mod)
            q=np.array(base[mod])+np.array([1,-1,0])*(i*.001*(j+1))
            for lv,value in enumerate(q,1):
                rows.append(dict(patient=p,modality=mod,factor=factor,level=lv,
                                 weight_mean=3*value,share_mean=value,n_points=20))
            cfg={k:1 for k in MODEL_KEYS}
            cfg.update(n_levels=3,constructor='training',strict_load=True,
                       base_resolution=10,physical_max_span=160.,training_seed='42')
            audit.append(dict(patient=p,modality=mod,factor=factor,model=cfg,
                checkpoint_sha256=f'{p}/{mod}',n_evaluated=20,
                roi=dict(n_points=20,n_gt_brain=200,coordinates_sha256=f'{p}/{mod}/coords')))
    pd.DataFrame(rows).to_csv(folder/'gating_weights_raw.csv',index=False)
    write_json(folder/'checkpoint_audit.json',audit)
    write_json(folder/'run_manifest.json',dict(status='COMPLETE',protocol_version=version,
        patient_ids=keys,n_levels=3,test_csv_sha256='same_test_csv',
        configuration=dict(factor=factor,points=20,seed=42,bounds_tolerance_mm=.0001,constructor='training'),
        runtime=dict(models_sha256='same_model'),code_sha256={'gating_backend.py':'same_backend'},
        raw_csv_sha256=sha_file(folder/'gating_weights_raw.csv'),
        checkpoint_audit_sha256=sha_file(folder/'checkpoint_audit.json')))
    return keys

class DiscoveryTests(unittest.TestCase):
    def test_t1_and_t1ce_are_distinct(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for name in ['003_t1ce_4x5_FRAME_gate_L10.pt','003_t1_4x5_L10.pt','003_t2_4x5_L10.pt']:
                (root/name).touch()
            d=index_files(root,['003'],('.pt',))
            self.assertEqual(len(d['003','t1']),1)
            self.assertEqual(len(d['003','t1ce']),1)
            self.assertIn('_t1_',d['003','t1'][0])
            d=index_files(root,['003'],('.pt',),modalities=('t1','t2'))
            self.assertEqual(set(d),{('003','t1'),('003','t2')})

    def test_discovery_cli_four_modalities_and_t1t2_only(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);gt=root/'gt';gt.mkdir();rec=root/'rec';rec.mkdir()
            csv=root/'test.csv';csv.write_text('Brats20ID\nBraTS20_Training_003\n')
            for mod in MODALITIES:
                (gt/f'BraTS20_Training_003_{mod}.nii').touch()
                stem=f'003_{mod}_4x5_'+('FRAME_gate_L10' if mod in ('flair','t1ce') else 'L10')
                for ext in ('.nii.gz','.pt'):(rec/(stem+ext)).touch()
            for label,mods in [('four',list(MODALITIES)),('extra',['t1','t2'])]:
                out=root/label
                r=subprocess.run([sys.executable,str(HERE/'extract_4x_gating.py'),
                    '--modalities',*mods,'--gt-root',str(gt),'--recon-root',str(rec),
                    '--test-csv',str(csv),'--expected-patients','1','--discovery-only','--out',str(out)],
                    capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
                d=pd.read_csv(out/'file_manifest.csv')
                self.assertEqual(set(d.modality),set(mods));self.assertEqual(len(d),len(mods))

    def test_duplicate_candidates_preserved_for_rejection(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for n in ['003_t1_4x5_L10.pt','003_t1_old_L10.pt']:(root/n).touch()
            self.assertEqual(len(index_files(root,['003'],('.pt',))['003','t1']),2)

    def test_extraction_tables_and_npz_no_plot_dependencies(self):
        from extract_4x_gating import condition_summary,save_spatial_data
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root/'data')
            d=pd.read_csv(root/'data'/'gating_weights_raw.csv',dtype={'patient':str})
            out=root/'numbers';out.mkdir()
            with mock.patch.dict(sys.modules,{'matplotlib':None,'scipy':None}):
                condition_summary(d,out,4,MODALITIES)
                save_spatial_data(np.ones((2,3)),np.array([0,3]),(1,2,2),out/'spatial','test',3)
            self.assertTrue((out/'level_summary.csv').exists());self.assertFalse(list(out.rglob('*.png')))

class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.keys=fixture(self.root/'legacy',('flair','t1ce'),'1.1.0')
        fixture(self.root/'extra',('t1','t2'))
    def tearDown(self):self.temp.cleanup()

    def test_reuse_legacy_and_merge(self):
        bundles,raw,keys,geometry=merge_inputs([self.root/'legacy',self.root/'extra'])
        self.assertEqual(keys,self.keys);self.assertEqual(len(raw),8*4*3)
        self.assertEqual(len(geometry),8*4)

    def test_single_four_modality_extraction(self):
        fixture(self.root/'all')
        self.assertEqual(len(merge_inputs([self.root/'all'])[1]),8*4*3)

    def test_paired_tests_and_single_holm_family(self):
        _,raw,keys,_=merge_inputs([self.root/'legacy',self.root/'extra'])
        d=modality_comparisons(raw,keys,[1,2,3])
        self.assertEqual(len(d),18);self.assertTrue(d.n_patients.eq(8).all())
        self.assertTrue(d.holm_family_size.eq(18).all())
        row=d[(d.modality_a=='flair')&(d.modality_b=='t1ce')&(d.level==1)].iloc[0]
        self.assertAlmostEqual(row.delta,-5.35)
        np.testing.assert_allclose(d.p_holm.fillna(1.),holm(d.p_raw.fillna(1.)))
        self.assertTrue(d.loc[d.p_raw.isna(),'inference_flag'].eq('constant_nonzero_difference_no_t_inference').all())

    def test_four_mean_does_not_replace_two_mean(self):
        _,raw,_,_=merge_inputs([self.root/'legacy',self.root/'extra'])
        summary,four,two=summarize(raw)
        a=four[(four.patient=='001')&(four.level==1)].share_mean.item()
        b=two[(two.patient=='001')&(two.level==1)].share_mean.item()
        self.assertAlmostEqual(a,.225);self.assertAlmostEqual(b,.225)
        a=four[(four.patient=='001')&(four.level==2)].share_mean.item()
        b=two[(two.patient=='001')&(two.level==2)].share_mean.item()
        self.assertAlmostEqual(a,.45);self.assertAlmostEqual(b,.45)
        # Different patient perturbations expose the distinct averaging definitions.
        a=four[(four.patient=='008')&(four.level==1)].share_mean.item()
        b=two[(two.patient=='008')&(two.level==1)].share_mean.item()
        self.assertNotAlmostEqual(a,b)
        self.assertTrue(summary.n_patients.eq(8).all())

    def test_missing_modality_rejected(self):
        with self.assertRaisesRegex(ValueError,'Four complete modalities'):merge_inputs([self.root/'legacy'])

    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,'Overlapping'):merge_inputs([self.root/'legacy',self.root/'legacy',self.root/'extra'])

    def test_wrong_factor_rejected(self):
        fixture(self.root/'x2',MODALITIES,factor=2)
        with self.assertRaisesRegex(ValueError,'4x ONLY'):merge_inputs([self.root/'x2'])

    def test_modified_raw_rejected(self):
        p=self.root/'extra'/'gating_weights_raw.csv';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'changed'):merge_inputs([self.root/'legacy',self.root/'extra'])

    def test_different_sampling_settings_rejected(self):
        p=self.root/'extra'/'run_manifest.json';m=json.loads(p.read_text());m['configuration']['seed']=99
        write_json(p,m)
        with self.assertRaisesRegex(ValueError,'seed'):merge_inputs([self.root/'legacy',self.root/'extra'])

    def test_different_backend_rejected(self):
        p=self.root/'extra'/'run_manifest.json';m=json.loads(p.read_text());m['code_sha256']['gating_backend.py']='changed'
        write_json(p,m)
        with self.assertRaisesRegex(ValueError,'backend differs'):merge_inputs([self.root/'legacy',self.root/'extra'])

    def test_cli_without_torch_nesvor(self):
        guard=self.root/'guard';guard.mkdir()
        (guard/'sitecustomize.py').write_text('''import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','nesvor','tinycudann'):
        raise ImportError('Backend blocked for analysis isolation test')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
''')
        spatial=self.root/'extra'/'spatial';spatial.mkdir()
        np.savez_compressed(spatial/'001_t1_4x.npz',weights=np.ones((3,4,3)))
        out=self.root/'analysis';env=dict(os.environ);env['PYTHONPATH']=str(guard)
        r=subprocess.run([sys.executable,str(HERE/'analyze_4x_modalities.py'),'--inputs',
            str(self.root/'legacy'),str(self.root/'extra'),'--out',str(out)],
            capture_output=True,text=True,env=env)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(json.loads((out/'run_manifest.json').read_text())['n_patients'],8)
        for name in ['four_modality_profiles.png','four_modality_boxplots.png',
                     'four_modality_heatmap.png','spatial/001_t1_4x.png','REPORT.md']:
            self.assertGreater((out/name).stat().st_size,100)

if __name__=='__main__':unittest.main(verbosity=2)
