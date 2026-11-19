"""Checks for collaborator setup and frozen artifacts; no training/test inference."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import csv

import numpy as np
import torch
from faster_rcnn import fetch_evaluator
from faster_rcnn.core import configure_data, resolve_data
from faster_rcnn.upstream import DEFAULT_WHEEL, load_exact_evaluator, prediction_tp

ROOT = Path(__file__).resolve().parents[1]


class SubmissionChecks(unittest.TestCase):
    def test_download_verified_before_publish(self):
        content = b'controlled test distribution'
        lock = {'filename':'sample.whl', 'url':'https://files.pythonhosted.org/test',
                'sha256':hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            lock_path = folder/'lock.json'
            lock_path.write_text(json.dumps(lock))
            destination = folder/'sample.whl'
            with patch.object(fetch_evaluator, 'LOCK', lock_path), patch.object(
                    fetch_evaluator.urllib.request, 'urlopen', return_value=io.BytesIO(content)):
                self.assertEqual(fetch_evaluator.acquire(destination).read_bytes(), content)
            destination.write_bytes(b'corrupted')
            with patch.object(fetch_evaluator, 'LOCK', lock_path):
                with self.assertRaises(ValueError): fetch_evaluator.acquire(destination)

    def test_bad_download_never_published(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); destination=folder/'sample.whl'; lock_path=folder/'lock.json'
            lock_path.write_text(json.dumps({'filename':'sample.whl','url':'https://files.pythonhosted.org/test','sha256':'0'*64}))
            with patch.object(fetch_evaluator, 'LOCK', lock_path), patch.object(
                    fetch_evaluator.urllib.request, 'urlopen', return_value=io.BytesIO(b'invalid')):
                with self.assertRaises(ValueError): fetch_evaluator.acquire(destination)
            self.assertFalse(destination.exists())

    def test_relative_dataset_template_and_version_guard(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            folder=Path(directory); data=folder/'data with spaces'
            (data/'images/train').mkdir(parents=True);(data/'labels/train').mkdir(parents=True)
            config=folder/'config.json'
            config.write_text(json.dumps({'dataset':'liugalhigh/beach-bird/versions/3','data_root':'data with spaces'}))
            configure_data(data_config=config)
            self.assertEqual(resolve_data(),data)
            config.write_text(json.dumps({'dataset':'unverified/versions/4','data_root':'data with spaces'}))
            with self.assertRaises(ValueError): configure_data(data_config=config)

    @unittest.skipUnless(DEFAULT_WHEEL.is_file(), 'Acquire pinned evaluator before metric checks')
    def test_original_metric_functions_and_synthetic_matching(self):
        namespace,_,functions=load_exact_evaluator()
        recorded=json.loads((ROOT/'results/verification.json').read_text())
        self.assertEqual({k:hashlib.sha256(v.encode()).hexdigest() for k,v in functions.items()},recorded['function_sha256'])
        target={'boxes':torch.tensor([[0.,0.,10.,10.]]),'labels':torch.tensor([1])}
        tp=prediction_tp(namespace,target['boxes'],target['labels'],target)
        self.assertTrue(tp.all())
        self.assertFalse(prediction_tp(namespace,torch.tensor([[20.,20.,30.,30.]]),target['labels'],target).any())
        ap=namespace['ap_per_class'](tp,np.array([.9]),np.array([1]),np.array([1]),plot=False,names={1:'bird'})[5]
        self.assertGreater(float(ap.mean()),.99)

    def test_freeze_and_result_coverage(self):
        with (ROOT/'results/results.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),50)
        self.assertEqual({(r['experiment'],r['split'],r['condition']) for r in rows},
            {(e,s,c) for e in 'ABCDE' for s in ('val','test') for c in ('clean','dark_mild','dark_strong','blur_mild','blur_strong')})
        validation_means={e:sum(float(r['map_50_95']) for r in rows if r['split']=='val' and r['experiment']==e)/5 for e in 'ABCDE'}
        freeze=json.loads((ROOT/'results/freeze.json').read_text())
        self.assertEqual(max('ABCDE',key=lambda e:validation_means[e]),freeze['final_choice'])
        self.assertFalse(freeze['test_tuning'])
        plan=json.loads((ROOT/'results/improvement_plan.json').read_text())
        self.assertEqual(plan['single_change'],'blur_probability 0.3 -> .6')
        for row in rows:
            config=json.loads(row['config'])
            self.assertEqual(config['max_detections'],300)
            self.assertEqual(config['score_threshold'],.001)
            self.assertEqual(config['pr_confidence'],.25)


if __name__=='__main__': unittest.main()
