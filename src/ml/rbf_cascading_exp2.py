"""Final COMP9517 test evaluation for the frozen two-stage HOG+SVM cascade.

Use in the SAME Jupyter kernel, immediately after the validation experiment:
    %run -i ./src/ml/rbf_cascading_exp.py
    %run -i ./src/ml/cascade_final_test.py

Prerequisites from machine_learning.ipynb:
    data_dir, output_dir, model_det, best_rbf,
    load_split_subset, run_raw, postprocess, BEST_NMS, detection_ap
Functions from rbf_cascading_exp.py:
    rescore_one_image, finalise_candidates

Reads the already-selected cascade configuration and global score-normalisation
statistics from outputs/cascade_validation_results.json. Never tunes on test.
Saves outputs/cascade_test_results.json and both trained models for reproduction.
"""

from pathlib import Path
import json
import time
import numpy as np
import joblib

_REQUIRED = [
    'output_dir', 'model_det', 'best_rbf', 'load_split_subset', 'run_raw',
    'postprocess', 'BEST_NMS', 'detection_ap', 'rescore_one_image',
    'finalise_candidates', 'CASCADE_MAX_CANDIDATES', 'CASCADE_PRE_NMS',
    'CASCADE_MAX_RAW_BEFORE_NMS',
]
_missing = [name for name in _REQUIRED if name not in globals()]
if _missing:
    raise RuntimeError(
        'Missing variables/functions: ' + ', '.join(_missing) + '. '
        'Run the ML notebook and its cascade validation cell first, '
        'in the same active Jupyter kernel.'
    )

output_dir = Path(output_dir)
validation_file = output_dir / 'cascade_validation_results.json'
test_file = output_dir / 'cascade_test_results.json'
if not validation_file.is_file():
    raise FileNotFoundError(
        f'{validation_file} not found. Run the validation cascade first.'
    )

with validation_file.open(encoding='utf-8') as f:
    validation = json.load(f)
if validation.get('split') != 'validation':
    raise ValueError('Expected validation-only calibration data.')

# Freeze all hyperparameters and score calibration before touching test images.
selection = validation['best_cascade']
alpha = float(selection['alpha_rbf'])
final_nms = float(selection['nms_iou'])
calibration = validation['normalization']
lin_mu = float(calibration['linear_mean'])
lin_sd = float(calibration['linear_std'])
rbf_mu = float(calibration['rbf_mean'])
rbf_sd = float(calibration['rbf_std'])
if min(lin_sd, rbf_sd) <= 0:
    raise ValueError('Invalid validation score-normalisation standard deviation.')

assert CASCADE_MAX_CANDIDATES == int(validation['shortlist']), 'Shortlist config changed!'
assert CASCADE_PRE_NMS == float(validation['pre_nms_iou']), 'Pre-NMS config changed!'
assert CASCADE_MAX_RAW_BEFORE_NMS == int(validation['max_raw_before_nms']), 'Raw candidate limit changed!'

# To avoid accidentally treating the held-out test set as a development set,
# do not silently run a second test evaluation if one was already saved.
if test_file.exists():
    print(f'Existing FINAL test result found at {test_file}.')
    print('No new test evaluation was run. Open that JSON file for the result.')
else:
    test_files = load_split_subset('test', None)
    if len(test_files) == 0:
        raise ValueError('No test images found.')

    print(f'Final test: {len(test_files)} images. Frozen RBF weight={alpha}, NMS={final_nms}.')
    print('Running original linear proposal detector...')
    raw_test, gt_test, linear_seconds = run_raw(test_files, 'test', model_det)

    # Evaluate the original method on this EXACT test run for a fair comparison.
    original_predictions = [
        postprocess(boxes, scores, nms_iou=BEST_NMS)
        for boxes, scores in raw_test
    ]
    baseline = detection_ap(original_predictions, gt_test)

    print('Re-scoring shortlisted proposals with the RBF SVM...')
    start_rescore = time.perf_counter()
    cascade_candidates_test = []
    for idx, (path, (boxes, scores)) in enumerate(zip(test_files, raw_test), 1):
        cascade_candidates_test.append(
            rescore_one_image(path, boxes, scores, best_rbf)
        )
        if idx % 25 == 0 or idx == len(test_files):
            print(f'RBF rescoring completed: {idx}/{len(test_files)} test images', flush=True)
    rescore_seconds = time.perf_counter() - start_rescore

    # IMPORTANT: use the frozen VALIDATION-set mean/std. Never recompute
    # score normalisation or optimise alpha/NMS based on test predictions.
    start_post = time.perf_counter()
    cascade_predictions_test = []
    for boxes, linear_score, rbf_score in cascade_candidates_test:
        fused_score = (
            (1.0 - alpha) * ((linear_score - lin_mu) / lin_sd)
            + alpha * ((rbf_score - rbf_mu) / rbf_sd)
        )
        cascade_predictions_test.append(
            finalise_candidates(boxes, fused_score, final_nms)
        )
    post_seconds = time.perf_counter() - start_post

    cascade = detection_ap(cascade_predictions_test, gt_test)
    n_images = len(test_files)
    baseline_ap = float(baseline['AP'])
    cascade_ap = float(cascade['AP'])
    delta = cascade_ap - baseline_ap

    output = {
        'split': 'test',
        'n_images': n_images,
        'selection': {
            'source': 'validation_only',
            'alpha_rbf': alpha,
            'nms_iou': final_nms,
            'validation_AP50': float(selection['AP50']),
            'shortlist': int(validation['shortlist']),
            'pre_nms_iou': float(validation['pre_nms_iou']),
            'max_raw_before_nms': int(validation['max_raw_before_nms']),
            'normalization': calibration,
        },
        'linear_baseline': {
            'AP50': baseline_ap,
            'TP': int(baseline['TP']),
            'FP': int(baseline['FP']),
            'GT': int(baseline['n_gt']),
        },
        'rbf_cascade': {
            'AP50': cascade_ap,
            'TP': int(cascade['TP']),
            'FP': int(cascade['FP']),
            'GT': int(cascade['n_gt']),
        },
        'delta_AP50': delta,
        'relative_improvement_percent': (100 * delta / baseline_ap if baseline_ap else None),
        'timing_s': {
            'linear_proposals_total': float(linear_seconds),
            'rbf_rescore_total': float(rescore_seconds),
            'cascade_postprocessing_total': float(post_seconds),
            'linear_seconds_per_image': float(linear_seconds / n_images),
            'cascade_seconds_per_image': float(
                (linear_seconds + rescore_seconds + post_seconds) / n_images
            ),
        },
    }

    # Save both fitted classifiers: the cascade needs the linear proposal
    # detector AND the RBF rescoring model to be reproduced later.
    joblib.dump(model_det, output_dir / 'svm_detector.joblib')
    joblib.dump(best_rbf, output_dir / 'rbf_patch_svm.joblib')
    with test_file.open('w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print('\n=== FROZEN FINAL TEST RESULTS ===')
    print(f'Original linear SVM AP@0.5: {baseline_ap:.4f} ({baseline_ap*100:.2f}%)')
    print(f'RBF cascade AP@0.5:        {cascade_ap:.4f} ({cascade_ap*100:.2f}%)')
    print(f'Difference:               {delta:+.4f} ({delta*100:+.2f} percentage points)')
    print(f'Original time/image:      {linear_seconds/n_images:.2f} s')
    print(f'Cascade time/image:       {(linear_seconds+rescore_seconds+post_seconds)/n_images:.2f} s')
    print(f'Saved {test_file}')
