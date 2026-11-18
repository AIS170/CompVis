"""COMP9517 optional HD experiment: two-stage HOG + SVM bird detection.

Run from a machine_learning.ipynb cell AFTER the validation evaluation cells:
    %run -i ./hd_rbf_cascade_experiment.py

Requires existing notebook variables: val_files, raws_f, gts_f, best_rbf,
extract_hog_features, nms, detection_ap, output_dir.
Does not train or overwrite models. Does not evaluate the test set.
"""
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image

# Tunable resource limits (not learned on the test set)
CASCADE_MAX_CANDIDATES = 250       # shortlist per validation image
CASCADE_PRE_NMS = 0.75            # lightly deduplicate linear SVM proposals
CASCADE_MAX_RAW_BEFORE_NMS = 3000 # avoid expensive NMS across all proposals
CASCADE_BATCH_SIZE = 128
CASCADE_FINAL_MAX_BOXES = 300
CASCADE_ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0) # RBF contribution
CASCADE_NMS_GRID = (0.3, 0.5, 0.7)


def shortlist_candidates(raw_box, raw_score):
    """Keep strong but diverse linear-SVM proposals. No ground truth used."""
    raw_box = np.asarray(raw_box, dtype=np.float32).reshape(-1, 4)
    raw_score = np.asarray(raw_score, dtype=np.float32).reshape(-1)
    if len(raw_score) == 0:
        return raw_box, raw_score
    idx = np.argsort(raw_score)[::-1][:CASCADE_MAX_RAW_BEFORE_NMS]
    b, s = raw_box[idx], raw_score[idx]
    selected = np.asarray(nms(b, s, CASCADE_PRE_NMS)[:CASCADE_MAX_CANDIDATES], dtype=int)
    return b[selected], s[selected]


def rescore_one_image(img_path, raw_boxes, raw_scores, rbf_model):
    """Crop ORIGINAL pixels, replicate training's 64x64 HOG, then score via RBF."""
    boxes, linear = shortlist_candidates(raw_boxes, raw_scores)
    if not len(boxes):
        return boxes, linear, np.empty(0, dtype=np.float32)
    feature_vectors = []
    valid_boxes = []
    valid_linear = []
    with Image.open(img_path) as opened:
        image = opened.convert('RGB')
        W, H = image.size
        for box, score in zip(boxes, linear):
            x1 = int(np.clip(np.floor(box[0]), 0, W))
            y1 = int(np.clip(np.floor(box[1]), 0, H))
            x2 = int(np.clip(np.ceil(box[2]), 0, W))
            y2 = int(np.clip(np.ceil(box[3]), 0, H))
            if x2 <= x1 or y2 <= y1:
                continue
            patch = image.crop((x1, y1, x2, y2)).resize((64, 64))
            feature_vectors.append(extract_hog_features(patch))
            valid_boxes.append(box)
            valid_linear.append(score)
    if not feature_vectors:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.float32)
    X = np.asarray(feature_vectors, dtype=np.float32)
    rbf = np.concatenate([
        rbf_model.decision_function(X[i:i+CASCADE_BATCH_SIZE])
        for i in range(0, len(X), CASCADE_BATCH_SIZE)
    ]).astype(np.float32)
    return np.asarray(valid_boxes, np.float32), np.asarray(valid_linear, np.float32), rbf


def finalise_candidates(boxes, scores, nms_iou):
    """Run final NMS on fusion scores; do not apply an arbitrary score floor."""
    if not len(scores):
        return []
    selected = nms(boxes, scores, nms_iou)[:CASCADE_FINAL_MAX_BOXES]
    return [tuple(map(float, boxes[i])) + (float(scores[i]),) for i in selected]


def compute_cascade_validation(val_files, raws_f, gts_f, best_rbf):
    assert len(val_files) == len(raws_f) == len(gts_f), 'Validation images/predictions misaligned'
    if not len(val_files):
        raise ValueError('No validation images available')

    start = time.perf_counter()
    all_cand = []
    for idx, (img_path, (raw_boxes, raw_scores)) in enumerate(zip(val_files, raws_f), start=1):
        all_cand.append(rescore_one_image(img_path, raw_boxes, raw_scores, best_rbf))
        if idx % 25 == 0 or idx == len(val_files):
            print(f'RBF rescoring completed: {idx}/{len(val_files)} validation images', flush=True)

    # Global, label-independent normalization preserves between-image score comparisons.
    all_linear = np.concatenate([c[1] for c in all_cand if len(c[1])])
    all_rbf = np.concatenate([c[2] for c in all_cand if len(c[2])])
    lin_mu, lin_sd = float(all_linear.mean()), max(float(all_linear.std()), 1e-6)
    rbf_mu, rbf_sd = float(all_rbf.mean()), max(float(all_rbf.std()), 1e-6)

    rows = []
    result_predictions = {}
    for alpha in CASCADE_ALPHA_GRID:
        for nms_iou in CASCADE_NMS_GRID:
            preds = []
            for boxes, lin, rbf in all_cand:
                combined = (1.0 - alpha) * ((lin - lin_mu) / lin_sd) + alpha * ((rbf - rbf_mu) / rbf_sd)
                preds.append(finalise_candidates(boxes, combined, nms_iou))
            r = detection_ap(preds, gts_f)
            row = {
                'alpha_rbf': float(alpha),
                'nms_iou': float(nms_iou),
                'AP50': float(r['AP']),
                'TP': int(r['TP']),
                'FP': int(r['FP']),
                'GT': int(r['n_gt'])
            }
            rows.append(row)
            result_predictions[(alpha, nms_iou)] = preds
            print(f"RBF weight={alpha:4.2f} | NMS={nms_iou:.1f} | AP@0.5={r['AP']:.4f} "
                  f"| TP={r['TP']} | FP={r['FP']}", flush=True)

    best = max(rows, key=lambda item: item['AP50'])
    baseline_ap = float(detection_ap(
        [postprocess(b, s, nms_iou=BEST_NMS) for b, s in raws_f], gts_f)['AP'])
    delta = best['AP50'] - baseline_ap
    print('\n*** TWO-STAGE DETECTION EXPERIMENT ***')
    print(f'Original linear SVM validation AP@0.5: {baseline_ap:.4f}')
    print(f"Best two-stage validation AP@0.5: {best['AP50']:.4f}")
    print(f'AP change: {delta:+.4f}')
    print(f"Best RBF weight: {best['alpha_rbf']}; best NMS IoU: {best['nms_iou']}")
    print(f'Additional RBF rescoring/validation time: {time.perf_counter() - start:.1f} seconds')

    summary = {
        'experiment': 'Linear SVM proposals + RBF SVM patch rescoring',
        'split': 'validation',
        'n_images': len(val_files),
        'baseline_AP50': baseline_ap,
        'best_cascade': best,
        'delta_AP50': delta,
        'shortlist': CASCADE_MAX_CANDIDATES,
        'pre_nms_iou': CASCADE_PRE_NMS,
        'max_raw_before_nms': CASCADE_MAX_RAW_BEFORE_NMS,
        'normalization': {
            'linear_mean': lin_mu, 'linear_std': lin_sd,
            'rbf_mean': rbf_mu, 'rbf_std': rbf_sd
        },
        'all_configs': rows,
    }
    Path(output_dir).mkdir(exist_ok=True)
    with open(Path(output_dir) / 'cascade_validation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved: outputs/cascade_validation_results.json')
    return summary, all_cand, result_predictions[(best['alpha_rbf'], best['nms_iou'])]


# Invoke after original notebook's validation cells have been run.
cascade_summary, cascade_candidates, cascade_val_predictions = compute_cascade_validation(
    val_files, raws_f, gts_f, best_rbf
)
