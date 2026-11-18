# Frozen Faster R-CNN results for group comparison

These are selected copies of the completed run, not a new experiment. Their
bytes match the original files in the author's local ignored `outputs/`.
No training or test inference was rerun to prepare this submission.

| File | Meaning |
|---|---|
| `results.csv` | 50 rows: A–E × validation/test × five conditions |
| `freeze.json` | Validation selection rule, final choice D, checkpoint SHA256 |
| `improvement_plan.json` | E starts fresh with D's settings, changing only blur probability 0.3 → 0.6 |
| `verification.json` | Historical environment and function checks from the original run; not a claim about the reader's machine |
| `teammate_alignment.json` | Original-image/label hash checks and validation corruption pixel checks against teammate artifacts |

`experiment` identifies A–E. `split=val` is validation, `split=test` is the
final held-out evaluation. `condition` is clean, dark_mild (×0.7), dark_strong
(×0.4), blur_mild (PIL GaussianBlur radius 1), or blur_strong (radius 2).
Corruptions precede model resizing; annotations remain unchanged.

All AP/P/R values are fractions in [0,1]. `map_50` is AP at matching IoU 0.50;
`map_50_95` averages AP at 0.50:0.05:0.95. There is one bird class, so class AP
equals mAP. AP uses the pinned Ultralytics evaluator with score cutoff 0.001.
`precision/recall/tp/fp/fn` use confidence ≥0.25, matching IoU ≥0.50, upstream
matching recomputed after filtering, and global micro counts.

Input preserves aspect ratio with short edge 800 and long edge capped at 1333,
then stride-32 padding. Maximum final detections=300; ROI NMS IoU=0.7 and RPN
NMS IoU=0.7 are separate from AP matching IoUs. `config` stores the experiment
settings. Each experiment starts independently from the same COCO weights,
seed 42, and 15-epoch budget. A=no random augmentation; B=flip; C=flip plus
brightness/contrast; D=C plus blur; E=D with increased blur probability.

`selected_epoch` maximizes clean validation AP50–95 within an experiment,
retaining the earliest tie. The final model D was chosen by highest mean
five-condition validation AP50–95 across A–E, before any final test inference.
`freeze.json` records that decision. Test scores must not be used to revise it.

`training_seconds` sums training-only epochs, including the training loader
and computation, excluding validation/checkpoint saves. It is not end-to-end
pipeline wall time. `cpu_seconds_per_image` and `cpu_fps` summarize CPU batch-1
timing on 30 fixed preloaded images, five warmups per repeat, three repeats,
four threads. Timing includes tensor conversion, internal preprocessing,
inference, NMS and restoration of original coordinates; disk decoding and
model loading are excluded. These columns repeat across a model's conditions:
they are a **clean-image benchmark**, not per-corruption timing. FPS is
90 divided by the sum of the three repeat durations.

The author's CPU/PyTorch environment differs from the teammate's recorded
benchmark. Compare accuracy under the documented protocol, but remeasure both
models on identical hardware/software before attributing an FPS difference to
architecture. Joint evaluator review with the teammate is still required
before claiming fully verified cross-model alignment.

Dataset images, annotations, checkpoints, bulk predictions, logs and result
images are deliberately not included. The CSV suffices for table comparison;
re-evaluation with a common evaluator requires separately shared predictions
using dataset-relative image identifiers, original-pixel xyxy boxes, labels
(bird=1 here, YOLO bird=0), scores and the documented filtering rules.
