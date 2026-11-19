# Faster R-CNN: Beach Bird Experiments

Experiments A-E are complete. The frozen final model is **D**. Use the [selected results and column definitions](results/README.md) for group comparisons without retraining or rerunning test evaluation. Large original artifacts remain in the local ignored `outputs/` directory. The two-image smoke run is a pipeline check, not a reportable experiment. Results from the older CompVis implementation are not part of this experiment.

## Environment and data audit

The recorded environment was Python 3.12.4, PyTorch 2.6.0+cu124, TorchVision 0.21.0+cu124, an RTX 4070 Laptop GPU with 8 GB of memory, and an Intel Core i9-13980HX CPU. CPU timing used four threads. These are historical experiment details, not claims about a teammate's hardware or a fresh installation. Standalone setup instructions are below; no sibling repository or personal environment is required. The course specification was reviewed, and the experiment follows the protocol requested for this project.

The dataset is located through `kagglehub.dataset_download('liugalhigh/beach-bird/versions/3')`, followed by inspection of its actual directory structure. Alternatively, configure your own dataset root using the options below; no author's local path or uncommitted audit file is needed. Images in `images/{train,val,test}` correspond to annotations in `labels/{train,val,test}`. YOLO class-0 normalized `xc yc w h` annotations are converted to absolute original-image `xyxy` coordinates, with bird=1 and background=0. The original splits are preserved.

| Split | Images | Boxes | Negative images | Classes |
|---|---:|---:|---:|---:|
| train | 1369 | 8663 | 154 | 1 |
| val | 291 | 1843 | 33 | 1 |
| test | 240 | 1518 | 27 | 1 |

Every image was decoded and checked for missing annotations, non-finite or invalid boxes, out-of-bounds coordinates, empty annotations, duplicate stems, and orphan labels. No issues were found. Negative images retain empty `(0,4)` boxes. Invalid input raises an error rather than being silently discarded; clipping is allowed only within a normalized-coordinate floating-point tolerance of 1e-6. RGB conversion occurs in memory without modifying source data. Training records SHA256 manifests for train/validation images and annotations. Initial test access was limited to structural and file-identity auditing, without predictions or tuning.

[The alignment summary](results/teammate_alignment.json) records the counts of successful SHA256 identity checks for all 240 original test images and annotations in the teammate's protocol; it does not contain the individual file hashes. All 1455 derived validation images across five conditions matched pixel-for-pixel, and their 1455 annotation copies matched byte-for-byte.

## Model and training configuration

The model is TorchVision Faster R-CNN ResNet50-FPN with COCO_V1 pretrained weights, a new two-class prediction head, and the last three backbone stages trainable. This experiment consistently uses FPN V1 with the existing complete pretrained weights, not a mixture of V1 and V2. Each independent experiment resets seed 42 and starts from the same COCO weights and identically initialized random prediction head. Experiments do not continue from another augmentation experiment's checkpoint.

Original-resolution RGB float32 tensors in [0,1] undergo internal COCO mean/std normalization, aspect-preserving resizing with short edge 800 and long edge capped at 1333, and stride-32 batch padding. Predictions are mapped back to original-image `xyxy` coordinates before evaluation, without another letterbox operation. Exports record actual `input_hw`, `resized_hw`, and `padded_batch_shape`. A 300x600 input was verified to resize to 666x1333 and pad to 672x1344.

The five FPN levels use anchor sizes 16/32/64/128/256, respectively, with height/width ratios .5/1/2 at each level. The original candidate sizes 8/16/32/64/128 broadly cover sqrt(area) values from 4 to 181, containing 86.38% of scaled train boxes. Train sqrt(area) statistics were median=67.80, P5=20.16, P95=280.52, and maximum=771.61. The sizes were therefore shifted up one level before formal A-D experiments, using train statistics only, to accommodate larger boxes. These statistics use an analytical 800/1333 scaling approximation; prediction exports record actual integer-rounded sizes. Extreme small, large, and elongated boxes may still have limited anchor coverage. Anchor sizes are not hard detection-size limits. Neither test nor validation predictions informed this choice.

All A-D experiments use SGD with lr=.005, momentum=.9, weight_decay=.0005, and 15 epochs. MultiStepLR multiplies the learning rate by .1 after epochs 10 and 13: epochs 1-10 use .005, 11-13 use .0005, and 14-15 use .00005. Batch size=2, AMP is disabled, gradient accumulation=1, and workers=0. The GPU smoke check passed. Formal runtime errors stop execution and are recorded rather than silently changing an individual experiment's configuration.

Python, NumPy, Torch, and CUDA use seed 42. The DataLoader has an independent generator seeded with 42; workers=0 requires no worker seeds. cuDNN benchmark=False and deterministic=True. Bitwise reproducibility across hardware or software versions is not guaranteed. Every epoch is evaluated on validation data. The checkpoint with maximum clean validation mAP50-95 is selected, retaining the earliest epoch on ties. Best/last checkpoints, optimizer/scheduler states, component losses, learning rates, and training times are recorded. A-D differ only in augmentation.

## A-D training augmentation

Operations run at original resolution in this order: flip -> photometric transform -> blur -> tensor conversion -> internal model preprocessing.

- A: no random augmentation.
- B: horizontal flip with p=.5, updating boxes as xmin'=W-xmax and xmax'=W-xmin.
- C: B plus a shared p=.5 photometric trigger, with independent uniform alpha[.8,1.2] and beta[-25.5,25.5]. Pixels use `round(clip(alpha*RGB_uint8+beta,0,255)).astype(uint8)`.
- D: C plus PIL GaussianBlur with p=.3 and radius sampled uniformly from [.5,2].

Validation and test data receive no random training augmentation.

## Evaluator and group alignment

The original `box_iou`, `compute_ap`, `ap_per_class`, `match_predictions`, and supporting functions are extracted directly from the official Ultralytics 8.4.208 wheel, matching the teammate's recorded version. COCO AP is not substituted for YOLO AP. The wheel's package initialization is not executed, and a second Ultralytics installation is unnecessary. The wheel is not committed: `python -m faster_rcnn.fetch_evaluator` requires network access to the pinned `files.pythonhosted.org` URL when the wheel is absent. An existing valid local wheel is reused, and `--check` verifies it without network access. SHA256 is checked before every load. See [the evaluator lock](evaluator.lock.json) for the source, version, and hash, [third-party attribution](THIRD_PARTY.md), and [the original verification record](results/verification.json) for function hashes. Acquisition does not alter the original AP or matching logic.

The local distribution metadata identifies version 8.4.208 and license AGPL-3.0, includes `ultralytics-8.4.208.dist-info/licenses/LICENSE`, and matches SHA256 `e4798477691ce744e84bfb31d7d7cff2a0875846533d4f330f19d41c477c932e` in the lock and original verification record. `THIRD_PARTY.md` describes compiling the selected original functions without installing/importing the full package. These are verified attribution and implementation facts, not a conclusion about license compliance or an exemption based on downloading the dependency separately; any unresolved distribution obligations require review before redistribution.

YOLO bird class 0 maps to Faster R-CNN bird class 1. AP uses IoU=.50:.05:.95, the upstream 101-point interpolation/integration, and `use_scipy=False` matching and deduplication. The ROI head retains scores >.001, and returned predictions are exported directly, satisfying the .001 confidence cutoff. Each image has at most 300 final detections. Precision/recall first filter confidence>=.25, then recompute matching with the same upstream matcher at IoU>=.50 and aggregate global micro TP/FP/FN. An automatically selected maximum-F1 threshold is not used.

ROI NMS=.7 and RPN NMS=.7 are separate from AP matching thresholds. RPN pre-NMS train/test limits are 2000/1000, and post-NMS limits are 2000/1000, both above 300. Score filtering and NMS may still yield fewer than 300 predictions; outputs are not padded with boxes.

Local checks cover identical upstream functions, dataset identity, and validation corruption consistency. Joint review with the teammate is still pending. Confirm common re-evaluation, fixed-threshold P/R matching, image identity, and timing boundaries before claiming fully verified cross-model alignment.

## Robustness, failure analysis, and improvement E

The five conditions are clean, dark_mild (pixels multiplied by .7), dark_strong (multiplied by .4), and PIL GaussianBlur with radius 1 or 2. Corruptions precede model resizing, and annotations remain unchanged. Darkness uses `clip(float32 RGB*factor,0,255).astype(uint8)`, with truncation rather than rounding and no JPEG recompression. All derived validation images were checked against the teammate's images pixel-for-pixel.

After A-D validation, representative cases are selected by FN, FP, bird count, and small-box count. Ground truth is green when matched and orange when missed; predictions are cyan when matched and red for false positives, with scores displayed. Visualization uses diagnostic score-greedy matching, while reported metrics use the upstream matcher. Occlusion/NMS, low contrast, and blurred edges are hypotheses to verify visually rather than established causes.

The E design rule was fixed before test evaluation: select the A-D base model with the highest five-condition validation mean AP50-95, compare strong-darkness and strong-blur AP, and change only one training parameter targeting the weaker condition. The recorded base was D, the weaker condition was blur, and E's sole change was blur_probability .3 -> .6. E restarted from COCO weights with the same remaining settings and training budget. `results/improvement_plan.json` records the rationale. Both E and its base model's validation results are retained, without assuming E improves performance. The final frozen selection remained D.

After A-E validation, the final model is selected by five-condition validation mean AP50-95, breaking ties alphabetically. [The original freeze record](results/freeze.json) records that choice and all checkpoint SHA256 values. Each final model/condition then receives one test evaluation. Output directories must be new; overwriting or repeating test evaluation is rejected. Interrupted runs retain their artifacts and stop. Test results must not be used to reselect or retrain models. D remains the validation-frozen final model. The recorded checkpoint hashes identify the original experiment's weight files; those files are not supplied in this repository and cannot be reconstructed or downloaded from a hash alone.

## CPU timing and limitations

[The copied timing protocol](timing_protocol.json) matches the teammate's original `benchmark_settings.json` from benchmark `20261115_153901`, whose field is `warmup_calls_per_model_per_repeat=5`. The teammate's notebook also places the warmup loop inside the repeat/model loops. Both implementations therefore perform **five warmup calls before each repeat for each model: 15 warmup calls per model across three repeats, not five calls in total**. Each repeat then measures 30 images, giving 90 timed images per model. Warmups are excluded from the reported duration. The existing timing results were not rerun for this documentation review because the warmup counts already agree.

The same 30 image names and three timed-image orders are used. Stems map to original JPG files; the teammate's PNG files are clean derived copies. Images are loaded and decoded to RGB before timing. CPU batch size=1 and threads=4. The current Faster R-CNN implementation warms up on canonical image indices 0-4 before every repeat, retains one loaded model across repeats, and times the whole 30-image loop. The teammate warms up on the first five images of that repeat's shuffled order, reloads the model for each model/repeat, and sums separately timed prediction calls. Thus warmup count and timed-image order agree, but warmup-image choice, model lifecycle, and timing boundaries are not identical. A strict timing comparison must first agree on those details and use the same hardware/software; any necessary rerun should be CPU timing only, not accuracy test evaluation.

Timing includes PIL-to-tensor conversion, normalization, resizing, model execution, NMS, and restoration of original coordinates. Disk decoding, model loading, drawing, and evaluation are excluded. Each repeat reports total seconds, per-image latency, and FPS. Aggregate FPS=90/sum(seconds), rather than the mean of individual repeat FPS values.

The teammate's CPU is Family 6 Model 170; the author's i9-13980HX is Model 183. The teammate recorded Torch 2.15 CPU, while this environment used Torch 2.6 CUDA with inference on the CPU for timing. Identical timing procedures do not eliminate hardware/software differences. A strict speed comparison requires both models to be measured on the same machine and environment; current FPS differences cannot be attributed directly to architecture.

## Setup and execution from a fresh clone

The following are Windows PowerShell recipes, run from the repository root with Python 3.12. Create and activate an environment first:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Choose **one** torch installation block. For the CUDA 12.4 build used in the recorded run, with a compatible NVIDIA driver/GPU:

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
@('torch==2.6.0+cu124', 'torchvision==0.21.0+cu124') | Set-Content -Encoding ascii .venv/torch-constraints.txt
```

For CPU-only use, choose this block instead:

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
@('torch==2.6.0+cpu', 'torchvision==0.21.0+cpu') | Set-Content -Encoding ascii .venv/torch-constraints.txt
```

Then install the remaining requirements under the selected torch constraints and check the resolved environment:

```powershell
python -m pip install -c .venv/torch-constraints.txt -r faster_rcnn/requirements.txt
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m faster_rcnn.fetch_evaluator
python -m faster_rcnn.fetch_evaluator --check
```

[The requirements file](requirements.txt) does not itself request torch, torchvision, or the full Ultralytics package. Install the chosen torch pair first; the constraints ensure any later dependency request must remain compatible with that exact CPU/CUDA pair rather than silently selecting another build. Keep these constraints on subsequent installs into this environment. Installation requires network access unless the corresponding packages are already available locally. The CUDA availability output is a diagnostic, not an automatic guarantee of driver compatibility or successful training.

The recorded Windows CUDA environment and earlier CPU checks with CUDA hidden are the only validation evidence described here. The revised install recipes were inspected but were not executed into a fresh environment during this review. A CPU-only wheel installation, Linux/macOS setup, and cross-platform reproduction are not claimed as validated. The Windows `+cpu`/`+cu124` constraints are not universal platform instructions. Dependency versions are a snapshot of the original environment, not proof of current package availability on every platform.

Viewing the recorded results requires no dataset, checkpoints, or training: open `faster_rcnn/results/results.csv`, or use `FasterRCNN.ipynb` after installing its dependencies.

The default dataset is pinned to Kaggle `liugalhigh/beach-bird/versions/3`; downloading an uncached copy requires network access and any authentication required by Kaggle. Set your own local root with `--data-root`, the `FASTER_RCNN_DATA_ROOT` environment variable, or `--data-config faster_rcnn/data_config.example.json`. [The example configuration](data_config.example.json) resolves paths relative to its own directory, so its `../data/Beach Bird Detection Dataset v1.0` value points to `data/Beach Bird Detection Dataset v1.0` under the repository root. Expected structure: `images/{train,val,test}` and `labels/{train,val,test}`. The provider must confirm a local directory is version 3; a path declaration does not verify dataset identity. Compare manifests when available. Do not commit configuration copies containing personal paths. No dataset is included in the repository.

```powershell
python -m faster_rcnn.run --help
python -m faster_rcnn.pipeline --help
# Optional audit and independent functional checks; verify may download COCO weights but does not train
python -m faster_rcnn.run audit --data-root "data/Beach Bird Detection Dataset v1.0"
python -m faster_rcnn.verify
# Reproduction commands for new experiments; not required to view recorded results
python -m faster_rcnn.run train --experiment A --data-config faster_rcnn/data_config.example.json
python -m faster_rcnn.run robust --experiment A --data-config faster_rcnn/data_config.example.json
python -u -m faster_rcnn.pipeline --data-config faster_rcnn/data_config.example.json
```

`train` supports A-D; the pipeline derives E from validation evidence. The complete pipeline trains, freezes, and finally evaluates test data. Do not rerun it merely to compare existing results; existing output directories reject overwriting. A fresh clone contains no checkpoints: `robust/test/timing` require the corresponding local weights and cannot run from result summaries alone. Original freeze records in `results/` are not automatically copied into a new experiment's `outputs/`.

Run collaboration checks with `python -m unittest faster_rcnn.tests.test_submission -v`. Submission preparation verified pinned-wheel acquisition/checking, CLI entry points, five unit checks, and CPU functional checks with CUDA hidden in a temporary source export. It used the existing experiment dependencies, not a fresh installation or all-platform validation, and did not run training or dataset test inference. The original `outputs/verification.json` was not overwritten; `results/verification.json` preserves the original experiment record.

## Committed results and local artifacts

The committed result directory contains exactly these six files:

- [results/README.md](results/README.md): result definitions and comparison limitations.
- [results/results.csv](results/results.csv): 50 rows covering A-E, validation/test, and five conditions.
- [results/freeze.json](results/freeze.json): original validation-selected final model D and checkpoint hashes.
- [results/improvement_plan.json](results/improvement_plan.json): the validation-led E design and its single change.
- [results/verification.json](results/verification.json): original environment, functional checks, wheel hash, and function hashes.
- [results/teammate_alignment.json](results/teammate_alignment.json): historical consistency-check counts, not per-file hashes.

The CSV columns match the actual file:

| Column(s) | Definition |
|---|---|
| `experiment`, `split`, `condition` | A-E; `val` or final held-out `test`; one of the five fixed conditions |
| `map_50`, `map_50_95` | Fractional AP at IoU 0.50 and mean AP over 0.50:0.05:0.95 |
| `precision`, `recall`, `tp`, `fp`, `fn` | Global micro metrics/counts at confidence >=0.25 and matching IoU >=0.50 |
| `selected_epoch` | Best clean-validation AP50-95 epoch, earliest on ties |
| `training_seconds` | Sum of training-only epoch durations, excluding validation and checkpoint saving |
| `cpu_seconds_per_image`, `cpu_fps` | One clean-image CPU benchmark per model, repeated across its validation/test and corruption rows; not per-condition timing |
| `config` | JSON-encoded settings for that experiment |

For three equal 30-image repeats, mean per-image latency is sum(repeat seconds)/90 and aggregate FPS is 90/sum(repeat seconds). The repository does not include `results/results.json`; the CSV is the single committed numerical summary.

Full local runtime outputs can contain manifests, epoch histories, checkpoints, bulk predictions, timing details, failure figures, and logs. They are generated under ignored `outputs/` and are not prerequisites for reading the committed tables. A fresh clone does not contain those files or trained weights. Checkpoint hashes in the committed records are identifiers from the original run, not included weight files. To run checkpoint-dependent commands, obtain the original weights separately through the group's agreed mechanism and verify the corresponding recorded hash, or conduct a clearly separate new experiment without overwriting the frozen records. Neither is required for this documentation review.

Training duration includes the training loader and computation but excludes validation and saving; it is not total pipeline wall time. The final course source ZIP must exclude weights, input images, and result images and respect the 25 MB limit.

Sources: the earlier local CompVis data/training implementation was copied independently and revised for the optimizer, resizing, anchors, checkpoint criterion, and evaluator. Models use TorchVision; evaluation reuses the pinned official Ultralytics wheel. Method references: Ren et al., Faster R-CNN (NeurIPS 2015), and Lin et al., FPN (CVPR 2017). Dataset: Kai Liu, Beach Bird Detection Dataset v1.0, Kaggle version 3.
