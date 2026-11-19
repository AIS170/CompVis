# Third-party evaluator dependency

The evaluator loads selected original functions from **Ultralytics 8.4.208**:
`box_iou`, `smooth`, `compute_ap`, `ap_per_class`, and `match_predictions`.
The package is third-party software by Ultralytics, not original project code.

- Upstream: https://github.com/ultralytics/ultralytics
- Package metadata: https://pypi.org/pypi/ultralytics/8.4.208/json
- Distribution metadata license: **AGPL-3.0**.
- The distribution includes its upstream license materials. Preserve those
  materials and attribution when sharing upstream code or distributions.
- Exact filename, fixed PyPI-hosted URL and SHA256: `evaluator.lock.json`.
- Acquire locally: `python -m faster_rcnn.fetch_evaluator`.
- Verify offline: `python -m faster_rcnn.fetch_evaluator --check`.

The wheel is downloaded to ignored `faster_rcnn/vendor/` and is not included
in this Git submission. It is not installed or imported as a package.
`upstream.py` verifies the wheel and compiles the named original functions
without changing their matching or AP definitions. The original experiment's
function hashes are preserved in `results/verification.json`.

Faster R-CNN model construction and pretrained weights come from TorchVision
(PyTorch project); the weights are not included in this submission.
The training/data implementation was adapted from this group's earlier local
CompVis work and revised for this protocol. See `README.md` for the method
references and the frozen experimental settings.
