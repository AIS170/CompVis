import ast, zipfile
from .fetch_evaluator import verify_wheel
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
DEFAULT_WHEEL = Path(__file__).parent / "vendor/ultralytics-8.4.208-py3-none-any.whl"
def load_exact_evaluator(wheel=DEFAULT_WHEEL):
    """Compile only named pure metric functions; do not execute package startup."""
    if not Path(wheel).is_file():
        raise FileNotFoundError('Run python -m faster_rcnn.fetch_evaluator to acquire the pinned evaluator')
    verify_wheel(wheel)
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read('ultralytics-8.4.208.dist-info/METADATA').decode()
        if '\nVersion: 8.4.208\n' not in metadata:
            raise ValueError('Expected the official 8.4.208 wheel')
        files = {name: archive.read(name).decode() for name in (
            'ultralytics/utils/metrics.py', 'ultralytics/engine/validator.py',
            'ultralytics/models/yolo/detect/val.py')}
    namespace = {'np': np, 'torch': torch, 'Path': Path,
                 '_AP_X': np.linspace(0, 1, 101),
                 '_TRAPZ': np.trapezoid if hasattr(np, 'trapezoid') else np.trapz}
    selected = {}
    for name, wanted in [
        ('ultralytics/utils/metrics.py', ['box_iou', 'smooth', 'compute_ap', 'ap_per_class']),
        ('ultralytics/engine/validator.py', ['match_predictions'])]:
        tree = ast.parse(files[name])
        nodes = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in wanted]
        if sorted(node.name for node in nodes) != sorted(wanted):
            raise ValueError(f'Missing/duplicate upstream functions in {name}')
        for node in nodes:
            selected[node.name] = ast.get_source_segment(files[name], node)
            exec(compile(ast.Module(body=[node], type_ignores=[]), name, 'exec'), namespace)
    return namespace, files, selected


def prediction_tp(namespace, boxes, labels, target):
    # Identical default iouv and use_scipy=False to DetectionValidator.
    validator = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    if not len(boxes) or not len(target['boxes']):
        return np.zeros((len(boxes), 10), dtype=bool)
    iou = namespace['box_iou'](target['boxes'], boxes)
    return namespace['match_predictions'](
        validator, labels, target['labels'], iou).numpy()
