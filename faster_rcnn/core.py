"""Experiment A: pretrained Faster R-CNN, no random augmentation.

References and experiment assumptions are documented in faster_rcnn/README.md.
"""
import argparse
import os
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm


@dataclass
class Config:
    epochs: int = 15
    batch_size: int = 2
    seed: int = 42
    lr: float = 0.005
    weight_decay: float = 0.0005
    image_size: int = 800
    max_size: int = 1333
    anchor_sizes: tuple = (16, 32, 64, 128, 256)
    max_detections: int = 300
    score_threshold: float = 0.001
    nms_iou: float = 0.7
    pr_confidence: float = 0.25
    workers: int = 0  # Safe for Windows notebooks.
    horizontal_flip_probability: float = 0.0  # A remains augmentation-free.
    photometric_probability: float = 0.0
    contrast_range: tuple = (0.8, 1.2)
    brightness_range: tuple = (-25.5, 25.5)
    blur_probability: float = 0.0
    blur_radius_range: tuple = (0.5, 2.0)
    dark_probability: float = 0.0


def configure_data(data_root=None, data_config=None):
    if data_root and data_config:
        raise ValueError('Choose --data-root or --data-config, not both')
    if data_config:
        config_path = Path(data_config).resolve()
        settings = json.loads(config_path.read_text(encoding='utf-8'))
        if settings.get('dataset') != 'liugalhigh/beach-bird/versions/3':
            raise ValueError('Data configuration must declare Kaggle version 3')
        configured_root = settings.get('data_root')
        if configured_root:
            data_root = config_path.parent / configured_root
    if data_root:
        os.environ['FASTER_RCNN_DATA_ROOT'] = str(Path(data_root).resolve())


def resolve_data(root=None):
    if root is None:
        root = os.environ.get('FASTER_RCNN_DATA_ROOT')
    if root is None:
        import kagglehub
        root = kagglehub.dataset_download('liugalhigh/beach-bird/versions/3')
    root = Path(root)
    candidates = [root] + [p.parent for p in root.rglob('images') if p.is_dir()]
    candidates = [p for p in candidates if (p / 'images/train').is_dir()
                  and (p / 'labels/train').is_dir()]
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ValueError(f'Expected one dataset root, found {candidates}')
    return candidates[0]


def horizontal_flip(image, target):
    """Flip a PIL image and pixel xyxy boxes; leave the original target intact."""
    boxes = target['boxes'].clone()
    boxes[:, 0] = image.width - target['boxes'][:, 2]
    boxes[:, 2] = image.width - target['boxes'][:, 0]
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), {**target, 'boxes': boxes}


def brightness_contrast(image, contrast, brightness):
    """RGB uint8: clip(alpha * input + beta, 0, 255), rounded to nearest integer."""
    pixels = np.asarray(image, dtype=np.float32)
    pixels = np.rint(np.clip(contrast * pixels + brightness, 0, 255)).astype(np.uint8)
    return Image.fromarray(pixels)


def gaussian_blur(image, radius):
    """PIL GaussianBlur at original resolution; leaves box coordinates unchanged."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


class BirdDataset(Dataset):
    def __init__(self, root, split, horizontal_flip_probability=0.0,
                 photometric_probability=0.0, contrast_range=(0.8, 1.2),
                 brightness_range=(-25.5, 25.5), blur_probability=0.0,
                 blur_radius_range=(0.5, 2.0)):
        if split not in ('train', 'val', 'test'):
            raise ValueError('This baseline entry point only permits train/val.')
        self.root, self.split = Path(root), split
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError('Flip probability must be between 0 and 1')
        if not 0 <= photometric_probability <= 1:
            raise ValueError('Photometric probability must be between 0 and 1')
        if not 0 <= blur_probability <= 1:
            raise ValueError('Blur probability must be between 0 and 1')
        if (len(blur_radius_range) != 2 or not np.isfinite(blur_radius_range).all()
                or not 0 <= blur_radius_range[0] <= blur_radius_range[1]):
            raise ValueError('Blur radius bounds must be finite, ordered and nonnegative')
        if split != 'train' and (horizontal_flip_probability or photometric_probability or blur_probability):
            raise ValueError('Random augmentation is only allowed on train')
        self.horizontal_flip_probability = horizontal_flip_probability
        self.photometric_probability = photometric_probability
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.blur_probability = blur_probability
        self.blur_radius_range = blur_radius_range
        self.paths = sorted(p for p in (self.root / 'images' / split).iterdir()
                            if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if not self.paths:
            raise ValueError(f'No images in {split}')
        stems = [p.stem for p in self.paths]
        if len(stems) != len(set(stems)):
            raise ValueError(f'Duplicate image stems in {split}')

    def __len__(self):
        return len(self.paths)

    def label_path(self, path):
        return self.root / 'labels' / self.split / (path.stem + '.txt')

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as source:
            image = source.convert('RGB')
        width, height = image.size
        label_path = self.label_path(path)
        if not label_path.is_file():
            raise FileNotFoundError(f'Missing annotation: {label_path}')
        boxes = []
        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            values = np.array([float(x) for x in line.split()])
            if len(values) != 5 or not np.isfinite(values).all():
                raise ValueError(f'Invalid annotation: {label_path}:{line_number}')
            cls, xc, yc, bw, bh = values
            if cls != 0 or not (0 <= xc <= 1 and 0 <= yc <= 1
                                and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError(f'Expected class 0 and valid YOLO box: {label_path}:{line_number}')
            if xc-bw/2 < -1e-6 or yc-bh/2 < -1e-6 or xc+bw/2 > 1+1e-6 or yc+bh/2 > 1+1e-6:
                raise ValueError(f'Out-of-bounds annotation: {label_path}:{line_number}')
            box = [max(0, (xc-bw/2)*width), max(0, (yc-bh/2)*height),
                   min(width, (xc+bw/2)*width), min(height, (yc+bh/2)*height)]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f'Degenerate box: {label_path}:{line_number}')
            boxes.append(box)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {'boxes': boxes, 'labels': torch.ones(len(boxes), dtype=torch.int64),
                  'image_id': torch.tensor(index),
                  'area': (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]),
                  'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)}
        # Apply at original resolution, before the model's internal resizing.
        # With probability zero, do not consume RNG: preserve A's behavior.
        if self.horizontal_flip_probability and random.random() < self.horizontal_flip_probability:
            image, target = horizontal_flip(image, target)
        if self.photometric_probability and random.random() < self.photometric_probability:
            image = brightness_contrast(image, random.uniform(*self.contrast_range),
                                        random.uniform(*self.brightness_range))
        if self.blur_probability and random.random() < self.blur_probability:
            image = gaussian_blur(image, random.uniform(*self.blur_radius_range))
        if getattr(self, 'dark_probability', 0) and random.random() < self.dark_probability:
            image = Image.fromarray((np.asarray(image,dtype=np.float32)*random.uniform(.4,.7)).clip(0,255).astype(np.uint8))
        return to_tensor(image), target


def audit(dataset):
    """Validate labels and hash image/label bytes without accessing test data."""
    rows, objects, empty = [], 0, 0
    for index, path in enumerate(dataset.paths):
        _, target = dataset[index]
        objects += len(target['boxes'])
        empty += int(len(target['boxes']) == 0)
        for file in (path, dataset.label_path(path)):
            rows.append({'path': file.relative_to(dataset.root).as_posix(),
                         'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    return {'images': len(dataset), 'objects': objects, 'empty_images': empty,
            'files': rows}


def collate(batch):
    return tuple(zip(*batch))


def build_model(config, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None, min_size=config.image_size,
        max_size=config.max_size, box_detections_per_img=config.max_detections,
        box_score_thresh=config.score_threshold, box_nms_thresh=config.nms_iou,
        trainable_backbone_layers=3 if pretrained else None,
    )
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    model.rpn.anchor_generator = AnchorGenerator(tuple((x,) for x in config.anchor_sizes), ((0.5, 1.0, 2.0),)*5)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def train(config, data_root=None, output='outputs/faster_rcnn_A', device=None,
          smoke=False):
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    if not 0 <= config.horizontal_flip_probability <= 1:
        raise ValueError('Flip probability must be between 0 and 1')
    if not 0 <= config.photometric_probability <= 1:
        raise ValueError('Photometric probability must be between 0 and 1')
    if not 0 <= config.blur_probability <= 1:
        raise ValueError('Blur probability must be between 0 and 1')
    if (len(config.blur_radius_range) != 2 or not np.isfinite(config.blur_radius_range).all()
            or not 0 <= config.blur_radius_range[0] <= config.blur_radius_range[1]):
        raise ValueError('Blur radius bounds must be finite, ordered and nonnegative')
    for bounds in (config.contrast_range, config.brightness_range):
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError('Augmentation bounds must be two ordered finite values')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    root = resolve_data(data_root)
    train_data, val_data = BirdDataset(root, 'train'), BirdDataset(root, 'val')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    info = {'config': asdict(config), 'dataset': 'liugalhigh/beach-bird/versions/3',
            'data_root': str(root.resolve()), 'local_root_version_verified': data_root is None and not os.environ.get('FASTER_RCNN_DATA_ROOT'),
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'device': str(device), 'device_name': torch.cuda.get_device_name(device)
            if device.type == 'cuda' else 'CPU',
            'weights': 'FasterRCNN_ResNet50_FPN_Weights.COCO_V1',
            'resize': f'aspect-preserving short edge {config.image_size}, max {config.max_size}; stride32 padding',
            'scheduler': 'SGD momentum .9; MultiStepLR epochs 10,13 gamma .1; AMP off; accumulation 1', 'smoke_only': smoke}
    # Explicit evaluation provenance for future runs; does not alter metrics.
    info['evaluation_protocol'] = {
        'ap_confidence_floor': config.score_threshold,
        'ap_evaluator': 'Ultralytics 8.4.208 exact upstream match_predictions and ap_per_class',
        'ap_ious': '0.50:0.05:0.95', 'ap_max_detections': config.max_detections,
        'pr_confidence': config.pr_confidence, 'pr_match_iou': 0.5,
        'pr_matching': 'Ultralytics match_predictions, use_scipy=False, rematched at conf >= .25',
        'pr_aggregation': 'global micro TP/FP/FN',
        'final_roi_nms_iou': config.nms_iou, 'rpn_nms_iou':0.7, 'rpn_pre_nms_train_test':[2000,1000], 'rpn_post_nms_train_test':[2000,1000], 'alignment':'same functions as teammate library; confirm joint re-evaluation before claiming strict cross-model alignment'}
    if config.photometric_probability:
        info['photometric_implementation'] = {
            'formula': 'round(clip(alpha * RGB_uint8 + beta, 0, 255))',
            'sampling': 'independent continuous uniform alpha and beta; shared trigger',
            'order': 'horizontal flip, then photometric, then internal model resize',
            'alignment': 'local convention based on teammate ranges; exact YOLO formula unconfirmed'}
    if config.blur_probability:
        info['blur_implementation'] = {
            'library': 'PIL.ImageFilter.GaussianBlur',
            'sampling': 'independent Bernoulli trigger; continuous uniform radius',
            'order': 'horizontal flip, photometric, blur at original resolution, model resize',
            'pillow_version': __import__('PIL').__version__}
        if config.photometric_probability:
            info['photometric_implementation']['order'] = info['blur_implementation']['order']
    save_json(output / 'config.json', info)
    manifests = {split: audit(ds) for split, ds in [('train', train_data), ('val', val_data)]}
    save_json(output / 'data_manifest.json', manifests)
    # Audit original data without consuming augmentation RNG; augment train only.
    train_data.horizontal_flip_probability = config.horizontal_flip_probability
    train_data.photometric_probability = config.photometric_probability
    train_data.contrast_range = config.contrast_range
    train_data.brightness_range = config.brightness_range
    train_data.dark_probability = config.dark_probability
    train_data.blur_probability = config.blur_probability
    train_data.blur_radius_range = config.blur_radius_range
    for split, expected in [('train', 1369), ('val', 291)]:
        actual = manifests[split]['images']
        print(f'{split}: {actual} images, {manifests[split]["objects"]} boxes')
        if actual != expected and not smoke:
            raise ValueError(f'{split}: expected {expected} images, found {actual}')
    if smoke:
        train_data.paths = train_data.paths[:2]
        val_data.paths = val_data.paths[:2]
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.workers, collate_fn=collate, generator=generator)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                            num_workers=config.workers, collate_fn=collate)
    model = build_model(config).to(device)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                  lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 13], gamma=0.1)
    history, best, training_seconds = [], -1.0, 0.0
    for epoch in range(1, (1 if smoke else config.epochs)+1):
        model.train()
        sync(device)
        start = time.perf_counter()
        loss_sum = 0.0
        components = {}
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch}'):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            for key,value in losses.items(): components[key]=components.get(key,0)+float(value.detach())*len(images)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss: {losses}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(images)
        sync(device)
        seconds = time.perf_counter()-start
        training_seconds += seconds
        metrics, predictions = evaluate(model, val_loader, device, config)
        epoch_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': loss_sum/len(train_data),
               'train_seconds': seconds, 'lr': epoch_lr, 'loss_components':{k:v/len(train_data) for k,v in components.items()}, **metrics}
        history.append(row)
        print(json.dumps(row))
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
                      'epoch': epoch, 'config': asdict(config), 'metrics': metrics}
        torch.save(checkpoint, output / 'last.pth')
        if metrics['map_50_95'] > best:
            best = metrics['map_50_95']
            torch.save(checkpoint, output / 'best.pth')
            save_json(output / 'best_val_predictions.json',
                      {'image_paths': [str(p.relative_to(root)) for p in val_data.paths],
                       'predictions': predictions})
        save_json(output / 'history.json', history)
    # Benchmark the selected checkpoint, not whichever epoch happened to finish last.
    selected = torch.load(output / 'best.pth', map_location=device, weights_only=True)
    model.load_state_dict(selected['model'])
    save_json(output / 'summary.json', {'best_val_map_50_95': best,
              'selected_epoch': selected['epoch'], 'training_seconds': training_seconds,
              'val_seconds_per_image_batch1': None,
              'timing_scope': 'transfer + internal preprocessing + inference + NMS; excludes decoding',
              'test_evaluated': False, 'smoke_only': smoke})
    return history


from .evaluation import evaluate
