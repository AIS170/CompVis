from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from PIL import Image

DATASET_PATH = Path(kagglehub.dataset_download("liugalhigh/beach-bird")) / "Beach Bird Detection Dataset v1.0"


def load_split(split):
    samples = []

    for img_path in sorted((DATASET_PATH / "images" / split).glob("*.jpg")):
        img_width, img_height = Image.open(img_path).size
        label_path = DATASET_PATH / "labels" / split / f"{img_path.stem}.txt"

        # Read yolo annotations and convert to float
        yolo_data = []
        for line in label_path.read_text().splitlines():
            if line.strip():
                values = list(map(float, line.split()))
                yolo_data.append(values)

        yolo = np.array(yolo_data, dtype = float).reshape(-1, 5)

        # Convert centre coords into pixel based corner coords
        centre_x, centre_y = yolo[:, 1] * img_width, yolo[:, 2] * img_height
        box_width, box_height = yolo[:, 3] * img_width, yolo[:, 4] * img_height

        samples.append({
            "path": img_path,
            "boxes": np.stack([centre_x - box_width / 2, centre_y - box_height / 2, centre_x + box_width / 2, centre_y + box_height / 2], axis = 1),
            "size": (img_width, img_height)
        })

    return samples


def show_plt(img_samples, predictions, num_imgs):
    _, axes = plt.subplots(1, num_imgs, figsize = (6 * num_imgs, 6), squeeze = False)

    for img_idx, (ax, sample) in enumerate(zip(axes[0], img_samples[:num_imgs])):
        ax.imshow(Image.open(sample["path"]))
        ax.axis("off")

        # Draw the ground truth bounding boxes
        for x_min, y_min, x_max, y_max in sample["boxes"]:
            ax.add_patch(
                patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill = False,
                    color = "yellow",
                )
            )

        # Draw the predicted bounding boxes
        if predictions is not None:
            for x_min, y_min, x_max, y_max in predictions[img_idx][0]:
                ax.add_patch(
                    patches.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        fill = False,
                        color = "magenta",
                    )
                )

        ax.set_title(f"{sample['path'].name}: {len(sample['boxes'])} birds")

    plt.show()


def calc_iou(ref_box, compare_boxes):
    inter_width = np.clip(
        np.minimum(ref_box[2], compare_boxes[:, 2]) - np.maximum(ref_box[0], compare_boxes[:, 0]), 0, None
    )
    inter_height = np.clip(
        np.minimum(ref_box[3], compare_boxes[:, 3]) - np.maximum(ref_box[1], compare_boxes[:, 1]), 0, None
    )

    # Calculate intersection area
    inter_area = inter_width * inter_height

    # Calculate box areas
    ref_area = (ref_box[2] - ref_box[0]) * (ref_box[3] - ref_box[1])
    compare_area = (compare_boxes[:, 2] - compare_boxes[:, 0]) * (compare_boxes[:, 3] - compare_boxes[:, 1])

    return inter_area / (ref_area + compare_area - inter_area + 1e-9)


def calc_avg_precision(predictions, GT_boxes, iou_thresh = 0.5):
    prediction_records = []

    # Collect all predictions with image index and confidence score
    for img_idx, (predicted_boxes, conf_scores) in enumerate(predictions):
        for predicted_box, conf in zip(predicted_boxes, conf_scores):
            prediction_records.append((conf, img_idx, predicted_box))

    prediction_records.sort(key = lambda record: -record[0])

    # Track the already matched ground truth boxes
    matched_GT = []
    for boxes in GT_boxes:
        matched_GT.append(np.zeros(len(boxes), dtype = bool))

    TP = np.zeros(len(prediction_records))
    FP = np.zeros(len(prediction_records))

    # Match each prediction to the GT box with highest IoU
    for prediction_idx, (_, img_idx, predicted_box) in enumerate(prediction_records):
        img_GT = GT_boxes[img_idx]
        if len(img_GT) == 0:
            FP[prediction_idx] = 1
            continue

        iou_scores = calc_iou(predicted_box, img_GT)
        best_match_idx = int(iou_scores.argmax())

        if iou_scores[best_match_idx] >= iou_thresh and not matched_GT[img_idx][best_match_idx]:
            TP[prediction_idx] = 1
            matched_GT[img_idx][best_match_idx] = True
        else:
            FP[prediction_idx] = 1

    # Calculate cumulative true positives
    cumulative_TP = np.cumsum(TP)

    total_GT = 0
    for boxes in GT_boxes:
        total_GT += len(boxes)

    # Calculate recall
    recall = cumulative_TP / max(total_GT, 1)

    # Calculate precision and build recall precision points
    recall_pts = np.concatenate(([0.0], recall, [1.0]))
    precision_pts = np.concatenate(([0.0], cumulative_TP / np.maximum(cumulative_TP + np.cumsum(FP), 1e-9), [0.0]))

    for point_idx in range(len(precision_pts) - 2, -1, -1):
        precision_pts[point_idx] = max(precision_pts[point_idx], precision_pts[point_idx + 1])

    recall_changes = np.where(recall_pts[1:] != recall_pts[:-1])[0]

    # Calculate area under precision recall curve
    return float(np.sum((recall_pts[recall_changes + 1] - recall_pts[recall_changes]) * precision_pts[recall_changes + 1]))
