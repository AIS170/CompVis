import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_laplace, maximum_filter

NON_MAX_SUPRESS_IOU = 0.3
SIGMAS = np.geomspace(8, 40, 6)
THRESH = 0.03


def non_max_supress(boxes, scores, iou_thresh):
    # Boxes processed from highest to lowest confidence
    order = np.argsort(-scores)
    keep = []

    while order.size > 0:
        idx = order[0]
        keep.append(idx)
        remainder = order[1:]

        # Calculate iou between selected and remaining boxes
        inter_area = np.clip(
            np.minimum(boxes[idx, 2], boxes[remainder, 2]) - np.maximum(boxes[idx, 0], boxes[remainder, 0]), 0, None
        ) * np.clip(
            np.minimum(boxes[idx, 3], boxes[remainder, 3]) - np.maximum(boxes[idx, 1], boxes[remainder, 1]), 0, None
        )

        box_area = (boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])
        remainder_areas = (boxes[remainder, 2] - boxes[remainder, 0]) * (boxes[remainder, 3] - boxes[remainder, 1])

        order = remainder[inter_area / (box_area + remainder_areas - inter_area + 1e-9) <= iou_thresh]

    return np.array(keep, dtype = int)


def detect_birds(img_path):
    img = np.asarray(Image.open(img_path).convert("L"), dtype=float) / 255.0
    img_height, img_width = img.shape

    # Calculate LoG for different scales
    log_responses = []
    for sigma in SIGMAS:
        response = np.abs(sigma**2 * gaussian_laplace(img, sigma))
        log_responses.append(response)

    log_stack = np.stack(log_responses)

    # Identifies local maxima above detection threshold
    sigma_idx, y_idx, x_idx = np.nonzero((maximum_filter(log_stack, size = (3, 15, 15)) == log_stack) & (log_stack > THRESH))

    if sigma_idx.size == 0:
        return np.zeros((0, 4)), np.zeros(0)

    scores = log_stack[sigma_idx, y_idx, x_idx]
    radius = np.sqrt(2) * SIGMAS[sigma_idx]

    # Convert the detected points into bounding boxes
    boxes = np.stack([
        np.clip(x_idx - radius, 0, img_width), np.clip(y_idx - radius, 0, img_height),
        np.clip(x_idx + radius, 0, img_width), np.clip(y_idx + radius, 0, img_height),
    ], axis = 1)

    keep_idx = non_max_supress(boxes, scores, NON_MAX_SUPRESS_IOU)

    return boxes[keep_idx], scores[keep_idx]
