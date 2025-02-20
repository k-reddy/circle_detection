from typing import NamedTuple, Optional, Tuple, Generator

import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import circle_perimeter_aa
from torch import Tensor

# import torch.multiprocessing as mp


class CircleParams(NamedTuple):
    row: int
    col: int
    radius: int


def draw_circle(img: np.ndarray, row: int, col: int, radius: int) -> np.ndarray:
    """
    Draw a circle in a numpy array, inplace.
    The center of the circle is at (row, col) and the radius is given by radius.
    The array is assumed to be square.
    Any pixels outside the array are ignored.
    Circle is white (1) on black (0) background, and is anti-aliased.
    """
    rr, cc, val = circle_perimeter_aa(row, col, radius)
    valid = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
    img[rr[valid], cc[valid]] = val[valid]
    return img


def noisy_circle(
    img_size: int,
    min_radius: float,
    max_radius: float,
    noise_level: float,
    crop: bool = False,
) -> Tuple[np.ndarray, CircleParams]:
    """
    Draw a circle in a numpy array, with normal noise.
    """

    # Create an empty image
    img = np.zeros((img_size, img_size))

    radius = np.random.randint(min_radius, max_radius)

    # x,y coordinates of the center of the circle
    row, col = np.random.randint(img_size, size=2)

    # Draw the circle inplace
    draw_circle(img, row, col, radius)

    if crop:
        # somewhat arbitrary numbers for crop size - just looking for it to not be too small or too big
        crop_size = img_size // (np.random.randint(8, 20))
        crop_row = np.random.randint(0, img_size - crop_size)
        crop_col = np.random.randint(0, img_size - crop_size)
        img[crop_row : crop_row + crop_size, crop_col : crop_col + crop_size] = 0

    added_noise = np.random.normal(0.5, noise_level, img.shape)
    img += added_noise

    return img, CircleParams(row, col, radius)


def show_circle(img: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title("Circle")
    plt.show()


def generate_examples(
    noise_level: float = 0.5,
    img_size: int = 50,
    min_radius: Optional[int] = None,
    max_radius: Optional[int] = None,
    dataset_path: str = "ds",
) -> Generator[Tuple[np.ndarray, CircleParams], None, None]:
    if not min_radius:
        min_radius = img_size // 10
    if not max_radius:
        max_radius = img_size // 2
    assert max_radius > min_radius, "max_radius must be greater than min_radius"
    assert img_size > max_radius, "size should be greater than max_radius"
    assert noise_level >= 0, "noise should be non-negative"

    params = (
        f"{noise_level=}, {img_size=}, {min_radius=}, {max_radius=}, {dataset_path=}"
    )
    print(f"Using parameters: {params}")
    while True:
        img, params = noisy_circle(
            img_size=img_size,
            min_radius=min_radius,
            max_radius=max_radius,
            noise_level=noise_level,
        )
        yield img, params


def count_ious_over_thresholds(predictions: Tensor, targets: Tensor) -> dict:
    """Counts number of predictions where IOU is over the threshold"""
    ious_over_thresholds = {0.1: 0, 0.25: 0, 0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0}

    for i in range(predictions.shape[0]):
        iou_score = iou(CircleParams(*targets[i]), CircleParams(*predictions[i]))
        for threshold in ious_over_thresholds.keys():
            if iou_score > threshold:
                ious_over_thresholds[threshold] += 1

    return ious_over_thresholds


def iou(a: CircleParams, b: CircleParams) -> float:
    """Calculate the intersection over union of two circles"""
    r1, r2 = a.radius, b.radius
    d = np.linalg.norm(np.array([a.row, a.col]) - np.array([b.row, b.col]))
    if d > r1 + r2:
        # If the distance between the centers is greater than the sum of the radii, then the circles don't intersect
        return 0.0
    if d <= abs(r1 - r2):
        # If the distance between the centers is less than the absolute difference of the radii, then one circle is
        # inside the other
        larger_r, smaller_r = max(r1, r2), min(r1, r2)
        return smaller_r**2 / larger_r**2
    r1_sq, r2_sq = r1**2, r2**2
    d1 = (r1_sq - r2_sq + d**2) / (2 * d)
    d2 = d - d1
    sector_area1 = r1_sq * np.arccos(d1 / r1)
    triangle_area1 = d1 * np.sqrt(r1_sq - d1**2)
    sector_area2 = r2_sq * np.arccos(d2 / r2)
    triangle_area2 = d2 * np.sqrt(r2_sq - d2**2)
    intersection = sector_area1 + sector_area2 - (triangle_area1 + triangle_area2)
    union = np.pi * (r1_sq + r2_sq) - intersection
    return intersection / union
