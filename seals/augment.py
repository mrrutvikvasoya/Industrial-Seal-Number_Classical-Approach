"""Label-preserving augmentation of normalized crops, applied to training data only."""
import cv2
import numpy as np

# Kept below the point where a rotated/scaled digit could read as another.
MAX_ANGLE = 8.0
SCALE_RANGE = (0.9, 1.1)
# Small shifts within the crop padding.
MAX_SHIFT_FRACTION = 0.06
# Blur half the time; sigma capped so thin strokes survive.
BLUR_PROBABILITY = 0.5
BLUR_SIGMA_RANGE = (0.4, 1.5)
# Darken one side half the time with a linear ramp, mimicking uneven lighting.
SHADOW_PROBABILITY = 0.5
SHADOW_STRENGTH_RANGE = (0.15, 0.45)
# Illumination variation on the 0-255 scale.
GAMMA_RANGE = (0.7, 1.4)


def apply_shadow(values, generator):
    """Multiply the crop by a linear brightness ramp in a random direction."""
    side = values.shape[0]
    angle = generator.uniform(0, 2 * np.pi)
    rows, columns = np.mgrid[0:side, 0:side].astype(np.float32)
    projection = np.cos(angle) * columns + np.sin(angle) * rows
    projection = (projection - projection.min()) / (np.ptp(projection) + 1e-6)
    ramp = 1.0 - generator.uniform(*SHADOW_STRENGTH_RANGE) * projection
    return values * ramp


def augment_crop(crop, generator):
    """Return one randomly transformed copy: geometry, blur, shadow, then gamma."""
    side = crop.shape[0]
    matrix = cv2.getRotationMatrix2D((side / 2, side / 2),
                                     generator.uniform(-MAX_ANGLE, MAX_ANGLE),
                                     generator.uniform(*SCALE_RANGE))
    matrix[:, 2] += generator.uniform(-MAX_SHIFT_FRACTION, MAX_SHIFT_FRACTION, 2) * side
    warped = cv2.warpAffine(crop, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
    if generator.random() < BLUR_PROBABILITY:
        warped = cv2.GaussianBlur(warped, (0, 0), generator.uniform(*BLUR_SIGMA_RANGE))
    values = warped.astype(np.float32)
    if generator.random() < SHADOW_PROBABILITY:
        values = apply_shadow(values, generator)
    gamma = generator.uniform(*GAMMA_RANGE)
    return np.rint((np.clip(values, 0, 255) / 255.0) ** gamma * 255).astype(np.uint8)
