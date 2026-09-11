"""Cut and normalize a single digit; used identically for training and inference."""
import cv2
import numpy as np

# A small margin keeps full strokes after tight localization boxes.
CROP_MARGIN = 0.15
# Deskew only past this tilt; smaller angles are not worth the interpolation.
DESKEW_ANGLE = 1.0
# The recognizer input side; fixed so features have a constant length.
NORMALIZED_SIZE = 32
# Ignore the darkest/brightest 2 percent when stretching contrast.
STRETCH_PERCENTILES = (2, 98)


def crop_digit(gray, box, angle, margin=CROP_MARGIN):
    """Cut a padded patch around one box and deskew it by the row angle."""
    x, y, width, height = box
    pad = int(margin * height)
    # cut that padded rectangle out of the original grayscale image
    patch = gray[max(0, y - pad):y + height + pad, max(0, x - pad):x + width + pad]
    if patch.size == 0 or abs(angle) <= DESKEW_ANGLE:
        return patch
    center = (patch.shape[1] / 2, patch.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(patch, matrix, (patch.shape[1], patch.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def normalize_digit(patch, size=NORMALIZED_SIZE):
    """Return a size x size crop of bright ink on a black background."""
    # Make ink bright on black
    threshold, _ = cv2.threshold(cv2.GaussianBlur(patch, (3, 3), 0), 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if (patch > threshold).mean() > 0.5:
        patch = 255 - patch
    # Stretch the contrast
    low, high = np.percentile(patch, STRETCH_PERCENTILES)
    stretched = np.clip((patch.astype(np.float32) - low) * 255.0 / max(high - low, 1), 0, 255)
    # Resize, keeping the shape
    scale = size / max(stretched.shape)
    resized = cv2.resize(stretched.astype(np.uint8),
                         (max(1, round(stretched.shape[1] * scale)),
                          max(1, round(stretched.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    # Center the resized patch in a square canvas
    canvas = np.zeros((size, size), np.uint8)
    top, left = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas
