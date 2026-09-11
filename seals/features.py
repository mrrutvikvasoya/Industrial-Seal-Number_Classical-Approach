"""Turn a normalized 32x32 digit crop into a fixed 343-length feature vector."""
import cv2
import numpy as np

# One HOG descriptor reused for every crop; it is stateless and C++ fast.
HOG = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
# 4x4 occupancy grid over the binarized crop.
ZONING_GRID = 4
# Contours smaller than this are threshold speckle, not real holes.
MIN_HOLE_AREA = 6.0


def hole_features(binary):
    """Count interior holes and their mean height, which separate 0/6/8/9."""
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = [cv2.boundingRect(contour) for index, contour in enumerate(contours) #
             if hierarchy is not None and hierarchy[0][index][3] != -1
             and cv2.contourArea(contour) > MIN_HOLE_AREA]
    if not holes:
        return [0.0, 0.5]
    center_y = np.mean([y + height / 2 for _, y, _, height in holes]) / binary.shape[0]
    return [float(len(holes)), float(center_y)]


def features(crop):
    """Concatenate HOG (324) + zoning (16) + holes (2) + aspect (1) into one vector."""
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # HOG descriptor is 324-length, but we flatten it to 1D for concatenation. edge/stroke
    hog = HOG.compute(crop).ravel()
    # where the ink sits (4×4 grid)
    zones = (binary.reshape(ZONING_GRID, crop.shape[0] // ZONING_GRID,
                            ZONING_GRID, crop.shape[1] // ZONING_GRID).mean(axis=(1, 3)) / 255).ravel()
    rows, columns = np.nonzero(binary)
    aspect = [(np.ptp(columns) + 1) / (np.ptp(rows) + 1)] if columns.size else [1.0]
    return np.concatenate([hog, zones, hole_features(binary), aspect]).astype(np.float32)
