"""Recognize a localized seal into a seven-digit string."""
import cv2
import numpy as np

from seals.config import DIGIT_COUNT
from seals.crops import crop_digit, normalize_digit
from seals.features import features
from seals.localize import find_digits

# The competition forbids an empty field; emitted only when no row is localized.
FALLBACK_NUMBER = '1580000'


def load_recognizer(path):
    """Load the persisted StandardScaler -> SVC pipeline from disk."""
    import joblib
    return joblib.load(path)


def digit_crops(gray, localization):
    """Return the normalized 32x32 crop for each localized digit box."""
    return [normalize_digit(crop_digit(gray, box, localization.angle))
            for box in localization.digit_boxes]


def recognize_crops(classifier, crops):
    """Classify each crop and join the seven digits into a number string."""
    feature_matrix = np.stack([features(crop) for crop in crops])
    digits = classifier.decision_function(feature_matrix).argmax(axis=1)
    return ''.join(str(int(digit)) for digit in digits)


def read_number(classifier, gray):
    """Localize then recognize; return the number string, or None if no row is found."""
    localization = find_digits(gray)
    if localization is None or len(localization.digit_boxes) != DIGIT_COUNT:
        return None
    return recognize_crops(classifier, digit_crops(gray, localization))


def predict_with_status(classifier, gray):
    """Recognize with a 180-degree retry and a fallback; return (number, status)."""
    number = read_number(classifier, gray)
    if number is not None:
        return number, 'localized'
    number = read_number(classifier, cv2.rotate(gray, cv2.ROTATE_180))
    if number is not None:
        return number, 'rotated'
    return FALLBACK_NUMBER, 'fallback'


def predict_number(classifier, gray):
    """Return the recognized seven-digit string for one seal image."""
    return predict_with_status(classifier, gray)[0]
