"""Read seal images and label manifests from disk."""
import csv
import logging
from pathlib import Path

import cv2
import numpy as np

from seals.config import DIGIT_COUNT

LOGGER = logging.getLogger(__name__)


def read_gray(path):
    """Decode an image file to a grayscale uint8 array, or None if it cannot be read."""
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('decoder returned no image')
        return image
    except (OSError, ValueError, cv2.error) as error:
        LOGGER.error('Cannot decode %s: %s', path, error)
        return None


def read_manifest(path, image_directory):
    """Read a `filename;number` CSV into sorted (image_path, code) pairs, skipping bad rows."""
    records = []
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream, delimiter=';'):
                filename, number = row.get('filename') or '', row.get('number') or ''
                if Path(filename).name != filename or not filename.lower().endswith('.png'):
                    LOGGER.warning('Skipping invalid filename: %r', filename)
                    continue
                if len(number) != DIGIT_COUNT or not number.isascii() or not number.isdigit():
                    LOGGER.warning('Skipping invalid label for %s', filename)
                    continue
                records.append((image_directory / filename, number))
    except (OSError, UnicodeError, csv.Error) as error:
        LOGGER.error('Cannot read manifest %s: %s', path, error)
        return []
    return sorted(records, key=lambda record: record[0].name)
