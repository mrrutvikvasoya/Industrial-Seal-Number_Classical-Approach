"""Load harvested crop manifests into feature matrices for the SVM."""
import csv
import logging
from collections import defaultdict

import cv2
import numpy as np

from seals.augment import augment_crop
from seals.config import DEFAULT_SEED
from seals.features import features

LOGGER = logging.getLogger(__name__)


def read_pairs(manifest):
    """Read a harvest `path;label` CSV into (crop_path, digit) pairs."""
    pairs = []
    with manifest.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream, delimiter=';'):
            path, label = row.get('path') or '', row.get('label') or ''
            if label.isdigit() and len(label) == 1:
                pairs.append((manifest.parent / path, label))
    return pairs


def balance(pairs, cap, seed):
    """Keep at most `cap` crops per digit class to curb the strong digit imbalance."""
    if cap <= 0:
        return pairs
    generator = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for pair in pairs:
        by_label[pair[1]].append(pair)
    selected = []
    for label in sorted(by_label):
        group = by_label[label]
        if len(group) > cap:
            group = [group[index] for index in sorted(generator.choice(len(group), cap, replace=False))]
        selected.extend(group)
    return selected


def extract_features(pairs, copies=0, seed=DEFAULT_SEED):
    """Build the feature matrix and labels, adding `copies` augmented variants per crop."""
    matrix, labels = [], []
    for index, (path, label) in enumerate(pairs):
        crop = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if crop is None:
            LOGGER.warning('Cannot read crop %s', path)
            continue
        matrix.append(features(crop))
        labels.append(int(label))
        if copies:
            generator = np.random.default_rng(np.random.SeedSequence([seed, index]))
            for _ in range(copies):
                matrix.append(features(augment_crop(crop, generator)))
                labels.append(int(label))
    return np.stack(matrix), np.array(labels, dtype=int)
