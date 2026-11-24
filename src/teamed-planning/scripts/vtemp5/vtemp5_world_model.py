"""Learn the shape and traversability of one reported local obstacle in metres.

Inference requires only NumPy. Regressions predict the lateral entrance offset and
thickness. The third output is the probability of this side being traversable.
The detector, survey scheduler, and side lock are outside this model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FEATURES = ('reported_river', 'side', 'front_visible_width_m',
            'front_slope', 'surveyed_side_lower_bound_m', 'edge_observed',
            'measured_gate_offset_m', 'measured_thickness_m', 'side_verdict')
OUTPUTS = ('gate_offset_abs_m', 'obstacle_thickness_m', 'open_probability')
INPUT_SCALE = np.array([1., 1., 100., .12, 100., 1., 140., 75., 1.], dtype=np.float32)
OUTPUT_CENTER = np.array([135., 75.], dtype=np.float32)
OUTPUT_SCALE = np.array([40., 25.], dtype=np.float32)
FORMAT = 'vtemp5_local_obstacle_mlp_v1'


def fuse_measurements(predictions, features):
    """A delivered measurement replaces a prediction for the same quantity."""
    pred = np.array(predictions, copy=True)
    pred[..., 0] = np.maximum(0., pred[..., 0])
    pred[..., 1] = np.maximum(0., pred[..., 1])
    pred[..., 0] = np.where(features[..., 5] > 0., features[..., 6], pred[..., 0])
    pred[..., 1] = np.where(features[..., 7] > 0., features[..., 7], pred[..., 1])
    verdict = features[..., 8]
    pred[..., 2] = np.where(verdict == 0, pred[..., 2], (verdict > 0).astype(float))
    return pred


class LocalWorldModel:
    def __init__(self, w1, b1, w2, b2, metadata):
        self.w1, self.b1 = np.asarray(w1, dtype=np.float32), np.asarray(b1, dtype=np.float32)
        self.w2, self.b2 = np.asarray(w2, dtype=np.float32), np.asarray(b2, dtype=np.float32)
        self.metadata = dict(metadata)
        hidden = len(self.b1)
        if (self.w1.shape != (hidden, len(FEATURES)) or self.w2.shape != (3, hidden)
                or self.b2.shape != (3,) or self.metadata.get('format') != FORMAT):
            raise ValueError('unsupported local world-model format or dimensions')
        if not all(np.isfinite(x).all() for x in (self.w1, self.b1, self.w2, self.b2)):
            raise ValueError('world-model weights must be finite')

    def raw_prediction(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim not in (1, 2) or x.shape[-1] != len(FEATURES) or not np.all(np.isfinite(x)):
            raise ValueError('expected finite 9-feature observation per side')
        hidden = np.tanh((x / INPUT_SCALE) @ self.w1.T + self.b1)
        output = hidden @ self.w2.T + self.b2
        output[..., :2] = OUTPUT_CENTER + OUTPUT_SCALE * output[..., :2]
        output[..., 2] = 1. / (1. + np.exp(-np.clip(output[..., 2], -50., 50.)))
        return output

    def predict(self, features):
        features = np.asarray(features, dtype=np.float32)
        return fuse_measurements(self.raw_prediction(features), features)

    def predict_sides(self, features):
        """Predict both candidate sides in one forward pass at a vehicle review."""
        return self.predict(features)

    __call__ = predict

    def save(self, path):
        np.savez_compressed(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                            metadata_json=json.dumps(self.metadata, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(archive['w1'], archive['b1'], archive['w2'], archive['b2'],
                       json.loads(archive['metadata_json'].item()))
