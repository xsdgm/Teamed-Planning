"""Multiscale semantic completion trained on real measured terrain columns.

This model has no simulated dynamics, control/action head or ground-truth map
input at deployment. A consumer must separately establish traversability.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_data import CHANNELS, CLASSES, CLASS_NAMES, NATIVE_IDS

ARCHITECTURE = 'semantic_completion_vtmp2'
CHECKPOINT_FORMAT = 'vtmp2-semantic-completion-v1'
SCHEMA_VERSION = 5


def block(inputs, outputs, dilation=1):
    groups = min(8, outputs)
    while outputs % groups:
        groups -= 1
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=dilation, dilation=dilation),
        nn.GroupNorm(groups, outputs), nn.SiLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(groups, outputs), nn.SiLU())


class _SemanticUNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.encoders = nn.ModuleList([block(CHANNELS, hidden), block(hidden, hidden*2),
            block(hidden*2, hidden*4), block(hidden*4, hidden*8, dilation=2)])
        self.decoders = nn.ModuleList([block(hidden*12, hidden*4), block(hidden*6, hidden*2),
                                      block(hidden*3, hidden)])
        self.output = nn.Conv2d(hidden, CLASSES, 1)

    def forward(self, maps):
        levels = []
        value = maps
        for index, encoder in enumerate(self.encoders):
            value = encoder(value if index == 0 else F.avg_pool2d(value, 2))
            levels.append(value)
        for decoder, skip in zip(self.decoders, reversed(levels[:-1])):
            value = decoder(torch.cat((F.interpolate(value, size=skip.shape[-2:], mode='bilinear',
                                                    align_corners=False), skip), 1))
        return self.output(value)


class SemanticCompletionModel(nn.Module):
    model_architecture = ARCHITECTURE
    algorithm = 'vtmp2'

    def __init__(self, hidden=24, members=3):
        super().__init__()
        if type(hidden) is not int or not 4 <= hidden <= 64 or type(members) is not int or not 1 <= members <= 5:
            raise ValueError('Invalid semantic model dimensions')
        self.hidden, self.members = hidden, members
        self.predictors = nn.ModuleList(_SemanticUNet(hidden) for _ in range(members))

    def configuration(self):
        return dict(hidden=self.hidden, members=self.members)

    def forward(self, maps):
        if maps.ndim != 4 or maps.shape[1] != CHANNELS or min(maps.shape[-2:]) < 16:
            raise ValueError('Expected a batch of semantic observations')
        # Defensive masking also protects callers bypassing the public encoder.
        observed = maps[:, CLASSES:CLASSES+1]
        safe = maps.clone()
        safe[:, :CLASSES] = torch.where(observed.bool(), safe[:, :CLASSES], 0.)
        height_known = safe[:, CLASSES+2:CLASSES+3]*observed
        safe[:, CLASSES+1:CLASSES+2] = torch.where(height_known.bool(), safe[:, CLASSES+1:CLASSES+2], 0.)
        safe[:, CLASSES+2:CLASSES+3] = height_known
        return torch.stack([member(safe) for member in self.predictors])

    @torch.no_grad()
    def predict(self, maps):
        members = self(maps).softmax(2)
        observed = maps[None, :, CLASSES:CLASSES+1].bool()
        members = torch.where(observed, maps[None, :, :CLASSES], members)
        probability = members.mean(0)
        entropy = -(probability*probability.clamp_min(1e-8).log()).sum(1)
        unknown = 1-maps[:, CLASSES]
        return dict(semantics=probability, entropy=entropy*unknown,
                    disagreement=members.std(0, unbiased=False).mean(1)*unknown)


def completion_loss(logits, target, hidden_valid, class_weights):
    """Only measured, withheld targets supervise the model; unseen source is ignored."""
    weighted = target*class_weights[None, :, None, None]
    # Class-balanced supervision otherwise learns q(class) proportional to
    # weight * p(class). Apply the same offset to the training logits so raw
    # deployment softmax still represents p, not that reweighted distribution.
    adjusted = logits+class_weights.clamp_min(1e-8).log()[None, None, :, None, None]
    ce = -(adjusted.log_softmax(2)*weighted[None]).sum(2)
    denominator = (weighted.sum(1)*hidden_valid).sum().clamp_min(1e-8)
    return (ce*hidden_valid[None]).sum()/denominator/logits.shape[0]


def load_checkpoint(path, *, device='cpu'):
    document = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (document.get('format') != CHECKPOINT_FORMAT or document.get('schema_version') != SCHEMA_VERSION
            or document.get('model_architecture') != ARCHITECTURE
            or document.get('classes') != list(CLASS_NAMES) or document.get('native_ids') != list(NATIVE_IDS)
            or document.get('input_channels') != CHANNELS):
        raise ValueError('Expected a schema-5 semantic-completion checkpoint')
    if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in document['model'].values()):
        raise ValueError('Invalid semantic checkpoint weights')
    model = SemanticCompletionModel(**document['architecture'])
    model.load_state_dict(document['model'], strict=True)
    return model.to(device).eval().requires_grad_(False), document
