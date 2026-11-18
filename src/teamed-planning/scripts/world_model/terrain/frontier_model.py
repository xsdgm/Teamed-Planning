"""Learn coarse structural extrapolation to aid independent aerial exploration.

The network learns bounded corrections to an observed-only continuation prior.
It outputs hypotheses and uncertainty, never a vehicle or aircraft command.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .frontier_data import CHANNELS, CLASSES, CLASS_NAMES, DISTANCE, OBSERVED, PRIOR
from .semantic_model import block, completion_loss

ARCHITECTURE = 'coarse_frontier_vtmp2'
CHECKPOINT_FORMAT = 'vtmp2-coarse-frontier-v1'
SCHEMA_VERSION = 6


class _FrontierMember(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.first = block(CHANNELS, hidden)
        self.second = block(hidden, hidden*2)
        self.third = block(hidden*2, hidden*4)
        self.up_second = block(hidden*6, hidden*2)
        self.up_first = block(hidden*3, hidden)
        self.output = nn.Conv2d(hidden, CLASSES, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, maps):
        first = self.first(maps)
        second = self.second(F.avg_pool2d(first, 2))
        third = self.third(F.avg_pool2d(second, 2))
        second = self.up_second(torch.cat((second, F.interpolate(third, size=second.shape[-2:],
                                                mode='bilinear', align_corners=False)), 1))
        first = self.up_first(torch.cat((first, F.interpolate(second, size=first.shape[-2:],
                                               mode='bilinear', align_corners=False)), 1))
        return 2.*torch.tanh(self.output(first)/2.)


class FrontierWorldModel(nn.Module):
    algorithm = 'vtmp2'
    model_architecture = ARCHITECTURE

    def __init__(self, hidden=24, members=3, cell_size=16., frontier_m=32.):
        super().__init__()
        if (type(hidden) is not int or not 4 <= hidden <= 64 or type(members) is not int
                or not 1 <= members <= 5 or not 0 < cell_size <= frontier_m <= 256):
            raise ValueError('Invalid coarse frontier model configuration')
        self.hidden, self.members = hidden, members
        self.cell_size, self.frontier_m = float(cell_size), float(frontier_m)
        self.predictors = nn.ModuleList(_FrontierMember(hidden) for _ in range(members))

    def configuration(self):
        return dict(hidden=self.hidden, members=self.members, cell_size=self.cell_size, frontier_m=self.frontier_m)

    def forward(self, maps):
        if maps.ndim != 4 or maps.shape[1] != CHANNELS or min(maps.shape[-2:]) < 8:
            raise ValueError('Expected coarse frontier observation batch')
        base = maps[:, PRIOR:PRIOR+CLASSES].clamp_min(1e-6).log()
        return torch.stack([base+member(maps) for member in self.predictors])

    @torch.no_grad()
    def predict(self, maps):
        probabilities = self(maps).softmax(2)
        known = maps[:, OBSERVED] > .5
        frontier = ~known & (maps[:, DISTANCE] > 0) & (maps[:, DISTANCE] <= 1.+1e-6)
        frontier &= known.any(-1).any(-1)[:, None, None]
        probability = probabilities.mean(0)
        probability = torch.where(known[:, None], maps[:, :CLASSES], probability)
        # Outside the current frontier we explicitly abstain. Predictions never
        # turn into observed evidence for another outward expansion.
        probability = torch.where((known | frontier)[:, None], probability, 0.)
        entropy = -(probability*probability.clamp_min(1e-8).log()).sum(1)
        return dict(semantics=probability, prediction_mask=frontier, observed_mask=known,
            entropy=entropy*frontier, disagreement=probabilities.std(0, unbiased=False).mean(1)*frontier)


def frontier_loss(logits, batch, weights):
    # Partial labelled support is confidence in the label, not online input
    # evidence about a hidden region. It only weights the offline loss.
    mask = batch['target_mask']*batch['target_support']
    return completion_loss(logits, batch['target'], mask, weights)


def load_checkpoint(path, *, device='cpu'):
    document = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (document.get('format') != CHECKPOINT_FORMAT or document.get('schema_version') != SCHEMA_VERSION
            or document.get('model_architecture') != ARCHITECTURE
            or document.get('classes') != list(CLASS_NAMES) or document.get('input_channels') != CHANNELS):
        raise ValueError('Expected a schema-6 coarse-frontier checkpoint')
    if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in document['model'].values()):
        raise ValueError('Invalid coarse-frontier checkpoint weights')
    model = FrontierWorldModel(**document['architecture'])
    model.load_state_dict(document['model'], strict=True)
    return model.to(device).eval().requires_grad_(False), document
