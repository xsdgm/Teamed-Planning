"""Audit learned gain on held-out GOOSE data without a navigation simulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'world_model'

import numpy as np
import torch
from torch.utils.data import DataLoader

from .frontier_data import FrontierPatches, load_regional_manifest
from .frontier_model import load_checkpoint
from .frontier_training import evaluate
from .frontier_selective import collect, rescale, selective_metrics
from .semantic_data import atomic_json, sha256


def assessment(checkpoint, data_manifest, calibration_path, output, *, device='cpu'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    model, config = load_checkpoint(checkpoint, device=device)
    calibration = json.loads(Path(calibration_path).read_text())
    if calibration['checkpoint_sha256'] != sha256(checkpoint):
        raise ValueError('Calibration/checkpoint identity mismatch')
    prior = np.array(config['class_prior'], np.float32)
    groups, _ = load_regional_manifest(data_manifest, cell_size=config['cell_size_m'], minimum_support=config['minimum_support'])

    def assess(scenes):
        dataset = FrontierPatches(scenes, prior=prior, side=config['side'], validation=True,
            masks_per_patch=config['validation_masks'], frontier_m=config['frontier_m'], seed=20261115)
        metrics = evaluate(model, DataLoader(dataset,batch_size=32,num_workers=0),device,prior,baselines=True)
        p, y, weight = collect(model,dataset,device)
        metrics['selective'] = selective_metrics(rescale(p,calibration['temperature']),y,weight,calibration['threshold'],
                                                 calibration['validated_prediction_classes'])
        return metrics

    result = dict(format='vtmp2-goose-assessment-v1', checkpoint_sha256=sha256(checkpoint),
        source_manifest_sha256=sha256(data_manifest), best_epoch=config['epoch'],
        training_validation=config['validation'], training_baselines=config['baselines'],
        held_out_test=assess(groups['test']), simulation_used=False,
        graph_navigation_validated=False, deployment_ready=False, execution_policy='existing_vtmp')
    validation = config['validation']['model']
    baseline = config['baselines']['structure_extension']
    prior_baseline = config['baselines']['training_prior']
    coverage = json.loads((Path(data_manifest).parent/'audit.json').read_text())
    predicted = np.array(validation['prediction_counts']); targets=np.array(validation['target_counts'])
    collapse = bool(predicted.max()/max(1,predicted.sum())>.95 and targets.max()/max(1,targets.sum())<.85)
    result['gates'] = dict(
        all_classes_have_minimum_coverage=coverage['all_classes_coverage_gate_passed'],
        validation_beats_structure_by_two_percent=validation['balanced_brier'] < .98*baseline['balanced_brier'],
        validation_beats_prior_by_two_percent=validation['balanced_brier'] < .98*prior_baseline['balanced_brier'],
        no_majority_class_collapse=not collapse,
        selective_test_operating_point=calibration['test_meets_operating_point'],
        measured_navigation_graph_benefit=False)
    result['unvalidated_classes'] = [name for name, value in coverage['class_readiness'].items()
                                    if not value['minimum_coverage_gate_passed']]
    result['navigation_requirement'] = ('Compare delivered observed graphs at identical current vehicle/final-goal endpoints; '
        'require verified endpoint attachments and edges. Coverage/entropy/class scores cannot substitute for this test.')
    atomic_json(output,result)
    print(json.dumps({k:result[k] for k in ('best_epoch','gates','unvalidated_classes','deployment_ready','execution_policy')}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-manifest',type=Path,required=True)
    parser.add_argument('--calibration-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    assessment(**vars(parser.parse_args()))


if __name__=='__main__':main()
