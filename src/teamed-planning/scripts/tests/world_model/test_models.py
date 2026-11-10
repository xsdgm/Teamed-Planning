"""Version dispatch and state-dict compatibility across shared model modules."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from simulator.train_world_model import build_command
from world_model.models import make_model


class ModelDispatchTests(unittest.TestCase):
    def test_version_contracts_cannot_be_silently_mixed(self):
        with self.assertRaisesRegex(ValueError, 'disagree'):
            make_model(dict(algorithm='vtmp2', model_architecture='rssm_ab_vtmp3'))
        v2 = make_model(dict(algorithm='vtmp2', hidden=8, latent=4))
        v3 = make_model(dict(algorithm='vtmp3', hidden=8, latent=4))
        with self.assertRaises(RuntimeError):
            v3.load_state_dict(v2.state_dict(), strict=True)

    def test_legacy_parameter_names_load_and_keep_deterministic_predictions(self):
        for algorithm, state_head, features, actions in (
                ('vtmp2', 'features_head', 16, 3), ('vtmp3', 'state_head', 8, 4)):
            with self.subTest(algorithm=algorithm):
                torch.manual_seed(73)
                model = make_model(dict(algorithm=algorithm, hidden=8, latent=4))
                state = model.state_dict()
                self.assertIn(state_head+'.2.weight', state)
                self.assertFalse(any(key.startswith('base.') for key in state))
                loaded = make_model(dict(algorithm=algorithm, hidden=8, latent=4))
                loaded.load_state_dict(state, strict=True)
                maps, states, action = torch.zeros(1, 4, 32, 32), torch.zeros(1, features), torch.zeros(1, actions)
                h, z = model.initial(maps, states)
                lh, lz = loaded.initial(maps, states)
                actual = model.imagine(h, z, action, maps, states)
                restored = loaded.imagine(lh, lz, action, maps, states)
                for index in (0, 1, 3, 4, 5):
                    self.assertTrue(torch.equal(actual[index], restored[index]))

    def test_common_entry_resolves_config_and_keeps_cli_overrides_last(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(dict(algorithm='vtmp3', model_architecture='rssm_ab_vtmp3',
                parameters=dict(epochs=100, v11_config='v11.json'))))
            command = build_command(path, ['--epochs', '2'])
            self.assertEqual(command[-2:], ['--epochs', '2'])
            self.assertIn(str(Path(directory)/'v11.json'), command)
            self.assertEqual(Path(command[3]).name, 'ab_pipeline.py')
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                build_command(path, algorithm='vtmp2')


if __name__ == '__main__':
    unittest.main()
