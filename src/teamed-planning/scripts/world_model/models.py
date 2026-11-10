"""Model selection by architecture/config, following simulator.train_mappo."""

ARCHITECTURES = {'vtmp2': 'rssm_target_vtmp2', 'vtmp3': 'rssm_ab_vtmp3'}


def make_model(config):
    algorithm = config.get('algorithm')
    architecture = config.get('model_architecture', ARCHITECTURES.get(algorithm))
    if architecture not in ARCHITECTURES.values():
        raise ValueError('Unknown world-model architecture')
    if algorithm is not None and ARCHITECTURES.get(algorithm) != architecture:
        raise ValueError('Algorithm and world-model architecture disagree')
    options = {key: config.get(key, default) for key, default in (('hidden', 128), ('latent', 24))}
    if any(type(value) is not int or value < 1 for value in options.values()):
        raise ValueError('Model dimensions must be positive integers')
    if architecture == 'rssm_target_vtmp2':
        from .model import Vtmp2WorldModel
        return Vtmp2WorldModel(**options)
    from .ab_model import ABWorldModel
    return ABWorldModel(**options)
