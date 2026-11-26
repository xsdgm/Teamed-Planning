"""Native 2D circle scenes for the V9.1 large-obstacle prediction pilot.

A positive map adds a connected group of one to three overlapping large
circles. Negative maps retain the same distribution of small distractors.
The simulator consumes `circles`. Only the training labeler uses
`large_circles`; there are no semantic obstacle IDs in the received rasters.
Coordinates are in metres.
"""

import numpy as np


def make_case(seed, has_large=None):
    """Generate a reproducible map; contiguous even-sized seed ranges are balanced."""
    seed = int(seed)
    if has_large is None:
        has_large = seed % 2 == 0
    rng = np.random.default_rng(seed)
    start = rng.uniform(300., 700., 2)
    heading = rng.uniform(-np.pi, np.pi)
    forward = np.array([np.cos(heading), np.sin(heading)])
    lateral = np.array([-forward[1], forward[0]])
    goal = start + rng.uniform(115., 180.) * forward

    # Put the front of the block within reach of the first aerial surveys,
    # while varying both its size and its sides across maps.
    radius = rng.uniform(13., 29.)
    center = start + (radius + rng.uniform(13., 34.)) * forward
    center += rng.uniform(-.90, .90) * radius * lateral
    group = [(center, radius)]
    for _ in range(int(rng.integers(0, 3))):
        angle = rng.uniform(-np.pi, np.pi)
        direction = np.cos(angle) * forward + np.sin(angle) * lateral
        lobe = center + rng.uniform(.55, .95) * radius * direction
        lobe_radius = rng.uniform(.55, .95) * radius
        if min(np.linalg.norm(lobe - start), np.linalg.norm(lobe - goal)) > lobe_radius + 9.:
            group.append((lobe, lobe_radius))

    # Include small near-route obstacles even in clear-of-large scenes.
    # Draw all background geometry independently of the class.
    clutter = [(center, rng.uniform(2.5, 5.5))]
    for _ in range(int(rng.integers(5, 13))):
        xy = start + rng.uniform(0., 108.) * forward + rng.uniform(-65., 65.) * lateral
        r = rng.uniform(2., 5.5)
        if min(np.linalg.norm(xy - start), np.linalg.norm(xy - goal)) <= r + 7.:
            continue
        if any(np.linalg.norm(xy - c) <= r + other_r + 3. for c, other_r in clutter + group):
            continue
        clutter.append((xy, r))

    large = [(c.tolist(), float(r)) for c, r in group] if has_large else []
    circles = [(c.tolist(), float(r)) for c, r in clutter] + large
    return dict(seed=seed, start=start.tolist(), goal=goal.tolist(), attempts=1,
                circles=circles, large_circles=large)


def rasterize_large(scene, world_xy):
    """Return the teacher's large-obstacle mask at the specified world cell centres."""
    xy = np.asarray(world_xy, dtype=float)
    if xy.ndim < 1 or xy.shape[-1] != 2:
        raise ValueError('Expected (..., 2) world XY coordinates')
    mask = np.zeros(xy.shape[:-1], dtype=bool)
    for center, radius in scene['large_circles']:
        mask |= np.square(xy - center).sum(axis=-1) <= radius**2
    return mask
