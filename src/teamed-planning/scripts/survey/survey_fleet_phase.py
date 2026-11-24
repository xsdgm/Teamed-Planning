"""Check native V9 local raster clearance after a single real ground movement."""
from itertools import product
import math

import numpy as np

from survey.survey_fleet_clearance import native_segment_clear


def _pixel_shifts(distance, resolution):
    """Possible changes of a static obstacle's integer centre on this axis."""
    if distance == 0.:
        return (0,)
    sign = 1 if distance > 0. else -1
    cells = abs(distance) / resolution
    low, high = math.floor(cells), math.ceil(cells)
    # Native circle centres are converted with int(), including centres outside
    # the image. If a centre crosses the left/bottom origin, truncation through
    # zero can skip a pixel shift compared with ordinary floor quantization.
    return tuple(sign * n for n in sorted({low, high, max(0, low - 1)}))


def endpoint_phase_clear(tile_yx, sensor_center_xy, sensor_resolution, car_xy, goal_xy, clearance_m=1.5):
    """Certify the move on the current tile and a disk at the next sensor centre.

    This bound uses the frozen simulator's static, globally XY-aligned pixel
    rasterization. Each translated object has an unknown subpixel phase. Under a
    shift d of the tile origin its raster translates by d - k * resolution,
    where k is one of the discrete pixel shifts below. Testing the next centre
    against all corresponding locations on the current tile covers their union.
    An unchanged axis receives no expansion. Rotation alone does not rotate the
    native ground sensor grid.
    """
    if not native_segment_clear(tile_yx, sensor_center_xy, sensor_resolution,
                                car_xy, goal_xy, clearance_m):
        return False
    start = np.asarray(car_xy, float)
    delta = np.asarray(goal_xy, float) - start
    if not delta.any():
        return True
    x = _pixel_shifts(delta[0], sensor_resolution)
    y = _pixel_shifts(delta[1], sensor_resolution)
    for shift in product(x, y):
        projected = start + sensor_resolution * np.asarray(shift)
        if not native_segment_clear(tile_yx, sensor_center_xy, sensor_resolution,
                                    projected, projected, clearance_m):
            return False
    return True
