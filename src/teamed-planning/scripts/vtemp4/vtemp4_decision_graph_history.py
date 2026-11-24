"""Causal, receiver-side history for the vehicle decision graph."""
from array import array
from bisect import bisect_left, bisect_right
from collections import defaultdict
from operator import itemgetter

import numpy as np

from vtemp4.vtemp4_unified_attention import GoalAwareTerrainSummaryEpisode


COMMAND_FEATURES = 11
DRIVE_FEATURES = 10
RECENT_COMMANDS = 8
NEARBY_COMMANDS = 8
VISIT_CELL_M = 50.
_ISSUED = itemgetter('issued_at_s')
_REPLANNED = itemgetter('time_s')


class FullCommandHistory:
    """Append measured world-frame data once; express selected records in the current vehicle frame."""

    def __init__(self):
        self._path = array('d')  # World X, Y and cumulative metres at each physics tick.
        self._commands = array('d')  # Issue time, origin XY, shared target XY.
        self._replans = array('d', [-1., np.nan, np.nan])  # Initial interval has no local target.

    def _record_movement(self, now, position, trajectory, commands):
        last = len(self._path) // 3 - 1
        if now < last or len(trajectory) < now:
            raise ValueError('Decision history requires a contiguous measured 1 Hz vehicle trajectory')
        if now == last:
            return
        start = max(0, last)
        # Inside the native post-fusion callback, step() has not appended pose(now) yet.
        poses = np.vstack((trajectory[start:now], position)) if start < now else position[None]
        metres = np.cumsum(np.linalg.norm(np.diff(poses, axis=0), axis=1))
        if last < 0:
            rows = np.column_stack((poses, np.r_[0., metres]))
        else:
            rows = np.column_stack((poses[1:], self._path[-1] + metres))
        self._path.frombytes(rows.astype(np.float64, copy=False).tobytes())
        return start, poses

    def _select(self, count, position):
        return np.arange(count)

    def observation(self, now, position, goal, radius, trajectory, commands, ground_events):
        position = np.asarray(position, dtype=float)
        self._record_movement(now, position, trajectory, commands)
        # Only previously issued commands and replans influence this review.
        count = bisect_left(commands, now, key=_ISSUED)
        for command in commands[len(self._commands) // 5:count]:
            self._commands.extend((command['issued_at_s'], *command['origin_xy'], *command['target']))
        replan_end = bisect_left(ground_events, now, key=_REPLANNED)
        for event in ground_events[len(self._replans) // 3 - 1:replan_end]:
            xy = event['target']
            self._replans.extend((event['time_s'], *(xy if xy is not None else (np.nan, np.nan))))
        if count == 0:
            return dict(command_nodes=np.empty((0, COMMAND_FEATURES), np.float32),
                        command_predecessor=np.empty(0, np.int64),
                        drive_segments=np.empty((0, DRIVE_FEATURES), np.float32),
                        drive_command=np.empty(0, np.int64))

        path = np.frombuffer(self._path).reshape(-1, 3)
        issued = np.frombuffer(self._commands).reshape(-1, 5)
        replans = np.frombuffer(self._replans).reshape(-1, 3)
        times = issued[:, 0].astype(np.int64)
        selected = self._select(count, position)
        local = np.full(count, -1, np.int64)
        local[selected] = np.arange(len(selected))
        start = times[selected]
        end = np.r_[times[1:], now][selected]
        origin, target = issued[selected, 1:3], issued[selected, 3:5]
        arrival = path[end, :2]
        target_gain = np.linalg.norm(target - origin, axis=1) - np.linalg.norm(target - arrival, axis=1)
        goal_gain = np.linalg.norm(goal - origin, axis=1) - np.linalg.norm(goal - arrival, axis=1)
        command_rows = np.column_stack(((origin-position)/radius, (target-position)/radius,
                                        (arrival-position)/radius, np.log1p(now-start), np.log1p(end-start),
                                        target_gain/radius, goal_gain/radius, selected == count-1))
        predecessors = np.r_[-1, local[:-1]][selected]

        # Each half-open interval has one shared command and one then-effective
        # local target. Union includes upper issues even when the local target persists.
        split = np.union1d(times, replans[1:, 0].astype(np.int64))
        split = split[split >= times[0]]
        until = np.r_[split[1:], now]
        owners = local[np.searchsorted(times, split, side='right') - 1]
        keep = owners >= 0
        start, end = split[keep], until[keep]
        target = replans[np.searchsorted(replans[:, 0], start, side='right')-1, 1:3]
        valid = np.isfinite(target).all(axis=1)
        target_offset = np.where(valid[:, None], (target-position)/radius, 0.)
        drive_rows = np.column_stack(((path[start, :2]-position)/radius, (path[end, :2]-position)/radius,
                                      target_offset, valid, np.log1p(now-start), np.log1p(end-start),
                                      (path[end, 2]-path[start, 2])/radius))
        return dict(command_nodes=command_rows.astype(np.float32), command_predecessor=predecessors,
                    drive_segments=drive_rows.astype(np.float32), drive_command=owners[keep])


class CommandHistory(FullCommandHistory):
    """Original selection: recent commands and past visits near the current location."""

    def __init__(self):
        super().__init__()
        self.visits = defaultdict(list)

    def _record_movement(self, now, position, trajectory, commands):
        update = super()._record_movement(now, position, trajectory, commands)
        if update is None:
            return
        start, poses = update
        cells = np.floor(poses / VISIT_CELL_M).astype(int)
        owner = bisect_right(commands, start, key=_ISSUED) - 1
        for offset in range(now - start):
            while owner + 1 < len(commands) and _ISSUED(commands[owner + 1]) <= start + offset:
                owner += 1
            if owner >= 0:
                for cell in cells[offset:offset + 2]:
                    history = self.visits[tuple(cell)]
                    if not history or history[-1] != owner:
                        history.append(owner)

    def _select(self, count, position):
        recent_start = max(0, count - RECENT_COMMANDS)
        centre = np.floor(position / VISIT_CELL_M).astype(int)
        nearby = {}
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                entries = self.visits.get((centre[0] + dx, centre[1] + dy), ())
                stop = bisect_left(entries, recent_start)
                for owner in entries[max(0, stop - NEARBY_COMMANDS):stop]:
                    distance = dx * dx + dy * dy
                    nearby[owner] = min(nearby.get(owner, distance), distance)
        previous = sorted(nearby, key=lambda i: (nearby[i], -i))[:NEARBY_COMMANDS]
        return np.asarray(sorted(previous) + list(range(recent_start, count)), np.int64)


class DecisionGraphHistoryEpisode(GoalAwareTerrainSummaryEpisode):
    command_history_type = CommandHistory

    def __init__(self, *args, **kwargs):
        self.command_history = self.command_history_type()
        if kwargs.get('ground_control', 'vtemp4') != 'vtemp4':
            self._native_drive_events = []
        super().__init__(*args, **kwargs)

    def _decision_history_observation(self):
        local_events = self.ground_events if self._native_drive_events is None else self._native_drive_events
        return self.command_history.observation(self.clock_s, self.position(), self.goal, self.lookahead_m,
                                                self.trajectory, self.requests, local_events)
