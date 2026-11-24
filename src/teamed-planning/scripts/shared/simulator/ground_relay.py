"""Receiver-local UGV state evidence from the frozen V9 UAV wire envelopes."""
from __future__ import annotations

import numpy as np


class GroundRelayObservation:
    """Observe native UAV sends and simulate only their extra final hop to the UGV.

    The original V9 channel owns all incoming UAV traffic. The observer never
    calls a UAV's outgoing/cache API itself or sends anything back to a UAV.
    """

    def __init__(self, experiment, *, seed):
        self._experiment = experiment
        self._simulation = experiment.env
        self._channel = experiment.relay_channel
        if self._channel is None or experiment.ground_relay is None:
            raise ValueError("Ground relay observations require the native V9 relay channel")
        self._ground = experiment.ground
        self._drone_names = {rid: controller.drone_id
                             for rid, controller in zip(experiment.ids, experiment.controllers)}
        # The archived transport already draws a private RNG for delay and loss.
        # An independent derived seed prevents extra UGV sends shifting V9 draws.
        self._uplink = type(self._channel)(
            seed=int(seed) ^ 0x554756, loss=self._channel.loss,
            delay_jitter=self._channel.delay_jitter, period=self._channel.period,
        )
        self._receiver = type(experiment.ground_relay)(experiment.ground_controller.robot_id)
        self._delivered_frame = int(self._simulation.step_count) - 1
        self._original_transmit = self._channel.transmit
        self._previous_binding = vars(self._channel).get("transmit")
        self._channel.transmit = self._observe_transmit

    def _ground_neighbors(self):
        neighbors = self._experiment.obs[self._ground]["neighbors"]
        return {self._drone_names[row["id"]]: row for row in neighbors
                if row["type"] == "drone" and row["id"] in self._drone_names}

    def _observe_transmit(self, envelope, receivers, frame):
        result = self._original_transmit(envelope, receivers, frame)
        # Native RelayManager.outgoing() lists the sender's own entry first.
        # Ignore the ground vehicle's own downlink and sends outside its range.
        if not envelope:
            return result
        sender = envelope[0]["sender_id"]
        if sender in self._ground_neighbors():
            state_packets = [entry for entry in envelope if entry["payload_type"] == "drone_state"]
            if state_packets:
                self._uplink.transmit(state_packets, (self._ground,), frame)
        return result

    def observe(self):
        """Return arrived absolute relay tokens and independently sensed one-hop geometry."""
        frame = int(self._simulation.step_count)
        timestamp = float(self._simulation.time)
        dt = float(self._simulation.config.dt)
        if frame < self._delivered_frame:
            raise RuntimeError("Ground relay observations belong to one mission")
        # Slow decisions can be many fast ticks apart. Deliver each intermediate
        # frame at its own time so cached records have their actual arrival age.
        for tick in range(self._delivered_frame + 1, frame + 1):
            arrival = timestamp - (frame - tick) * dt
            for envelope in self._uplink.deliver(self._ground, tick):
                self._receiver.receive(envelope, arrival)
        self._delivered_frame = frame

        direct = []
        for origin_id, row in sorted(self._ground_neighbors().items()):
            velocity = row.get("relative_velocity")
            direct.append(dict(
                origin_id=origin_id,
                relative_position=np.asarray(row["relative_position"], dtype=float).copy(),
                relative_velocity=None if velocity is None else np.asarray(velocity, dtype=float).copy(),
                velocity_known=velocity is not None,
            ))
        return dict(frame=frame, timestamp=timestamp, relay_tokens=self._receiver.tokens(timestamp),
                    one_hop_neighbors=direct)

    def close(self):
        """Restore the channel method when this episode stops using the observer."""
        if self._channel.transmit == self._observe_transmit:
            if self._previous_binding is None:
                del self._channel.transmit
            else:
                self._channel.transmit = self._previous_binding
