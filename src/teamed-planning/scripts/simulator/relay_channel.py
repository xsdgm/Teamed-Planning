"""Simulated v8 relay transport: one event queue per directed physical link.

A packet is transmitted only to devices within actual radio range at its send
step, arrives no earlier than the next decision step (so nothing propagates
recursively through the whole network within one step), and may be lost or
delayed (jitter also reorders packets). Every hop is a new transmission.
Receivers see packets only once arrival <= their decision step.
"""
import numpy as np


class RelayChannel:
    def __init__(self, seed, loss=0., delay_jitter=0, period=1):
        if not 0. <= loss < 1.:
            raise ValueError('relay loss must be in [0, 1)')
        if type(delay_jitter) is not int or delay_jitter < 0 or type(period) is not int or period < 1:
            raise ValueError('relay delay_jitter must be a nonnegative and period a positive integer')
        # A private stream: the relay draws never shift other simulator randomness.
        self.rng = np.random.default_rng([int(seed), 0x7638])
        self.loss, self.delay_jitter, self.period = float(loss), delay_jitter, period
        self.phase = {}
        self.queues = {}
        self.order = 0
        self.stats = dict(sent=0, lost=0, delivered=0)

    def due(self, sender, frame):
        if sender not in self.phase:
            self.phase[sender] = int(self.rng.integers(self.period))
        return (frame+self.phase[sender]) % self.period == 0

    def transmit(self, envelope, receivers, frame):
        for receiver in receivers:
            self.stats['sent'] += 1
            if self.rng.random() < self.loss:
                self.stats['lost'] += 1
                continue
            arrival = frame+1+int(self.rng.integers(self.delay_jitter+1))
            self.order += 1
            # Each receiver gets its own copy; later sender cache changes cannot leak in.
            packet = [dict(e, state=dict(e['state'])) for e in envelope]
            self.queues.setdefault(receiver, []).append((arrival, self.order, packet))

    def deliver(self, receiver, frame):
        queue = self.queues.get(receiver, [])
        ready = sorted(item for item in queue if item[0] <= frame)
        self.queues[receiver] = [item for item in queue if item[0] > frame]
        self.stats['delivered'] += len(ready)
        return [packet for _, _, packet in ready]
