"""v8 per-device relay cache for state evidence (deployable logic, no transport).

Every entry keeps its ORIGINAL measurement time and sequence number across
hops; forwarding only increments hop_count and replaces sender_id. Each
(origin, boot, seq) counts once, an older sequence never replaces a newer one,
and a restarted origin (new boot_id) is accepted only with a newer source time.
expires_at is set once by the origin and never refreshed by a relay.

The receiver decides what is usable at decision time: an entry exists only
after an actual envelope delivered it; unknown remote devices have no entry.
Transport (radio, ROS topics, simulator links) is the caller's job.
"""
import numpy as np

FRAME_ID = 'world'
TYPES = ('drone_state', 'vehicle_state')
# Design parameters (not validated on hardware): validity window per payload
# type, origin measurement sigma (m), and the default hop limit.
TTL_S = {'drone_state': 30., 'vehicle_state': 60.}
SIGMA_M = .5
HOP_LIMIT = 16


def make_entry(origin_id, boot_id, seq, source_stamp, payload_type, position, velocity=None,
               goal=None, sigma=SIGMA_M, mission_id=None, goal_epoch=0, hop_limit=HOP_LIMIT,
               ttl=None):
    """Origin-side record; hop_count 0 means 'still at its origin'."""
    if payload_type not in TYPES:
        raise ValueError(f'Unknown relay payload type {payload_type}')
    ttl = TTL_S[payload_type] if ttl is None else ttl
    return dict(origin_id=str(origin_id), boot_id=int(boot_id), seq=int(seq),
                source_stamp=None if source_stamp is None else float(source_stamp),
                sender_id=str(origin_id), receive_stamp=None, hop_count=0,
                payload_type=payload_type, frame_id=FRAME_ID,
                state=dict(position=tuple(float(v) for v in position),
                           velocity=None if velocity is None else tuple(float(v) for v in velocity),
                           goal=None if goal is None else tuple(float(v) for v in goal)),
                sigma=float(sigma), mission_id=mission_id, goal_epoch=int(goal_epoch),
                hop_limit=int(hop_limit),
                expires_at=None if source_stamp is None or ttl is None else float(source_stamp)+ttl)


def _copy(entry):
    return dict(entry, state=dict(entry['state']))


class RelayManager:
    """Knowledge cache G_info,i of one device: dedupe, ordering, TTL, forwarding."""

    def __init__(self, device_id, boot_id=0, capacity=16, max_entries=64, mission_id=None,
                 future_tolerance=.5, clock_offset=None, clock_bound=0., per_hop_delay_bound=2.,
                 speed_bound=6., ttl=None):
        self.device_id = str(device_id)
        self.boot_id = int(boot_id)
        self.capacity = int(capacity)
        self.max_entries = int(max_entries)
        self.mission_id = mission_id
        self.future_tolerance = float(future_tolerance)
        # Estimated clock offset per origin (s, origin clock minus local clock) and its bound.
        self.clock_offset = dict(clock_offset or {})
        self.clock_bound = float(clock_bound)
        self.per_hop_delay_bound = float(per_hop_delay_bound)
        self.speed_bound = float(speed_bound)
        self.ttl = dict(TTL_S, **(ttl or {}))
        self.seq = -1
        self.own = None
        self.cache = {}
        self.goal_epochs = {}
        self._sent = set()
        self.stats = dict.fromkeys(('received', 'accepted', 'duplicate', 'better_path', 'stale',
                                    'future', 'expired', 'rejected_frame', 'rejected_mission',
                                    'rejected_epoch', 'hop_limit', 'evicted'), 0)

    # ----- origin side -------------------------------------------------
    def originate(self, now, payload_type, position, velocity=None, goal=None, sigma=SIGMA_M,
                  goal_epoch=0):
        self.seq += 1
        self.own = make_entry(self.device_id, self.boot_id, self.seq, now, payload_type,
                              position, velocity, goal, sigma, self.mission_id, goal_epoch,
                              ttl=self.ttl[payload_type])
        return self.own

    # ----- receiver side -----------------------------------------------
    def age(self, entry, now):
        """Conservative age (s) and whether the source time was unknown."""
        if entry['source_stamp'] is None:
            return max(now-entry['receive_stamp'], 0.)+entry['hop_count']*self.per_hop_delay_bound, True
        offset = self.clock_offset.get(entry['origin_id'], 0.)
        return max(now-(entry['source_stamp']-offset), 0.), False

    def expired(self, entry, now):
        if entry['expires_at'] is not None:
            # expires_at is on the origin clock (offset = origin minus local).
            return now+self.clock_offset.get(entry['origin_id'], 0.) > entry['expires_at']+1e-9
        return self.age(entry, now)[0] > self.ttl[entry['payload_type']]

    def _newer(self, incoming, stored):
        if incoming['boot_id'] == stored['boot_id']:
            return incoming['seq'] > stored['seq']
        # A restart resets seq; only a provably newer measurement replaces it.
        if incoming['source_stamp'] is None or stored['source_stamp'] is None:
            return False
        return incoming['source_stamp'] > stored['source_stamp']

    def receive(self, envelope, now):
        """Accept one delivered envelope (a list of wire entries) at local time now."""
        for wire in envelope or ():
            self.stats['received'] += 1
            entry = _copy(wire)
            entry['hop_count'] = int(wire['hop_count'])+1
            entry['receive_stamp'] = float(now)
            origin = entry['origin_id']
            if origin == self.device_id:
                continue  # Own state is authoritative; echoes are ignored.
            if entry['frame_id'] != FRAME_ID or entry['payload_type'] not in TYPES:
                self.stats['rejected_frame'] += 1
                continue
            if self.mission_id is not None and entry['mission_id'] not in (None, self.mission_id):
                self.stats['rejected_mission'] += 1
                continue
            if entry['hop_count'] > entry['hop_limit']:
                self.stats['hop_limit'] += 1
                continue
            if entry['source_stamp'] is not None:
                skew = entry['source_stamp']-self.clock_offset.get(origin, 0.)-now
                if skew > self.future_tolerance:
                    # Quarantined, never truncated into the freshest message.
                    self.stats['future'] += 1
                    continue
            if self.expired(entry, now):
                self.stats['expired'] += 1
                continue
            if entry['payload_type'] == 'vehicle_state':
                if entry['goal_epoch'] < self.goal_epochs.get(origin, -1):
                    self.stats['rejected_epoch'] += 1
                    continue
                self.goal_epochs[origin] = entry['goal_epoch']
            stored = self.cache.get(origin)
            if stored is not None and (stored['boot_id'], stored['seq']) == (entry['boot_id'], entry['seq']):
                # Same measurement via another path: one piece of evidence. Keep
                # the shorter path's transport metadata, never a new source time.
                self.stats['duplicate'] += 1
                if entry['hop_count'] < stored['hop_count']:
                    stored.update(hop_count=entry['hop_count'], sender_id=entry['sender_id'])
                    self.stats['better_path'] += 1
                continue
            if stored is not None and not self._newer(entry, stored):
                self.stats['stale'] += 1
                continue
            self.cache[origin] = entry
            self.stats['accepted'] += 1
        self.prune(now)

    def prune(self, now):
        for origin in [o for o, e in self.cache.items() if self.expired(e, now)]:
            del self.cache[origin]
        if len(self.cache) > self.max_entries:
            # Oldest first; one vehicle (task) record per origin is kept longest.
            order = sorted(self.cache, key=lambda o: (self.cache[o]['payload_type'] == 'vehicle_state',
                                                       -self.age(self.cache[o], now)[0], o))
            for origin in order[:len(self.cache)-self.max_entries]:
                del self.cache[origin]
                self.stats['evicted'] += 1

    # ----- forwarding --------------------------------------------------
    def outgoing(self, now):
        """Envelope for one actual transmission; call only when it is sent."""
        self.prune(now)
        candidates = [e for e in self.cache.values() if e['hop_count'] < e['hop_limit']]
        # Within the budget: vehicle/task first, then entries not yet forwarded,
        # then the freshest. Own state always goes first.
        candidates.sort(key=lambda e: (e['payload_type'] != 'vehicle_state',
                                       (e['origin_id'], e['boot_id'], e['seq']) in self._sent,
                                       self.age(e, now)[0], e['hop_count'], e['origin_id']))
        chosen = ([self.own] if self.own is not None else [])+candidates
        chosen = chosen[:self.capacity]
        self._sent = {(e['origin_id'], e['boot_id'], e['seq']) for e in chosen}
        envelope = []
        for entry in chosen:
            wire = _copy(entry)
            wire.update(sender_id=self.device_id, receive_stamp=None)
            envelope.append(wire)
        return envelope

    def tokens(self, now):
        """Decision-time view of the cache; never contains this device itself."""
        self.prune(now)
        rows = []
        for entry in self.cache.values():
            age, unknown = self.age(entry, now)
            state = entry['state']
            rows.append(dict(origin_id=entry['origin_id'], payload_type=entry['payload_type'],
                             position=np.asarray(state['position'], float),
                             velocity=None if state['velocity'] is None else np.asarray(state['velocity'], float),
                             goal=None if state['goal'] is None else np.asarray(state['goal'], float),
                             age=float(age), hop=int(entry['hop_count']), time_unknown=unknown,
                             # Clock error turns into position uncertainty at bounded speed.
                             sigma=float(entry['sigma'])+self.speed_bound*self.clock_bound))
        rows.sort(key=lambda row: (row['payload_type'] != 'vehicle_state', row['age'], row['hop'],
                                   row['origin_id']))
        return rows
