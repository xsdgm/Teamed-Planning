"""Select a measured, executable local branch using the existing vehicle actor."""
import math

import numpy as np
import torch
from torch.distributions import Categorical

from v91_rejoin_guidance import RejoinGuidance, _bearing, _local_sensor, _ray_option, _unit, _wrap
from v91_route_evidence import append_route_evidence, route_candidates


def branch_policy(base_actor, observations):
    """The same conditional categorical distribution for collection and PPO updates."""
    angle_policy, values = base_actor(observations)
    counts = [len(obs['branch_angles']) for obs in observations]
    angles = np.zeros((len(counts), max(counts)), dtype=np.float64)
    prior = np.zeros_like(angles)
    for i, obs in enumerate(observations):
        if counts[i] == 0 or len(obs['branch_prior']) != counts[i]:
            raise ValueError('Missing candidate support for a stored action')
        angles[i,:counts[i]] = obs['branch_angles']
        prior[i,:counts[i]] = obs['branch_prior']
    angles = torch.as_tensor(angles, device=values.device)
    valid = torch.arange(angles.shape[1],device=values.device)[None,:] < torch.tensor(counts,device=values.device)[:,None]
    turn = torch.ones_like(values,dtype=torch.bool)
    log_density = torch.stack([angle_policy.log_prob(turn,a) for a in angles.unbind(1)],dim=1)
    logits = log_density + torch.as_tensor(prior,device=values.device,dtype=values.dtype)
    probabilities = logits.masked_fill(~valid,-torch.inf).softmax(dim=1)
    # Explicit exploration gives each feasible branch a chance even if the warm-start
    # heading is concentrated on another bank. PPO reuses this identical mixture.
    probabilities = .88*probabilities + .12*valid/valid.sum(1,keepdim=True)
    logp = probabilities.clamp_min(torch.finfo(values.dtype).tiny).log().masked_fill(~valid,-torch.inf)
    return Categorical(logits=logp), values


def _options(car, goal, yaw, local, xy, free, occupied, probability, routes, size):
    distance = float(np.linalg.norm(goal-car))
    mission = _bearing(goal-car)
    offsets = [0., -10., 10., -20., 20., -35., 35., -50., 50., -65., 65.,
               -80., 80., -100., 100., -125., 125., -150., 150., -175., 175.]
    for route in routes.values():
        offsets.append(math.degrees(_wrap(_bearing(route.points[0]-car)-mission)))
    options = []
    delta = xy-car
    for degrees in sorted(set(offsets)):
        offset = math.radians(degrees)
        angle = mission+offset
        side = int(np.sign(degrees))
        option = _ray_option(side,angle,car,local,xy,free,occupied,probability,routes.get(side),yaw)
        if option is None:
            continue
        if degrees == 0.:
            option['target_m'] = min(option['target_m'],distance)
        target = car + option['target_m']*_unit(angle)
        if np.any(target < 0.) or np.any(target >= size):
            continue
        u = _unit(angle)
        along = delta @ u
        across = delta[...,0]*-u[1] + delta[...,1]*u[0]
        ahead = (along>=4.) & (along<=22.)
        flanks = []
        for sign in (-1,1):
            band = ahead & (sign*across>=3.) & (sign*across<=10.)
            flanks.append(float(occupied[band].mean()) if band.any() else 0.)
        option.update(degrees=degrees, target=target, paired_flanks=min(flanks),
                      final_goal=degrees==0. and option['target_m']==distance)
        option['score'] += .8*math.cos(offset)+.32*abs(math.sin(offset))-2.5*max(0.,abs(offset)-math.pi/2)
        options.append(option)
    if not options:
        return []
    # Bank-following and gap candidates compete at the same control tick. The gap
    # label represents occupancy geometry only; classes and map truth are unavailable.
    groups = [('forward',[o for o in options if abs(o['degrees'])<=10.]),
              ('opening',[o for o in options if abs(o['degrees'])<=100. and o['paired_flanks']>=.15
                          and o['reported_obstruction']<.2]),
              ('left',[o for o in options if o['degrees']>10.]),
              ('right',[o for o in options if o['degrees'] < -10.])]
    result = []
    for name, choices in groups:
        choices = [o for o in choices if all(abs(_wrap(other['angle']-o['angle']))>=math.radians(1.)
                                              for other in result)]
        if not choices:
            continue
        def rank(o):
            if name=='opening':
                return o['score']+1.5*o['paired_flanks']+.06*o['target_m']
            if name=='forward':
                return o['score']+.2*math.cos(math.radians(6.*o['degrees']))+float(o['final_goal'])
            return o['score']
        selected = max(choices,key=rank)
        result.append(dict(selected,branch=name))
    return result


class TrainableGuidance(RejoinGuidance):
    def __init__(self, *, deterministic, seed):
        super().__init__()
        self.deterministic = bool(deterministic)
        self.generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.transitions = []
        self.bootstrap_value = None
        self.bootstrap_step = None

    def _inputs(self, env, snapshot, fields, arm):
        from vtemp4.vtemp4_body_graph import body_frame_observation
        if arm not in ('predicted','observed'):
            raise ValueError('The branch selector needs a vehicle evidence arm')
        car, goal = snapshot['car_xy'], snapshot['goal_xy']
        now = int(snapshot['frame'])
        obs = env.obs[env.ground]
        yaw = float(obs['self']['orientation'])
        x,xy,presence,extent = fields
        free,occupied = x[0].astype(bool),x[1].astype(bool)
        size = snapshot['resolution']*snapshot['air_cells'].shape[0]
        prob = None
        if arm=='predicted':
            weight = min(1.,float(presence)/snapshot['presence_threshold'])
            prob = .5+weight*(extent-.5)
        routes = route_candidates(xy,car,goal,free,occupied,world_size=size,probabilities=prob)
        options = _options(car,goal,yaw,_local_sensor(obs),xy,free,occupied,prob,routes,size)
        base = self._base(env,snapshot,yaw,obs['self'])
        # The old 200m ring constrains far endpoints. These discrete actions
        # are prevalidated finite segments, so use the unconditional angular density.
        base['angle_arcs'] = np.array([[-math.pi,math.pi]],float)
        append_route_evidence(base,routes,car,goal,now)
        observation = body_frame_observation(base)
        observation['branch_angles'] = np.array([_wrap(o['angle']-yaw) for o in options] or [0.])
        observation['branch_prior'] = np.array([np.clip(o['score'],-2.,2.) for o in options] or [0.])
        return observation,options

    def bootstrap(self, env, snapshot, fields, actor, arm):
        observation,_ = self._inputs(env,snapshot,fields,arm)
        self.bootstrap_value = float(actor.critic_values([observation]).item())
        self.bootstrap_step = int(snapshot['frame'])

    def review(self, env, snapshot, fields, actor, arm):
        observation,options = self._inputs(env,snapshot,fields,arm)
        now = int(snapshot['frame'])
        car,goal = snapshot['car_xy'],snapshot['goal_xy']
        size = snapshot['resolution']*snapshot['air_cells'].shape[0]
        policy, values = branch_policy(actor,[observation])
        if self.deterministic:
            index = int(policy.probs[0].argmax().item())
        else:
            index = int(torch.multinomial(policy.probs[0].detach().cpu(),1,generator=self.generator).item())
        chosen = options[index] if options else None
        branch = chosen['branch'] if chosen else 'hold'
        angle = float(observation['branch_angles'][index])
        target = chosen['target'].copy() if chosen else car.copy()
        course = chosen['angle'] if chosen else None
        survey = np.clip(car+58.*_unit(course),1.5,size-1.5) if chosen else None
        if not chosen:
            # Aircraft continue probing while no certified initial segment is available.
            look = _bearing(goal-car)+(1 if (now//90)%2 else -1)*math.pi/2
            survey = np.clip(car+40.*_unit(look),1.5,size-1.5)
        controlled = len(options)>1
        logp = float(policy.log_prob(torch.tensor([index],device=values.device)).item()) if controlled else 0.
        self.transitions.append(dict(step=now,observation=observation,action=index,branch=branch,
            turn=True,angle_rad=angle,logp=logp,value=float(values.item()),policy_controlled=controlled))
        self.commands.append(dict(issued_at_s=now,origin_xy=car.copy(),target=target.copy()))
        previous = snapshot['execution_feedback'].get('accepted_target_xy')
        return dict(origin_xy=car.tolist(), target_xy=target.tolist(), survey_target_xy=survey.tolist(),
            course_world_rad=course, mode='follow_observed_branch' if chosen else 'wait_for_survey',
            source='sampled_received_branch' if controlled else 'forced_received_branch',
            intent_key=branch, force_replan=previous is None or not np.array_equal(target,previous),
            actor_called=controlled, angle_rad=angle, other_angle_rad=None,changed_angle_rad=None,
            candidate_count=len(options), branch_names=[o['branch'] for o in options],
            candidate_targets=[o['target'].tolist() for o in options], selected_branch=branch,
            action_index=index, action_probabilities=policy.probs[0].detach().cpu().tolist(),
            local_safe_prefix_m=chosen['prefix'] if chosen else 0.,
            final_goal=bool(chosen and chosen['final_goal']))

    def refresh_target(self, env, decision, car, accepted_target):
        if decision['final_goal']:
            return None
        return super().refresh_target(env,decision,car,accepted_target)

    def after_issue(self, decision, accepted_target, changed):
        if not np.array_equal(accepted_target,decision['target_xy']):
            raise RuntimeError('Issued vehicle target differs from sampled branch')
        super().after_issue(decision,accepted_target,changed)
