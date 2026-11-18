"""Action-conditioned latent transitions and abstract evidence-state predictions.

There is no map decoder or image/semantic reconstruction loss. Every output is
an offline observation-task hypothesis, not a verified vehicle graph or command.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .abstract_data import CHANNELS, STATE_DIM, STATE_NAMES, UTILITY_WEIGHTS, TRAVEL_WEIGHT

ARCHITECTURE='abstract_observation_vtmp2'
FORMAT='vtmp2-abstract-observation-v1'
SCHEMA_VERSION=7


class Encoder(nn.Module):
    def __init__(self,hidden,latent):
        super().__init__()
        self.visual=nn.Sequential(nn.Conv2d(CHANNELS,hidden,3,padding=1),nn.GroupNorm(4,hidden),nn.SiLU(),
            nn.Conv2d(hidden,hidden,3,stride=2,padding=1),nn.GroupNorm(4,hidden),nn.SiLU(),
            nn.Conv2d(hidden,hidden*2,3,stride=2,padding=1),nn.GroupNorm(4,hidden*2),nn.SiLU(),nn.Flatten())
        self.project=nn.Sequential(nn.Linear(hidden*2*4*4+4,latent),nn.LayerNorm(latent),nn.SiLU(),
                                   nn.Linear(latent,latent),nn.LayerNorm(latent))

    def forward(self,maps,task):
        observed=maps[:,5:6]>.5
        safe=torch.cat((torch.where(observed,maps[:,:5],0.),maps[:,5:6],
                        torch.where(observed,maps[:,6:7],0.),maps[:,7:]),1)
        if not torch.isfinite(safe).all():raise ValueError('Nonfinite observed map/task channels')
        return self.project(torch.cat((self.visual(safe),task),-1))


def advance_state(before,raw):
    delta=raw.sigmoid()
    values=before+(1-before)*delta
    connected=values[...,0]
    progress=torch.maximum(values[...,1],connected)
    quality=before[...,2]+(connected-before[...,2]).clamp_min(0)*delta[...,2]
    known=values[...,3]
    nonopen=before[...,4]+(known-before[...,4]).clamp_min(0)*delta[...,4]
    goal_observed=torch.maximum(values[...,6],connected)
    return torch.stack((connected,progress,quality,known,nonopen,values[...,5],goal_observed),-1)


class Member(nn.Module):
    def __init__(self,hidden,latent):
        super().__init__()
        self.encoder=Encoder(hidden,latent)
        self.teacher=deepcopy(self.encoder).requires_grad_(False)
        self.transition=nn.Sequential(nn.Linear(latent+STATE_DIM+7,latent*2),nn.SiLU(),nn.Linear(latent*2,latent))
        self.normalize=nn.LayerNorm(latent)
        self.outcome=nn.Sequential(nn.Linear(latent+STATE_DIM+7,latent),nn.SiLU(),nn.Linear(latent,STATE_DIM))
        nn.init.zeros_(self.outcome[-1].weight);nn.init.constant_(self.outcome[-1].bias,-3.)
        self.abstract_readout=nn.Linear(latent,STATE_DIM)

    def forward(self,maps,state,task,actions):
        batch,branches,horizon,_=actions.shape
        encoded=self.encoder(maps,task)
        z=encoded[:,None].expand(-1,branches,-1)
        current=state[:,None].expand(-1,branches,-1)
        task=task[:,None].expand(-1,branches,-1)
        states=[];latents=[]
        for step in range(horizon):
            inputs=torch.cat((z,current,task,actions[:,:,step]),-1)
            z=self.normalize(z+self.transition(inputs))
            current=advance_state(current,self.outcome(torch.cat((z,current,task,actions[:,:,step]),-1)))
            states.append(current);latents.append(z)
        return torch.stack(states,2),torch.stack(latents,2),encoded


class AbstractWorldModel(nn.Module):
    algorithm='vtmp2'
    model_architecture=ARCHITECTURE

    def __init__(self,hidden=24,latent=96,members=3):
        super().__init__()
        if hidden%4 or not 4<=hidden<=64 or not 16<=latent<=256 or not 1<=members<=5:
            raise ValueError('Invalid abstract model dimensions')
        self.hidden,self.latent,self.members=hidden,latent,members
        self.predictors=nn.ModuleList(Member(hidden,latent) for _ in range(members))

    def configuration(self):return dict(hidden=self.hidden,latent=self.latent,members=self.members)

    def forward(self,maps,state,task,actions):
        if maps.ndim!=4 or maps.shape[1:]!=(CHANNELS,16,16) or state.shape!=(len(maps),STATE_DIM) or task.shape!=(len(maps),4):
            raise ValueError('Invalid abstract observation/state/task batch')
        if actions.ndim!=4 or actions.shape[0]!=len(maps) or actions.shape[-1]!=3:
            raise ValueError('Expected candidate action sequences')
        if not all(torch.isfinite(x).all() for x in (state,task,actions)):
            raise ValueError('Nonfinite task/state/action')
        values=[member(maps,state,task,actions) for member in self.predictors]
        return dict(states=torch.stack([v[0] for v in values]),latents=torch.stack([v[1] for v in values]),
                    encoded=torch.stack([v[2] for v in values]))

    @torch.no_grad()
    def predict(self,maps,state,task,actions):
        output=self(maps,state,task,actions)
        return dict(state_mean=output['states'].mean(0),state_std=output['states'].std(0,unbiased=False),
                    state_members=output['states'],latent_mean=output['latents'].mean(0),
                    is_vehicle_traversability=False,action_domain='offline_observation_windows')

    @torch.no_grad()
    def update_teacher(self,rate=.99):
        for member in self.predictors:
            for target,source in zip(member.teacher.parameters(),member.encoder.parameters()):
                target.mul_(rate).add_(source,alpha=1-rate)


def loss(model,batch,scales):
    output=model(batch['maps'],batch['state'],batch['task'],batch['actions'])
    target=batch['target'];weights=target.new_tensor(UTILITY_WEIGHTS)
    truth_score=((target-batch['state'][:,None,None])*weights).sum(-1)-TRAVEL_WEIGHT*batch['costs']
    all_losses=[];diagnostics=[]
    for i,member in enumerate(model.predictors):
        p=output['states'][i]
        continuous=(((p[...,1:]-target[...,1:])/scales[1:])**2).mean((1,2,3))
        binary=F.binary_cross_entropy(p[...,0].clamp(1e-6,1-1e-6),target[...,0],reduction='none').mean((1,2))
        batch_size,branches,horizon=target.shape[:3]
        future=batch['future_maps'].flatten(0,2)
        task=batch['task'][:,None,None].expand(-1,branches,horizon,-1).reshape(-1,4)
        with torch.no_grad():future_z=member.teacher(future,task).reshape(batch_size,branches,horizon,-1)
        latent=(output['latents'][i]-future_z).square().mean((1,2,3))
        anchor=((member.abstract_readout(output['encoded'][i]).sigmoid()-batch['state'])**2).mean(1)
        score=((p-batch['state'][:,None,None])*weights).sum(-1)-TRAVEL_WEIGHT*batch['costs']
        rank=F.kl_div((score[:,:,0]/.2).log_softmax(1),(truth_score[:,:,0]/.2).softmax(1),reduction='none').sum(1)
        # Bootstrap members without selecting validation masks or labels.
        bootstrap=torch.poisson(torch.ones_like(binary))
        objective=continuous+2*binary+.2*latent+.2*anchor+.5*rank
        all_losses.append((objective*bootstrap).sum()/bootstrap.sum().clamp_min(1))
        diagnostics.append(torch.stack((continuous.mean(),binary.mean(),latent.mean(),rank.mean())))
    return torch.stack(all_losses).mean(),torch.stack(diagnostics).mean(0)


def load_checkpoint(path,device='cpu'):
    document=torch.load(Path(path),map_location='cpu',weights_only=True)
    if (document.get('format')!=FORMAT or document.get('schema_version')!=SCHEMA_VERSION or
        document.get('model_architecture')!=ARCHITECTURE or document.get('states')!=list(STATE_NAMES)):
        raise ValueError('Expected schema-7 abstract observation checkpoint')
    model=AbstractWorldModel(**document['architecture']);model.load_state_dict(document['model'],strict=True)
    if any(not torch.isfinite(x).all() for x in model.state_dict().values()):raise ValueError('Nonfinite checkpoint')
    return model.to(device).eval().requires_grad_(False),document
