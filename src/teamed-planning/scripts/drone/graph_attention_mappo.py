"""Bounded received-map GAT for the shared continuous-velocity MAPPO policy.

Same sensor/state prefix and +/-96m map window as v6. Real graph edges only;
self loops are neural aggregation identities, never added traversable edges.
No actor accesses the ground receiver map, simulator truth or other actors.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from drone.velocity_mappo import (VelocityActorCritic, GRAPH_RADIUS, SENSOR_SIZE, STATE_FEATURES,
                                  INPUT_VERSIONS)

MAX_FLEET = 16
MAX_NODES = 64
NEIGHBORS = 8  # one self loop + at most seven existing graph neighbors
NODE_FEATURES = 8
GRAPH_BLOCK = MAX_NODES*(NODE_FEATURES+NEIGHBORS+1)


def prefix_width(input_version=1):
    # State + 2*12*12 actual sensor pixels: 320 (v6, input 1) or 322 (v6.2, input 2).
    return STATE_FEATURES[input_version]+2*SENSOR_SIZE**2


PREFIX = prefix_width(1)
GAT_FEATURES = PREFIX + GRAPH_BLOCK
ENCODED_FEATURES = PREFIX + 64


def encode_graph(base, position, observation, prefix=PREFIX):
    """Deterministic induced subgraph; padding and absent edges are explicit.

    Capping bounds CPU/RAM. Nodes outside the original +/-96m square are not
    exposed. Caps may discard a useful distant node or high-degree edge; they
    are a performance tradeoff, not a full-map representation.
    """
    nodes=np.zeros((MAX_NODES,NODE_FEATURES),np.float32)
    neighbors=np.full((MAX_NODES,NEIGHBORS),-1,np.float32)
    valid=np.zeros(MAX_NODES,np.float32)
    graph=observation.get('graph')
    if graph is not None:
        p=np.asarray(position[:2]);selected=[]
        for node,data in graph.graph.nodes(data=True):
            xy=np.asarray(data['position'][:2],float)-p
            if np.all(xy>=-GRAPH_RADIUS) and np.all(xy<GRAPH_RADIUS):
                selected.append((float(xy@xy),float(xy[0]),float(xy[1]),str(node),node,xy))
        selected.sort(key=lambda row:row[:4]);selected=selected[:MAX_NODES]
        index={row[4]:i for i,row in enumerate(selected)}
        for i,row in enumerate(selected):
            node,xy=row[4:];data=graph.graph.nodes[node]
            count=max(float(data.get('observation_count',0.)),0.)
            goal=np.asarray(base[4:6])*350.-xy if base[10] else np.zeros(2)
            nodes[i]=[* (xy/GRAPH_RADIUS),graph.node_confidence(node),
                      min(np.log1p(count)/np.log(1001.),1.),
                      min(graph.graph.degree(node)/8.,1.),*(goal/350.),
                      np.linalg.norm(xy)/(GRAPH_RADIUS*np.sqrt(2.))]
            adjacent=[index[j] for j in graph.graph.neighbors(node) if j in index and j!=node]
            adjacent.sort(key=lambda j:(np.linalg.norm(selected[j][5]-xy),j))
            adjacent=[i]+adjacent[:NEIGHBORS-1]
            neighbors[i,:len(adjacent)]=adjacent;valid[i]=1.
    return np.r_[base[:prefix],nodes.ravel(),neighbors.ravel(),valid].astype(np.float32)


class AttentionLayer(nn.Module):
    def __init__(self, incoming, heads=2, width=16):
        super().__init__();self.heads=heads;self.width=width
        self.project=nn.Linear(incoming,heads*width,bias=False)
        self.source=nn.Parameter(torch.empty(heads,width))
        self.target=nn.Parameter(torch.empty(heads,width))
        self.edge=nn.Linear(1,heads,bias=False)
        nn.init.xavier_uniform_(self.source);nn.init.xavier_uniform_(self.target)
        self.residual=nn.Linear(incoming,heads*width,bias=False)

    def forward(self,x,indices,mask,coordinates):
        batch,nodes,_=x.shape
        ids=indices.clamp_min(0)
        bi=torch.arange(batch,device=x.device)[:,None,None]
        z=self.project(x).reshape(batch,nodes,self.heads,self.width)
        adjacent=z[bi,ids]
        delta=coordinates[bi,ids]-coordinates[:,:,None,:]
        length=delta.square().sum(-1,keepdim=True).sqrt()
        logits=(z*self.source).sum(-1)[:,:,None,:]+(adjacent*self.target).sum(-1)+self.edge(length)
        logits=torch.nn.functional.leaky_relu(logits,.2)
        legal=(indices>=0)&mask[:,:,None]&mask[bi,ids]
        weights=logits.masked_fill(~legal[...,None],-1e9).softmax(dim=2)
        out=(weights[...,None]*adjacent).sum(2).reshape(batch,nodes,-1)
        return torch.nn.functional.elu(out+self.residual(x))*mask[...,None]


class GraphEncoder(nn.Module):
    def __init__(self,prefix=PREFIX):
        super().__init__();self.layer1=AttentionLayer(NODE_FEATURES);self.layer2=AttentionLayer(32)
        self.prefix=prefix

    def forward(self,features):
        prefix=self.prefix
        shape=features.shape[:-1];flat=features.reshape(-1,prefix+GRAPH_BLOCK)
        outputs=[]
        # Bound intermediate gather tensors for large PPO/critic batches.
        for part in flat.split(128):
            offset=prefix
            nodes=part[:,offset:offset+MAX_NODES*NODE_FEATURES].reshape(-1,MAX_NODES,NODE_FEATURES)
            offset+=MAX_NODES*NODE_FEATURES
            indices=part[:,offset:offset+MAX_NODES*NEIGHBORS].reshape(-1,MAX_NODES,NEIGHBORS).long()
            mask=part[:,-MAX_NODES:]>0
            h=self.layer1(nodes,indices,mask,nodes[:,:,:2]);h=self.layer2(h,indices,mask,nodes[:,:,:2])
            mean=h.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            maximum=h.masked_fill(~mask[...,None],-1e9).amax(1)
            maximum=torch.where(mask.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
            outputs.append(torch.cat((part[:,:prefix],mean,maximum),dim=-1))
        return torch.cat(outputs).reshape(*shape,prefix+64)


class CriticGraphEncoder(nn.Module):
    def __init__(self,global_features,prefix=PREFIX):
        super().__init__();self.global_features=global_features;self.graph=GraphEncoder(prefix)

    def forward(self,x):
        # Encode each actor's own graph BEFORE pooling. Never average adjacency
        # indices from different maps. This encoder belongs to critic only.
        width=self.graph.prefix+GRAPH_BLOCK;encoded_width=self.graph.prefix+64
        valid=x[...,-1]>0
        flat=x[...,:width].reshape(-1,width)
        ids=valid.reshape(-1).nonzero().squeeze(-1)
        values=self.graph(flat[ids])
        encoded=values.new_zeros((len(flat),encoded_width)).index_copy(0,ids,values)
        encoded=encoded.reshape(*valid.shape,encoded_width).sum(-2)/valid.sum(-1,keepdim=True).clamp_min(1)
        if self.global_features:
            encoded=torch.cat((encoded,x[...,0,width:width+self.global_features]),dim=-1)
        return encoded


class GraphAttentionActorCritic(nn.Module):
    action_mode='velocity'
    policy_architecture='gat_map_v1'
    velocity=staticmethod(VelocityActorCritic.velocity)

    def __init__(self,global_features=10,input_version=1):
        super().__init__();self.global_features=int(global_features)
        self.input_version=int(input_version);self.prefix=prefix_width(self.input_version)
        # Checkpoint feature versions: 4 = v6 GAT inputs, 6 = v6.2 GAT inputs.
        self.feature_version=4 if self.input_version==1 else 6
        encoded=self.prefix+64
        self.gat_actor=nn.Sequential(GraphEncoder(self.prefix),nn.Linear(encoded,128),nn.Tanh(),
                                     nn.Linear(128,128),nn.Tanh(),nn.Linear(128,2))
        self.log_std=nn.Parameter(torch.full((2,),-.5))
        self.critic=nn.Sequential(CriticGraphEncoder(self.global_features,self.prefix),
            nn.Linear(encoded+self.global_features,128),nn.Tanh(),
            nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.orthogonal_(self.gat_actor[-1].weight,gain=.01);nn.init.zeros_(self.gat_actor[-1].bias)
        self.register_buffer('value_mean',torch.zeros(()));self.register_buffer('value_std',torch.ones(()))
        self.register_buffer('value_moments',torch.zeros(3))

    @classmethod
    def from_state_dict(cls,state):
        # The critic's first linear layer follows its private graph encoder.
        prefix=state['gat_actor.1.weight'].shape[1]-64
        versions=[v for v in INPUT_VERSIONS if prefix_width(v)==prefix]
        if not versions:raise ValueError(f'Unknown GAT actor prefix width {prefix}')
        model=cls(state['critic.1.weight'].shape[1]-prefix-64,versions[0])
        model.load_state_dict(state);return model

    def encode_graph(self,base,position,observation):
        return encode_graph(base,position,observation,self.prefix)

    def distribution(self,x,mask=None):
        return Independent(Normal(self.gat_actor(x),self.log_std.clamp(-5.,1.).exp()),1)

    def critic_input(self,x,mask=None,state=None):
        if not 0<x.shape[-2]<=MAX_FLEET:raise ValueError('Graph critic supports 1..16 agents')
        if self.global_features:
            if state is None:raise ValueError('Graph critic requires centralized training state')
            state=state.unsqueeze(-2).expand(*x.shape[:-1],self.global_features)
            x=torch.cat((x,state),dim=-1)
        x=torch.cat((x,x.new_ones((*x.shape[:-1],1))),dim=-1)
        padded=x.new_zeros((*x.shape[:-2],MAX_FLEET,x.shape[-1]))
        padded[...,:x.shape[-2],:]=x
        return padded

    def value(self,x,mask=None,state=None):
        return self.critic(self.critic_input(x,mask,state)).squeeze(-1)*self.value_std+self.value_mean
