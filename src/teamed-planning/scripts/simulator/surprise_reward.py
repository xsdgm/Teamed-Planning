"""Training-only recurrent surprise on newly delivered task-map evidence.

No world truth, actor features or deployment state are added. A frozen predictor
scores each complete on-policy batch before GAE. Predictor fitting follows PPO;
validation never receives intrinsic reward. All state is checkpointed.
"""
import copy
import numpy as np
import torch
from torch import nn

MAP_SIDE = 8
MAP_FEATURES = 2 * MAP_SIDE ** 2
ACTION_FEATURES = 4


class TaskMapEncoder:
    """Fixed world-aligned bins of the task mask, independent of vehicle motion."""
    def __init__(self, env):
        self.mask = env.reward_task_region.copy()
        y, x = np.nonzero(self.mask)
        if not len(x):
            raise ValueError('Surprise needs a nonempty task region')
        bx = np.minimum((x-x.min())*MAP_SIDE//max(int(x.max()-x.min()+1),1),MAP_SIDE-1)
        by = np.minimum((y-y.min())*MAP_SIDE//max(int(y.max()-y.min()+1),1),MAP_SIDE-1)
        self.bins = by*MAP_SIDE+bx
        self.counts = np.maximum(np.bincount(self.bins,minlength=MAP_SIDE**2),1)

    def encode(self, env):
        seen = env.ledger.seen[self.mask].astype(np.float32)
        confidence = env.ledger.confidence()[self.mask]
        return np.concatenate([np.bincount(self.bins,weights=a,minlength=MAP_SIDE**2)/self.counts
                               for a in (seen,confidence)]).astype(np.float32)


def transition(before, after, velocities, metrics, frame):
    """Fresh evidence only; no credit after connection, on missing links or step 0.

    The binary freshness mask excludes unchanged bins and zero-padded regions.
    Weight by fresh task area and received CBF margin, not exploration distance.
    """
    velocity = np.asarray(velocities,dtype=np.float32)[:, :2] / 6.
    action = np.r_[velocity.mean(axis=0),velocity.std(axis=0)].astype(np.float32)
    fresh = np.maximum(after[:MAP_SIDE**2]-before[:MAP_SIDE**2],0.)
    mask = np.tile((fresh>1e-8).astype(np.float32),2)
    gate = (float(frame>0) * float(bool(metrics['chain_links_available']))
            * float(not metrics['graph_task_connected'])
            * (1.-float(np.clip(metrics['cbf_margin_risk'],0.,1.)))
            * min(float(metrics['task_exploration_area'])/400.,1.))
    return dict(before=before,after=after,action=action,mask=mask,gate=gate)


class MapPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = nn.GRU(MAP_FEATURES+ACTION_FEATURES,64,batch_first=True)
        self.head = nn.Linear(64,MAP_FEATURES)
        nn.init.orthogonal_(self.head.weight,gain=.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, hidden=None):
        y, hidden = self.memory(x,hidden)
        # Predict map increments, preserving already known evidence.
        return x[...,:MAP_FEATURES]+self.head(y),hidden


class SurpriseReward:
    def __init__(self, config):
        self.config = config
        # Creating the auxiliary model must not change policy sampling RNG.
        with torch.random.fork_rng():
            torch.manual_seed(config['seed']+600006)
            self.model = MapPredictor()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config['surprise_learning_rate'])
        self.error_square_mean = 1e-4
        self.batches = 0

    def state_dict(self):
        return copy.deepcopy(dict(version=1,model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),error_square_mean=self.error_square_mean,batches=self.batches))

    def load_state_dict(self, state):
        if state['version']!=1:raise ValueError('Unsupported surprise checkpoint')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.error_square_mean=float(state['error_square_mean']);self.batches=int(state['batches'])

    def apply(self, rollouts, summaries, completed):
        """Mutate rewards once, before GAE, with fixed model and normalization.

        Environment reward stays in summary.reward/uav_reward for comparability;
        training_reward is the exact reward sum consumed by PPO. Metadata is
        removed so the existing six-field PPO rollout contract remains intact.
        """
        cfg=self.config
        progress=min(1.,completed/max(cfg['episodes']-1,1))
        beta=cfg['surprise_weight']+(cfg['surprise_weight_final']-cfg['surprise_weight'])*progress
        scale=max(self.error_square_mean,1e-4)
        self.model.eval()
        for rollout,summary in zip(rollouts,summaries):
            if any(len(r)!=7 for r in rollout):raise ValueError('Missing v6 surprise transition')
            total=0.;raw=[];eligible=0
            hidden=None
            with torch.no_grad():
                for i,row in enumerate(rollout):
                    item=row[6];bonus=0.
                    if item is not None:
                        x=torch.tensor(np.r_[item['before'],item['action']])[None,None]
                        predicted,hidden=self.model(x,hidden)
                        error=(predicted[0,0].numpy()-item['after'])**2
                        mask=item['mask'];valid=float(mask.sum())
                        mse=float((error*mask).sum()/max(valid,1.))
                        raw.append(mse)
                        if valid and item['gate']>0:
                            eligible+=1
                            bonus=min(beta*item['gate']*min(mse/scale,cfg['surprise_clip']),
                                      max(cfg['surprise_episode_budget']-total,0.))
                    total+=bonus
                    rollout[i]=(*row[:4],float(row[4])+bonus,row[5])
            summary.update(extrinsic_reward=summary['reward'],surprise_reward=total,
                surprise_error=float(np.mean(raw)) if raw else 0.,surprise_eligible_steps=eligible,
                surprise_weight=beta,surprise_scale=scale,
                training_reward=float(sum(row[4] for row in rollout)))

    def fit(self, transitions):
        """Episode-isolated TBPTT; no recurrent state crosses an episode.

        Fit all delivered map transitions, including familiar states, so surprise
        fades with learning. Normalize using eligible, fresh-bin prediction errors.
        """
        self.model.train();losses=[];errors=[]
        for _ in range(self.config['surprise_epochs']):
            for episode in transitions:
                items=[x for x in episode if x is not None]
                hidden=None
                for start in range(0,len(items),32):
                    chunk=items[start:start+32]
                    x=torch.tensor(np.array([np.r_[r['before'],r['action']] for r in chunk]))[None]
                    target=torch.tensor(np.array([r['after'] for r in chunk]))[None]
                    prediction,hidden=self.model(x,hidden)
                    squared=(prediction-target).square()
                    if _==0:
                        for error,item in zip(squared[0].detach().numpy(),chunk):
                            n=float(item['mask'].sum())
                            if n and item['gate']>0:errors.append(float((error*item['mask']).sum()/n))
                    loss=squared.mean()
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite surprise predictor loss')
                    self.optimizer.zero_grad();loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
                    self.optimizer.step();hidden=hidden.detach()
                    losses.append(float(loss.detach()))
        if errors:self.error_square_mean=max(.9*self.error_square_mean+.1*float(np.mean(errors)),1e-4)
        self.batches+=1
        return dict(surprise_predictor_loss=float(np.mean(losses)) if losses else 0.,
                    surprise_predictor_batches=self.batches)
