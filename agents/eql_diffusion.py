import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.sdes import TractableSDE
from agents.diffusion import DiffusionPolicy
from agents.model import EnsembleCritic

from agents.helpers import EMA
from tqdm import tqdm


class Diffusion_EQL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 schedule='cosine',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ent_coef=0.2,
                 num_critics=4,
                 pess_method='lcb', # ['min', 'lcb']
                 lcb_coef=4.,    # [4, 8]
                 loss_type='MLL',
                 action_clip=False
                 ):

        self.sde = TractableSDE(n_timesteps, schedule, action_clip, device=device)
        self.actor = DiffusionPolicy(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                                     sde=self.sde, n_timesteps=n_timesteps, loss_type=loss_type).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.pess_method = pess_method
        self.lcb_coef = lcb_coef


        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        self.num_critics = num_critics

        self.critic = EnsembleCritic(state_dim, action_dim, num_critics=num_critics).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup
        self.ent_coef = torch.tensor(ent_coef).to(self.device)


    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': [], 'entropy': []}

        for _ in tqdm(range(iterations)):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            ##################
            """ Q Training """
            ##################
            current_q_values = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                q_next_rpt = self.critic_target(next_state_rpt, next_action_rpt)
                q_next = q_next_rpt.view(batch_size, 10, -1).max(dim=1)[0]
            else:
                next_action = self.ema_model(next_state)
                q_next = self.critic_target(next_state, next_action) # shape: batch_siz, num_critic

            target_q = (reward + not_done * self.discount * q_next).detach()
            critic_loss = F.mse_loss(current_q_values, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            #######################
            """ Policy Training """
            #######################
            bc_loss, new_action = self.actor.loss(state, action)
            entropy = self.actor.entropy(new_action)

            q_values_new_action_ensembles = self.critic(state, self.actor.clip_action(new_action))
            if self.pess_method == 'min':
                q_values_new_action = q_values_new_action_ensembles.min(dim=1, keepdim=True)[0]
            elif self.pess_method == 'lcb':
                mu = q_values_new_action_ensembles.mean(dim=1, keepdim=True)
                std = q_values_new_action_ensembles.std(dim=1, keepdim=True)
                q_values_new_action = mu - self.lcb_coef * std

            q_loss = -q_values_new_action.mean() / q_values_new_action_ensembles.abs().mean().detach()
            entropy_loss = -entropy.mean() / entropy.abs().mean().detach()
            
            actor_loss = bc_loss + self.eta * q_loss + self.ent_coef * entropy_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())
            metric['entropy'].append(entropy.mean().item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric


    def _sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor(state).squeeze()
        return action.cpu().numpy()

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor(state_rpt)
            q_value = self.critic_target(state_rpt, action)
            q_mean = q_value.mean(dim=1,keepdim=True).flatten()
            q_std = q_value.std(dim=1,keepdim=True).flatten()
            q_lcb = q_mean - self.lcb_coef * q_std
            idx = torch.multinomial(F.softmax(q_lcb, dim=0), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))



