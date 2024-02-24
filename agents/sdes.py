import torch
import abc
import math

from agents.helpers import (constant_theta_schedule,
                            linear_theta_schedule,
                            cosine_theta_schedule,
                            vp_beta_schedule,
                            )

class SDEBase(abc.ABC):
    def __init__(self, T, device):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x_t, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x_t, t):
        pass

    ################################################################################

    def sde_reverse_drift(self, x_t, t, score):
        return self.drift(x_t, t) - self.dispersion(x_t, t)**2 * score

    def ode_reverse_drift(self, x_t, t, score):
        return self.drift(x_t, t) - 0.5 * self.dispersion(x_t, t)**2 * score

    def dw(self, x_t): # Wiener
        return torch.randn_like(x_t) * math.sqrt(self.dt)

    def forward_step(self, x_t, t):
        dx = self.drift(x_t, t) * self.dt + self.dispersion(x_t, t) * self.dw(x_t)
        return x_t + dx

    def reverse_sde_step(self, x_t, t, score):
        dx = self.sde_reverse_drift(x_t, t, score) * self.dt \
             + self.dispersion(x_t, t) * self.dw(x_t) * (t > 0)
        return x_t - dx

    def reverse_ode_step(self, x_t, t, score):
        return x_t - self.ode_reverse_drift(x_t, t, score) * self.dt

    def forward(self, x_0):
        x_t = x_0
        for t in range(self.T):
            x_t = self.forward_step(x_t, t)
        return x_t

    def reverse(self, x_t, score_fn, mode='sde', **kwargs):
        for t in reversed(range(self.T)):
            score = score_fn(x_t, t, **kwargs)
            if mode == 'sde':
                x_t = self.reverse_sde_step(x_t, t, score)
            elif mode == 'ode':
                x_t = self.reverse_ode_step(x_t, t, score)
            else:
                print('the mode should be sde or ode')
                break
        return x_t


#-----------------------------------------------------------------------------#
#------------------------------- Tractable SDE -------------------------------#
#-----------------------------------------------------------------------------#

# mean-reverting SDE with 'mu=0'
class TractableSDE(SDEBase):
    def __init__(self, T=100, schedule='cosine', action_clip=False, device=None):
        super().__init__(T=T, device=device)
        self.action_clip = action_clip
        
        # beta and sigma for the SDE
        if schedule == 'cosine':
            self.thetas = cosine_theta_schedule(T).to(device)
        elif schedule == 'linear':
            self.thetas = linear_theta_schedule(T).to(device)
        elif schedule == 'constant':
            self.thetas = constant_theta_schedule(T).to(device)
        elif schedule == 'vp':
            self.thetas = vp_beta_schedule(T).to(device)
        else:
            print('Not implemented such schedule yet!!!')

        self.sigmas = torch.sqrt(2 * self.thetas)

        # recompute dt to make sure the SDE converges to a Gaussian(0, 1)
        thetas_cumsum = torch.cumsum(self.thetas, dim=0)
        self.dt = -math.log(1e-3) / thetas_cumsum[-1]
        self.thetas_cumsum = torch.cat([torch.zeros(1).to(device), thetas_cumsum])

        # compute theta_bar and SDE's variance/standard
        self.thetas_bar = thetas_cumsum * self.dt
        self.vars = 1 - torch.exp(-2 * self.thetas_bar)
        self.stds = torch.sqrt(self.vars)

        self.posterior_vars = (1 - torch.exp(-2 * self.thetas * self.dt)) \
                             * (1 - torch.exp(-2 * self.thetas_cumsum[:-1] * self.dt)) \
                             / (1 - torch.exp(-2 * self.thetas_cumsum[1:] * self.dt))

        self.log_posterior_vars = torch.log(torch.clamp(self.posterior_vars, min=1e-20))


    def mean(self, x_0, t):
        return x_0 * torch.exp(-self.thetas_bar[t])

    def variance(self, t):
        return self.vars[t]

    def drift(self, x_t, t):
        return -self.thetas[t] * x_t

    def dispersion(self, x_t, t):
        return self.sigmas[t]

    def forward_state(self, x_0, t, noise):
        return self.mean(x_0, t) + self.stds[t] * noise

    def ground_truth_score(self, x_t, t, x_0):
        return -(x_t - self.mean(x_0, t)) / self.variance(t)

    def compute_score_from_noise(self, noise, t):
        return -noise / self.stds[t]

    def predict_start_from_score(self, x_t, t, score):
        return (x_t + self.variance(t) * score) * torch.exp(self.thetas_bar[t])

    def predict_start_from_noise(self, x_t, t, noise):
        return (x_t - self.stds[t] * noise) * self.thetas_bar[t].exp()

    def log_forward_transition(self, x1, x2, t1, t2): # t1 < t2
        tb = self.thetas_bar[t2] - self.thetas_bar[t1]
        log_p = torch.log(2*math.pi * (1 - torch.exp(-2 * tb))) \
                + (x1 - x2 * torch.exp(-tb))**2 / (1 - torch.exp(-2 * tb))
        return -0.5 * log_p

    def log_reverse_transition(self, x_0, x1, x2, t1, t2): # t1 < t2
        return self.log_forward_transition(x1, x2, t1, t2) \
                + self.log_forward_transition(x_0, x1, 0, t1) \
                - self.log_forward_transition(x_0, x2, 0, t2)

    # sample states for training
    def generate_random_states(self, x_0):
        noise = torch.randn_like(x_0)
        t = torch.randint(0, self.T, (x_0.shape[0], 1), device=x_0.device).long()
        xt = self.forward_state(x_0, t, noise)
        return xt, t, noise

    # optimum x_{t-1}
    def reverse_optimum_step(self, x_t, x_0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        # self.thetas_cumsum has length T+1
        B = torch.exp(-self.thetas_cumsum[t+1] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)
        return term1 * x_t + term2 * x_0

    def optimal_reverse(self, x_t, x_0):
        x = x_t.clone()
        for t in reversed(range(self.T)):
            x = self.reverse_optimum_step(x, x_0, t)

        return x

    def posterior_step(self, x_t, t, noise, clip_value=1.0):
        x_0 = self.predict_start_from_noise(x_t, t, noise)
        if self.action_clip and clip_value > 0:
            x_0.clamp_(-clip_value, clip_value)
        mean_t = self.reverse_optimum_step(x_t, x_0, t)
        std_t = (0.5 * self.log_posterior_vars[t]).exp()
        noise = torch.randn_like(x_t)
        return mean_t + std_t * noise * (t > 0)

    def reverse(self, x_t, score_fn, mode='posterior', clip_value=1.0, **kwargs):
        for t in reversed(range(self.T)):
            score = score_fn(x_t, t, **kwargs)
            if mode == 'sde':
                x_t = self.reverse_sde_step(x_t, t, score)
            elif mode == 'ode':
                x_t = self.reverse_ode_step(x_t, t, score)
            elif mode == 'posterior':
                x_t = self.posterior_step(x_t, t, score, clip_value)
            else:
                print('the mode should be sde or ode')
                break
        return x_t

