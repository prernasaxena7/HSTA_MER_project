# File: hsta_model_core/novograd.py

import torch
from torch.optim.optimizer import Optimizer

class NovoGrad(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.95, 0.98), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(NovoGrad, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NovoGrad does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                exp_avg.mul_(beta1).addcdiv_(grad, denom, value=1 - beta1)

                step_size = group['lr']
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'])

                p.data.add_(exp_avg, alpha=-step_size)

        return loss
