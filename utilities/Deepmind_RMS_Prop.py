import torch
from torch.optim import Optimizer


class DM_RMSprop(Optimizer):
    """Implements the form of RMSProp used in DM 2015 Atari paper.
    Inspired by https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/updates.py"""

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(DM_RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DM_RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            momentum = group['momentum']
            sq_momentum = group['alpha']
            epsilon = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)

                mom_buffer = state['momentum_buffer']
                square_avg = state['square_avg']


                state['step'] += 1

                mom_buffer.mul_(momentum)
                mom_buffer.add_((1 - momentum) * grad)

                square_avg.mul_(sq_momentum).addcmul_(1 - sq_momentum, grad, grad)

                avg = (square_avg -  mom_buffer**2 + epsilon).sqrt()

                p.data.addcdiv_(-group['lr'], grad, avg)

        return loss

