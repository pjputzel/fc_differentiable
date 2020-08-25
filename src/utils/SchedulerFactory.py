import torch

def create_scheduler(optimizer, params):
    if params['scheduler_type'] == 'constant':
        scheduler = ConstantScheduler(optimizer)
    elif params['scheduler_type'] == 'cosine_annealing_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         params['warm_restart_params']['t_0'],
                                                                         params['warm_restart_params']['t_mult'],
                                                                         params['warm_restart_params']['eta_min'],
                                                                         params['warm_restart_params']['last_epoch'])
    elif params['scheduler_type'] == 'reduce_lr_on_plateau':
        scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    return scheduler


class ConstantScheduler:
    def __init__(self, optimizer):
        return

    def step(self):
        return
