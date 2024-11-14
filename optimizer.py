import torch
import numpy as np

class ScheduledOptim:
    def __init__(self, model, t_config, m_config):

        betas = t_config["optimizer"]["betas"]
        eps = t_config["optimizer"]["eps"]
        weight_decay = t_config["optimizer"]["weight_decay"]
        self.warm_up_step = t_config["optimizer"]["warm_up_step"]
        self.anneal_steps = t_config["optimizer"]["anneal_steps"]
        self.anneal_rate = t_config["optimizer"]["anneal_rate"]
        self.d_model = m_config["common"]["d_hidden"]
        self.init_lr = np.power(self.d_model, -0.5)
        self.current_step = 0

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.warm_up_step, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr