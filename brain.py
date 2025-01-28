from modules import MLP
import torch.optim as optim
import torch.nn as nn
import torch

class Brain:

    def __init__(self, **kwargs):
        self.model = MLP(kwargs.get('input_dim', 59), kwargs.get('output_dim', 1), **kwargs)
        optim_class_name = kwargs.get('optimizer', 'Adam')
        self.optimizer = getattr(optim, optim_class_name)(self.model.parameters(), lr=kwargs.get('learning_rate', 0.001))
        self.loss_function = nn.BCELoss()
        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.model.to(self.device)


    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.loss_function(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return y_pred.detach(), loss.item()