from modules import MLP
import torch.optim as optim
import torch.nn as nn
import torch
import threading
from threading import Lock

class Brain:

    def __init__(self, **kwargs):
        self.model = MLP(**kwargs)
        optim_class_name = kwargs.get('optimizer', 'Adam')
        self.optimizer = getattr(optim, optim_class_name)(self.model.parameters(), lr=kwargs.get('learning_rate', 0.001))
        self.mode = kwargs.get('mode', 'OF')
        if self.mode == 'OF':
            self.loss_function = nn.BCELoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()

        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.model.to(self.device)
        self.model_lock = Lock()
        self.model_saving_path = kwargs.get('model_saving_path', 'default_model.pth')


    def train_step(self, x, y):
        with self.model_lock:
            self.model.train()
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            if self.mode == 'OF':
                loss = self.loss_function(y_pred, y)
            elif y_pred.shape[0] > 1:   # SW batch of 1 elem
                loss = self.loss_function(y_pred, y.long().squeeze())
            else:                       # SW batch of more elems
                loss = self.loss_function(y_pred.squeeze(), y.long().squeeze())           
            loss.backward()
            self.optimizer.step()
            return y_pred.detach(), loss.item()
    

    def get_brain_state_copy(self):
        with self.model_lock:
            return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def save_model(self):
        with self.model_lock:
            torch.save(self.model.state_dict(), self.model_saving_path)

    def update_weights(self, new_weights):
        """
        Safely update the model weights while preserving gradients
        """
        with self.model_lock:
            # Create a deep copy of the model's state dict
            current_state = self.model.state_dict()
            
            # Store references to the optimizer state
            optimizer_state = self.optimizer.state_dict()
            
            # Load the new weights
            self.model.load_state_dict(new_weights)
            
            # Make sure the new parameters are on the correct device
            for param in self.model.parameters():
                param.data = param.data.to(self.device)
            
            # Recreate the optimizer with the new parameters
            optim_class = self.optimizer.__class__
            self.optimizer = optim_class(
                self.model.parameters(),
                **{key: value for key, value in optimizer_state['param_groups'][0].items()
                if key != 'params'}
            )