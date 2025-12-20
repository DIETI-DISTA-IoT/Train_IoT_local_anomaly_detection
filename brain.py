from modules import MLP
import torch.optim as optim
import torch.nn as nn
import torch
import threading
from threading import Lock

class Brain:

    def __init__(self, **kwargs):
        self.model = MLP(**kwargs)
        optim_class_name = kwargs.get('optimizer')
        self.main_stream_optimizer = getattr(optim, optim_class_name)(self.model.parameters(), lr=kwargs.get('learning_rate'))
        self.main_stream_loss_function = nn.CrossEntropyLoss()
        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.model.to(self.device)
        self.model_lock = Lock()
        self.model_saving_path = kwargs.get('model_saving_path', 'default_model.pth')
        

    def train_step(self, feats, main_labels):
        
        with self.model_lock:
            self.model.train()
            self.main_stream_optimizer.zero_grad()

            main_pred, _ = self.model(feats)
            main_stream_loss = 0
            main_stream_loss = self.main_stream_loss_function(main_pred, main_labels.squeeze())
            main_stream_loss.backward()
            self.main_stream_optimizer.step()

            return main_pred.detach(), main_stream_loss.item()
    

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
            main_stream_optimizer_state = self.main_stream_optimizer.state_dict()
            
            # Load the new weights
            self.model.load_state_dict(new_weights)
            
            # Make sure the new parameters are on the correct device
            for param in self.model.parameters():
                param.data = param.data.to(self.device)
            
            # Recreate the optimizer with the new parameters
            main_stream_optim_class = self.main_stream_optimizer.__class__
            self.main_stream_optimizer = main_stream_optim_class(
                self.model.parameters(),
                **{key: value for key, value in main_stream_optimizer_state['param_groups'][0].items()
                if key != 'params'}
            )