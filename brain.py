from modules import MLP
import torch.optim as optim
import torch.nn as nn
import torch
from threading import Lock

class Brain:

    def __init__(self, **kwargs):

        self.seed = kwargs.get('seed', None)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if 'cuda' in kwargs.get('device', 'cpu'):
                torch.cuda.manual_seed(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.model = MLP(**kwargs)
        optim_class_name = kwargs.get('optimizer')
        self.main_stream_optimizer = getattr(optim, optim_class_name)(self.model.parameters(), lr=kwargs.get('learning_rate'))
        self.main_stream_loss_function = nn.CrossEntropyLoss()
        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.model.to(self.device)
        self.model_lock = Lock()
        self.model_saving_path = kwargs.get('model_saving_path', 'default_model.pth')

        # FedProx: proximal coefficient (0 disables the term, recovering FedAvg)
        self.fedprox_mu = kwargs.get('fedprox_mu', 0.0)
        # Frozen reference point updated each time the global model is pulled
        self.global_weights = None


    def train_step(self, feats, main_labels):

        with self.model_lock:
            self.model.train()
            self.main_stream_optimizer.zero_grad()

            main_pred, _ = self.model(feats)
            loss = self.main_stream_loss_function(main_pred, main_labels.squeeze())

            # FedProx proximal term: mu/2 * ||w - w_global||^2
            # Anchors local updates to the last received global model,
            # preventing divergence under heterogeneous data distributions.
            if self.fedprox_mu > 0.0 and self.global_weights is not None:
                prox_term = sum(
                    ((param - self.global_weights[name].to(self.device)) ** 2).sum()
                    for name, param in self.model.named_parameters()
                    if name in self.global_weights
                )
                loss = loss + (self.fedprox_mu / 2.0) * prox_term

            loss.backward()
            self.main_stream_optimizer.step()

            return main_pred.detach(), loss.item()


    def get_brain_state_copy(self):
        with self.model_lock:
            return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def save_model(self):
        with self.model_lock:
            torch.save(self.model.state_dict(), self.model_saving_path)
            

    def set_global_reference(self, weights):
        """Store a frozen copy of the global model to use as the FedProx anchor."""
        with self.model_lock:
            self.global_weights = {k: v.detach().clone().to(self.device) for k, v in weights.items()}

    def update_weights(self, new_weights):
        """
        Safely update the model weights while preserving gradients
        """
        with self.model_lock:
            current_state = self.model.state_dict()

            main_stream_optimizer_state = self.main_stream_optimizer.state_dict()
            self.model.load_state_dict(new_weights)

            for param in self.model.parameters():
                param.data = param.data.to(self.device)

            main_stream_optim_class = self.main_stream_optimizer.__class__
            self.main_stream_optimizer = main_stream_optim_class(
                self.model.parameters(),
                **{key: value for key, value in main_stream_optimizer_state['param_groups'][0].items()
                if key != 'params'}
            )
