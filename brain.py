from modules import MLP, CNN1D, TabResNet
import torch
import torch.optim as optim
import torch.nn as nn
torch.backends.mkldnn.enabled = True
torch.backends.nnpack.enabled = False

import logging
from threading import Lock

# Module-level logger. It inherits the root logging configuration set up by
# consume.py (logging.basicConfig), so these messages are emitted alongside
# the rest of the consumer's logs.
logger = logging.getLogger(__name__)

# Maps the (lower-cased) model_type config value to the architecture class
# actually instantiated, so the log lines below can name it explicitly.
_MODEL_TYPE_TO_CLASS = {
    'cnn': CNN1D,
    'resnet': TabResNet,
    'mlp': MLP,
}

class Brain:

    def __init__(self, **kwargs):

        self.seed = kwargs.get('seed', None)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if 'cuda' in kwargs.get('device', 'cpu'):
                torch.cuda.manual_seed(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        requested_model_type = kwargs.get('model_type', 'mlp')
        model_type = str(requested_model_type).lower()

        # Be explicit about which architecture is being initialised. If the
        # requested model_type is unknown we fall back to MLP (same behaviour
        # as before) but make that fallback visible in the logs.
        resolved_class = _MODEL_TYPE_TO_CLASS.get(model_type, MLP)
        if model_type not in _MODEL_TYPE_TO_CLASS:
            logger.warning(
                f"Unknown model_type='{requested_model_type}'; "
                f"falling back to MLP architecture."
            )
        logger.info(
            f"Initialising model architecture: requested model_type="
            f"'{requested_model_type}' -> instantiating class "
            f"'{resolved_class.__name__}'."
        )
        # Log the architecture-shaping hyperparameters actually in effect.
        logger.info(
            "Architecture hyperparameters: "
            f"input_dim={kwargs.get('input_dim', 40)}, "
            f"h_dim={kwargs.get('h_dim', 128)}, "
            f"num_layers={kwargs.get('num_layers', 1)}, "
            f"output_dim={kwargs.get('output_dim', 1)}, "
            f"dropout={kwargs.get('dropout', 0.1)}, "
            f"layer_norm={kwargs.get('layer_norm', False)}, "
            f"mode={kwargs.get('mode')}."
        )

        try:
            if model_type == 'cnn':
                self.model = CNN1D(**kwargs)
            elif model_type == 'resnet':
                self.model = TabResNet(**kwargs)
            else:
                self.model = MLP(**kwargs)
        except Exception as e:
            logger.error(
                f"Failed to instantiate architecture '{resolved_class.__name__}' "
                f"(requested model_type='{requested_model_type}'): {e}",
                exc_info=True,
            )
            raise

        # Confirm the architecture was built and report its size, so it is
        # easy to verify from the logs that the intended model is in place.
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Architecture '{type(self.model).__name__}' instantiated "
            f"successfully ({total_params} parameters, "
            f"{trainable_params} trainable)."
        )

        init_strategy = kwargs.get('initialization_strategy', None)
        if init_strategy == 'xavier':
            logger.info("Applying 'xavier' weight initialisation to Linear layers.")
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            logger.info(
                f"Using default weight initialisation "
                f"(initialization_strategy={init_strategy})."
            )

        optim_class_name = kwargs.get('optimizer')
        try:
            self.main_stream_optimizer = getattr(optim, optim_class_name)(self.model.parameters(), lr=kwargs.get('learning_rate'))
        except Exception as e:
            logger.error(
                f"Failed to build optimizer '{optim_class_name}' "
                f"(learning_rate={kwargs.get('learning_rate')}): {e}",
                exc_info=True,
            )
            raise
        logger.info(
            f"Optimizer '{optim_class_name}' built with "
            f"learning_rate={kwargs.get('learning_rate')}."
        )
        self.main_stream_loss_function = nn.CrossEntropyLoss()
        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.model.to(self.device)
        logger.info(f"Model moved to device '{self.device}'. Brain ready.")
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
