import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(MLP, self).__init__()
        h_dim = kwargs.get('h_dim', 128)
        dropout = kwargs.get('dropout', 0.5)
        num_layers = kwargs.get('num_layers', 3)

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h_dim

        layers.append(nn.Linear(h_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
