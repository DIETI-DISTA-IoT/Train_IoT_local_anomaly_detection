import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(MLP, self).__init__()

        h_dim = kwargs.get('h_dim', 128)
        dropout = kwargs.get('dropout', 0.1)
        num_layers = kwargs.get('num_layers', 3)
        layer_norm = kwargs.get('layer_norm', False)
        mode = kwargs.get('mode')
        probe_len = len(kwargs.get('probe_metrics'))

        main_stream = []
        aux_stream = []

        # stream:
        if layer_norm:
            main_stream.append(nn.LayerNorm(input_dim))
            aux_stream.append(nn.LayerNorm(probe_len))

        # layer iteration
        curr_output_dim = h_dim
        curr_main_input_dim = input_dim
        curr_aux_input_dim = probe_len

        for _ in range(num_layers):

            main_stream.append(nn.Linear(curr_main_input_dim, curr_output_dim))
            aux_stream.append(nn.Linear(curr_aux_input_dim, curr_output_dim))

            main_stream.append(nn.ReLU())
            aux_stream.append(nn.ReLU())

            main_stream.append(nn.Dropout(dropout))
            aux_stream.append(nn.Dropout(dropout))

            curr_aux_input_dim = curr_output_dim
            curr_main_input_dim = curr_output_dim

            curr_output_dim = curr_output_dim // 2

        # pre-heads:
        main_stream.append(nn.Linear(curr_main_input_dim, 1))
        main_stream.append(nn.Sigmoid())

        aux_stream.append(nn.Linear(curr_aux_input_dim, 1))
        aux_stream.append(nn.Sigmoid())

        # final-head:
        final_head_1 = nn.Linear(curr_main_input_dim + curr_aux_input_dim, curr_output_dim)
        final_head_2 = nn.Linear(curr_output_dim, output_dim)
        final_head_3 = nn.Softmax(dim=1)

        self.main_stream = nn.Sequential(*main_stream)
        self.aux_stream = nn.Sequential(*aux_stream)
        self.final_head = nn.Sequential(final_head_1, final_head_2, final_head_3)


    def forward(self, x):
        main, aux, final = None, None, None

        main = self.main_stream(x)

        if self.mode == 'SW':
            aux = self.aux_stream(x)
            final = self.final_head(torch.cat((main, aux), dim=1))
        
        return final, main, aux
