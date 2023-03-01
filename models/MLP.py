import torch.nn as nn
import torch

class mlp_model(nn.Module):
    def __init__(self, nb_hidden_layers, input_dim=(2,3), hidden_dim = 256, nb_of_joints = 16):
        super().__init__()
        flattened_input_dim = nb_of_joints * input_dim[0] 
        flattened_output_dim = nb_of_joints * input_dim[1]

        self.relu = nn.ReLU()
        self.input_layer = nn.Linear(flattened_input_dim, hidden_dim)
        layers = []
        for i in range(nb_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, flattened_output_dim)


    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layers(x)
        x = self.relu(x)
        out = self.output_layer(x)

        return out

        

