import torch
import torch.nn as nn
from ._l0_reg_modules import L0Linear
from typing import Dict, Tuple

class PolicyNet(nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_size:int=128):
        super(PolicyNet, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.fc3 = nn.Linear(self.hidden_size//2, self.hidden_size//4)
        self.fc4 = nn.Linear(self.hidden_size//4, self.output_size)
        self.mid_layer_activation = nn.ReLU()

    def forward(self, x:torch.Tensor):
        output = self.fc1(x)
        output = self.fc2(self.mid_layer_activation(output))
        output = self.fc3(self.mid_layer_activation(output))
        output = self.fc4(self.mid_layer_activation(output))
        return output
    

class SparsePolicyNet(nn.Module):
    def __init__(
            self, 
            input_size:int, 
            output_size:int, 
            hidden_size:int=128, 
            sparsity_on_eval: bool=True):
        super(SparsePolicyNet, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = L0Linear(self.input_size, self.hidden_size, sparsity_on_eval=sparsity_on_eval)
        self.fc2 = L0Linear(self.hidden_size, self.hidden_size//2, sparsity_on_eval=sparsity_on_eval)
        self.fc3 = L0Linear(self.hidden_size//2, self.hidden_size//4, sparsity_on_eval=sparsity_on_eval)
        self.fc4 = L0Linear(self.hidden_size//4, self.output_size, sparsity_on_eval=sparsity_on_eval)
        self.mid_layer_activation = nn.ReLU()

    def forward(self, x:torch.Tensor):
        output, p1 = self.fc1(x)
        output, p2 = self.fc2(self.mid_layer_activation(output))
        output, p3 = self.fc3(self.mid_layer_activation(output))
        output, p4 = self.fc4(self.mid_layer_activation(output))
        penalty = p1 + p2 + p3 + p4
        return output, penalty
    
    def compute_weight_sparsity(self) -> Tuple[Dict, float]:
        get_layer_sparisty = lambda z : z[z==0].shape[0] / (z.shape[0] * z.shape[1])
        layers = list(i[0] for i in self.named_children() if isinstance(i[1], L0Linear))

        layer_sparsities = {}
        for layer in layers:
            z, _ = getattr(self, layer)._compute_mask_and_penalty()
            layer_sparsities[layer] = get_layer_sparisty(z)

        percent_sparisty = sum(list(layer_sparsities.values())) / len(layer_sparsities)
        return layer_sparsities, percent_sparisty