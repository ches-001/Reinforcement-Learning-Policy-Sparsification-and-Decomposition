import torch
import torch.nn as nn
from ._l0_reg_modules import L0Linear, L0Conv2d
from typing import Dict, Tuple


class BaseDensePolicyNet(nn.Module):
    def __init__(self):
        super(BaseDensePolicyNet, self).__init__()

    def forward(self, x: torch.Tensor):
        pass


class BaseSparsePolicyNet(nn.Module):
    def __init__(self):
        super(BaseSparsePolicyNet, self).__init__()

    def forward(self, x: torch.Tensor):
        pass

    def compute_weight_sparsity(self) -> Tuple[Dict, float]:
        get_layer_sparisty = lambda z : z[z==0].shape[0] / (z.shape[0] * z.shape[1])
        layers = list(
            i[0] for i in self.named_children() if isinstance(i[1], L0Linear) or isinstance(i[1], L0Conv2d)
        )
        layer_sparsities = {}
        for layer in layers:
            z, _ = getattr(self, layer)._compute_mask_and_penalty()
            layer_sparsities[layer] = get_layer_sparisty(z)

        percent_sparisty = sum(list(layer_sparsities.values())) / len(layer_sparsities)
        return layer_sparsities, percent_sparisty


class PolicyNet(BaseDensePolicyNet):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.fc1(x)
        output = self.fc2(self.mid_layer_activation(output))
        output = self.fc3(self.mid_layer_activation(output))
        output = self.fc4(self.mid_layer_activation(output))
        return output
    

class SparsePolicyNet(BaseSparsePolicyNet):
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, p1 = self.fc1(x)
        output, p2 = self.fc2(self.mid_layer_activation(output))
        output, p3 = self.fc3(self.mid_layer_activation(output))
        output, p4 = self.fc4(self.mid_layer_activation(output))
        penalty = p1 + p2 + p3 + p4
        return output, penalty


class ConvPolicyNet(BaseDensePolicyNet):
    def __init__(self, in_channels: int, out_features: int, obs_h: int, obs_w: int):
        super(ConvPolicyNet, self).__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.obs_h = obs_h
        self.obs_w = obs_w

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=16, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1)
        self.fc = nn.Linear(self._get_fc_input_size(), out_features)
        self.mid_activation_layer = nn.ReLU()

    def _get_fc_input_size(self) -> int:
        x = torch.randn(5, self.in_channels, self.obs_h, self.obs_w)
        output = self.conv3(self.conv2(self.conv1(x)))
        output = output.flatten(start_dim=1, end_dim=-1)
        return output.shape[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv1(x)
        output = self.conv2(self.mid_activation_layer(output))
        output = self.conv3(self.mid_activation_layer(output))
        output = output.flatten(start_dim=1, end_dim=-1)
        output = self.fc(self.mid_activation_layer(output))
        return output


class SparseConvPolicyNet(BaseSparsePolicyNet):
    def __init__(self, in_channels: int, out_features: int, obs_h: int, obs_w: int):
        super(SparseConvPolicyNet, self).__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.obs_h = obs_h
        self.obs_w = obs_w

        self.conv1 = L0Conv2d(
            in_channels=self.in_channels, out_channels=16, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = L0Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = L0Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1)
        self.fc = L0Linear(self._get_fc_input_size(), out_features)
        self.mid_activation_layer = nn.ReLU()
    
    def _get_fc_input_size(self) -> int:
        x = torch.randn(5, self.in_channels, self.obs_h, self.obs_w)
        output = self.conv3(self.conv2(self.conv1(x)[0])[0])[0]
        output = output.flatten(start_dim=1, end_dim=-1)
        return output.shape[1]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, p1 = self.conv1(x)
        output, p2 = self.conv2(self.mid_activation_layer(output))
        output, p3 = self.conv3(self.mid_activation_layer(output))
        output = output.flatten(start_dim=1, end_dim=-1)
        output, p4 = self.fc(self.mid_activation_layer(output))
        p = p1 + p2 + p3 + p4
        return output, p