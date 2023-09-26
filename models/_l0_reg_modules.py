import torch, math
import torch.nn as nn
import torch.nn.functional as F

class L0NormModule(nn.Module):
    def __init__(
            self, 
            layer: nn.Module, 
            alpha_mean: float=0.0, 
            alpha_std: float=0.01,
            beta: float=0.5,
            gamma: float=-0.1, 
            zeta: float=1.1,
            fixed_beta: bool=True,
            sparsity_on_eval: bool=True,
        ):
        super(L0NormModule, self).__init__()
        
        self._layer = layer
        self._weight_size = self._layer.weight.shape
        self._log_alpha = nn.Parameter(torch.normal(alpha_mean, alpha_std, size=self._weight_size))
        self._beta = beta if fixed_beta else nn.Parameter(torch.zeros(1).fill_(beta))
        self._gamma = gamma
        self._zeta = zeta
        self._sparsity_on_eval = sparsity_on_eval

        if self._sparsity_on_eval:
            self.register_buffer("u", torch.rand(self._weight_size).requires_grad_(True).to(self._log_alpha.device))
    
    def _compute_mask_and_penalty(self):
        if self._sparsity_on_eval:
            s = torch.sigmoid((torch.log(self.u) - torch.log(1 - self.u) + self._log_alpha) / self._beta)
            s = s * (self._zeta - self._gamma) + self._gamma

        else:
            if self.training:
                # sample u ~ U(0, 1)
                u = torch.rand(self._weight_size).requires_grad_(True).to(self._log_alpha.device)
                # s = sigmoid( (log(u) - log(1-u) + log(alpha)) / beta)
                s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self._log_alpha) / self._beta)
                # s := s * (zeta - gamma) + gamma
                s = s * (self._zeta - self._gamma) + self._gamma
            else:
                s = torch.sigmoid(self._log_alpha) * (self._zeta - self._gamma) + self._gamma
            
        if self.training:
            #Σ(sigmoid(log(alpha) - beta * gamma))
            penalty = F.sigmoid(self._log_alpha - (self._beta * math.log(-self._gamma / self._zeta))).sum()

        else:
            penalty = torch.tensor(0).to(self._log_alpha.device)

        # z = min(1, max(0, s))
        z = L0NormModule._hard_sigmoid(s)
        return z, penalty

    @staticmethod
    def _hard_sigmoid(x: torch.Tensor):
        return torch.min(torch.ones_like(x), torch.max(torch.zeros_like(x), x))
    

class L0Linear(L0NormModule):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, **kwargs):
        layer = nn.Linear(in_features, out_features, bias)
        super(L0Linear, self).__init__(layer, **kwargs)

    def forward(self, x: torch.Tensor):
        weight_mask, l0_penalty = self._compute_mask_and_penalty()
        output = F.linear(x, weight=self._layer.weight * weight_mask, bias=self._layer.bias)
        return output, l0_penalty
    

class L0Conv2d(L0NormModule):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            stride: int=1, 
            padding: int=0, 
            dilation: int=1, 
            groups: int=1, 
            bias: bool=True, **kwargs):
        
        layer = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            groups, 
            bias
        )
        super(L0Conv2d, self).__init__(layer, **kwargs)


    def forward(self, x: torch.Tensor):
        weight_mask, l0_penalty = self._compute_mask_and_penalty()
        output = F.conv2d(
            x, 
            weight=self._layer.weight * weight_mask, 
            bias=self._layer.bias, 
            stride=self._layer.stride, 
            padding=self._layer.padding, 
            dilation=self._layer.dilation, 
            groups=self._layer.groups
        )
        return output, l0_penalty