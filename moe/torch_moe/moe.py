import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .gate import Gating
from .expert import Expert

class MoE(nn.Module):
    def __init__(self, trained_experts) -> None:
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)

        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
        
        num_experts = len(trained_experts)

        input_dim = trained_experts[0].layer1.in_features
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        weights = self.gating(x)

        outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2
        )

        weights = weights.unsqueeze(1).expand_as(outputs)

        return torch.sum(outputs * weights, dim=2)
    

    
