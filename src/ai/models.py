import torch
from torch import nn


class NNModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        # Feed Forward Layer 1 (Inputs)
        self.ff1 = nn.Linear(16, 64)
        self.relu1 = nn.ReLU()
        
        # Feed Forward Layer 2 (incl residual connections from input layer)
        self.ff2 = nn.Linear(64+16, 128)
        self.relu2 = nn.ReLU()
        
        # Feed Forward Layer 3 (incl residual connections from FF1 and input layer)
        self.ff3 = nn.Linear(128+16+64, 256)
        self.relu3 = nn.ReLU()
        
        # Feed Forward Layer 4 (incl residual connections from FF2 and input layer)
        self.ff4 = nn.Linear(256+16+64+128, 128)
        self.relu4 = nn.ReLU()
        
        # Feed Forward Layer 5
        self.ff5 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        
        # Output Layer
        self.output = nn.Linear(64, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def _log_2_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log2(x + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._log_2_transform(x)
        residual_input = x
        
        # Feed Forward Layer 1 (Inputs)
        x = self.ff1(x)
        x = self.relu1(x)
        residual_ff1 = x
        
        # Feed Forward Layer 2 (incl residual connections from input layer)
        x = self.ff2(torch.cat([x, residual_input], dim=1))
        x = self.relu2(x)
        residual_ff2 = x
        
        # Feed Forward Layer 3 (incl residual connections from input and FF1)
        x = self.ff3(torch.cat([x, residual_input, residual_ff1], dim=1))
        x = self.relu3(x)
        
        # Feed Forward Layer 4 (incl residual connections from input, FF1 and FF2)
        x = self.ff4(torch.cat([x, residual_input, residual_ff1, residual_ff2], dim=1))
        x = self.relu4(x)
        
        # Feed Forward Layer 5
        x = self.ff5(x)
        x = self.relu5(x)
        
        # Output layer (no residual connection needed here)
        x = self.output(x)
        return x