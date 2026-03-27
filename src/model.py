import torch
import torch.nn as nn

class SignalClassifier(nn.Module):
    """
    Feedforward neural network for Buy/Hold/Sell classification.
    Architecture matches Section IV-A of the paper:
        Input -> FC(64, ReLU) -> Dropout(0.3) -> FC(32, ReLU) -> FC(3)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
