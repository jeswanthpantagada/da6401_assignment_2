import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Custom Dropout using inverted dropout scaling.
    Does NOT use nn.Dropout or F.dropout anywhere.

    Training mode:
        mask ~ Bernoulli(1 - p)
        output = input * mask / (1 - p)   ← inverted dropout scaling

    Eval mode (self.training = False):
        output = input unchanged (deterministic)

    Args:
        p (float): probability of zeroing an element. Must be in [0, 1).
    """
    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f'p must be in [0,1), got {p}')
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0:
            keep_prob = 1.0 - self.p
            # Sample binary mask: 1 with prob keep_prob, 0 with prob p
            mask = torch.bernoulli(
                torch.full(x.shape, keep_prob,
                           device=x.device, dtype=x.dtype)
            )
            # Apply mask and scale (inverted dropout)
            x = x * mask / keep_prob
        return x

    def extra_repr(self) -> str:
        return f'p={self.p}'
