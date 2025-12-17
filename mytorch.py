# 2025/12/4
# zhangzhong
# impl the pytorch utils by ourselves.


from torch import nn, Tensor
import torch


class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features) -> None:
        super().__init__()

        # define a W matirx and B matrix
        self.W = nn.Parameter(torch.randn(size=(in_features, out_features)))
        # self.B = # TODO
        self.B = nn.Parameter(torch.randn(size=(out_features,)))
        # self.B = torch.randn(size=(out_features,))

    def forward(self, input: Tensor) -> Tensor:
        output = input @ self.W + self.B
        return output


class MyMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # TODO:
        return torch.mean((input - target) ** 2)


class MyReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        # TODO:
        zero = torch.zeros_like(x)
        # element-wise
        return torch.max(x, zero)


# https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html
def cross_entropy_logsumexp(logits, labels):
    """
    logits: (N, C)
    labels: (N,) integer class indices
    """
    N = logits.shape[0]
    log_sum_exp = torch.logsumexp(logits, dim=1)  # stable
    log_probs = logits[torch.arange(N), labels] - log_sum_exp
    loss = -log_probs.mean()
    return loss


def dropout(X: torch.Tensor, dropout: float) -> Tensor:
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X, device=X.device)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape, device=X.device) > dropout).float()
    return mask * X / (1.0 - dropout)
