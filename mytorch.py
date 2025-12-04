# 2025/12/4
# zhangzhong
# impl the pytorch utils by ourselves.

# TODO:
# 1. pytorch required grad
# 2. linear layer
# 3. activation function
# 4. MSE Loss
# 5. Softmax and cross entropy loss
# 6. use all mytorch to train a MLP model.

# Linear
class LinearRegressionScratch(nn.Module):
    # 这是我们的第一个模型
    def __init__(self, in_features: int) -> None:
        super().__init__()
        # 初始化参数
        # 默认使用N(0, 0.0.^2)
        # 但是模型的大小要怎么确定呢??
        # TODO: 你这个不对，初始化的参数的正态分布的方差应该是0.01
        # should be lazy also
        # 原来如此，Parameter是支持lazy的 这样我们的优化器也可以使用了 完美！
        self.w: nn.Parameter | None = None
        self.b: nn.Parameter | None = None
        self.in_features = in_features

        # default_device = mytorch.config.conf['device']
        # self.w = torch.randn((in_features, 1), requires_grad=True,
        #                      device=torch.device(default_device))
        # self.b = torch.randn((1,), requires_grad=True,
        #                      device=torch.device(default_device))
        # AttributeError: Can't pickle local object 'LinearRegressionScratch.__init__.<locals>.<lambda>'
        # self.net = lambda X: X @ self.w + self.b

    # def parameters(self):
    #     """
    #     Returns an iterator over module parameters.

    #     code example:
    #         for name, param in self.named_parameters(recurse=recurse):
    #             yield param
    #     """
    #     # 返回模型的参数
    #     # return self.w, self.b
    #     assert self.w is not None
    #     assert self.b is not None
    #     yield self.w
    #     yield self.b

    def forward(self, X: Tensor) -> Tensor:
        if self.w is None:
            self.w = nn.Parameter(
                torch.randn(
                    size=(self.in_features, 1), requires_grad=True, device=X.device
                )
            )
            self.b = nn.Parameter(
                torch.randn(size=(1,), requires_grad=True, device=X.device)
            )
        return X @ self.w + self.b


# MSELoss


# https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_hat, y):
        return torch.mean((y_hat - y) ** 2) / 2


# Activation Function

#


def dropout(X: torch.Tensor, dropout: float) -> Tensor:
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X, device=X.device)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape, device=X.device) > dropout).float()
    return mask * X / (1.0 - dropout)


def relu(x: Tensor) -> Tensor:
    zero = torch.zeros_like(x)
    return torch.max(x, zero)


# TODO: after refactor, rename this
dropout_layer = dropout
relu_layer = relu


def softmax(logits: Tensor):
    """
    logits: shape=(batch_size, num_labels)
    """
    exp_logits = torch.exp(logits)
    # 将每一行的exp求和，并且保持dim
    sumexp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
    # 保持dim才能保证这里的boardcasting是沿行扩展的
    p_matrix = exp_logits / sumexp_logits
    return p_matrix


# problems
# •	exp(logits) may overflow → inf
# •	Small exp values may underflow → 0
# •	log(probs) may produce log(0) → -inf
# •	The final loss may be NaN
def cross_entropy_naive(logits, labels):
    """
    logits: (N, C)
    labels: (N,) integer class indices
    """
    # step 1: softmax
    exp_logits = torch.exp(logits)  # VERY unstable
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)

    # step 2: negative log-likelihood
    N = logits.shape[0]
    log_probs = torch.log(probs[torch.arange(N), labels])
    loss = -log_probs.mean()
    return loss


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


# log sum exp trick
# 现在这个函数可以任意shape的向量 只要shape是一样的就行
def cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    assert logits.shape == labels.shape

    # step 1. max
    repeats = [1] * logits.dim()
    repeats[-1] = logits.shape[-1]
    c = logits.max(dim=-1, keepdim=True)[0].repeat(*repeats)
    assert logits.shape == c.shape

    # step 2. logsumexp
    y = logits - c
    logsumexp = y.exp().sum(dim=-1, keepdim=True).log().repeat(*repeats)
    assert logits.shape == logsumexp.shape

    # step 3. calculate
    return -torch.sum((logits - c - logsumexp) * labels, dim=-1)


# 像这种函数应该给一个device参数
def uniform_distribution(*shape: int, device: torch.device | None = None) -> Tensor:
    return torch.ones(shape, dtype=torch.float32, device=device) / shape[-1]


def cross_entropy_loss(
    logits: Tensor,
    labels: Tensor,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    valid_reductions = ["none", "mean", "sum"]
    if reduction not in valid_reductions:
        raise ValueError(f"invalid reduction: {reduction}")

    labels = one_hot(labels, num_classes=logits.shape[-1])
    assert logits.shape == labels.shape

    u = uniform_distribution(*logits.shape, device=logits.device)
    assert u.shape == logits.shape

    # label smoothing
    loss = (1.0 - label_smoothing) * cross_entropy(
        logits=logits, labels=labels
    ) + label_smoothing * cross_entropy(logits=logits, labels=u)

    # reduction: none, mean, sum
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
