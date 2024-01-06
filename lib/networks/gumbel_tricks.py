import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
# Borrowed from this gist
# https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def st_gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


# Pytorch official version
def gumbel_softmax(logits, tau=1., hard=False, eps=1e-10, dim=-1):
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def topk_fastv1(input, k, dim=None):
    sort, idx = input.sort(descending=True, dim=dim)
    return sort[:k], idx[:k]

# Pytorch official version
def gumbel_soft_topk(logits, k, tau=1., hard=False, eps=1e-10, dim=-1):
    r"""
    A generalization of gumbel softmax.
    When k = 1, this function is equivalent to gumbel softmax
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels[gumbels == float('inf')] = 0
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        # pdb.set_trace()
        index = y_soft.topk(k, dim=dim)[1]
        # index = topk_fastv1(y_soft, k, dim=dim)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    if torch.isnan(ret).sum() > 0:
        pdb.set_trace()
    return ret

# Pytorch official version
def soft_topk(logits, k, tau=1., hard=False, eps=1e-10, dim=-1):
    r"""
    A generalization of gumbel softmax.
    When k = 1, this function is equivalent to gumbel softmax
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = logits / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        # pdb.set_trace()
        index = y_soft.topk(k, dim=dim)[1]
        # index = topk_fastv1(y_soft, k, dim=dim)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

if __name__ == "__main__":
    rand_mat = torch.rand(4,8)
    rand_mat2 = torch.stack([rand_mat, 1-rand_mat], dim=-1)
    ret_max = gumbel_softmax(rand_mat2, hard=True)
    ret_topk = gumbel_soft_topk(rand_mat, k=3, hard=True)
    
    pdb.set_trace()