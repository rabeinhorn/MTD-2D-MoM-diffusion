import torch

def denoising_score_matching(score_network, x, eps=1e-4):
    # Sample time
    t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device) * (1 - eps) + eps

    # Compute log posterior terms
    int_beta = (0.1 + 0.5 * (20 - 0.1) * t) * t
    mu_t = x * torch.exp(-0.5 * int_beta)
    var_t = -torch.expm1(-int_beta)
    x_t = torch.randn_like(x) * var_t ** 0.5 + mu_t
    grad_log_p = -(x_t - mu_t) / var_t

    # Compute loss
    score = score_network(x_t, t)
    loss = (score - grad_log_p) ** 2
    lambda_t = var_t
    weighted_loss = lambda_t * loss
    return torch.mean(weighted_loss)