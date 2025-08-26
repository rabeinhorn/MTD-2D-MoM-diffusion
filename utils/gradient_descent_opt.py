import numpy as np
from utils.cost_grad_calc import cost_grad_calc_third_moment, cost_grad_calc_second_moment
import torch
from utils.prior_models import UnetScoreNetwork, UnetScoreNetwork_SR
import scipy.sparse as sp


def calcN_mat(L):
    """ Calculate matrix of the influence of the noise (1s and 0s according to the deltas in the proposition in the paper)
    Args:
        L: diameter of the target image

    Returns:
        sparse matrix of the influence of the noise
    """
    N_mat_out = np.zeros((L, L, L, L))
    for ii in range(L):
        for jj in range(L):
            N_mat_out[ii, jj, ii, jj] += 1
    N_mat_out[0, 0, :, :] += 1
    N_mat_out[:, :, 0, 0] += 1

    return sp.csr_matrix(np.reshape(N_mat_out, (L ** 4, 1)))


def run_nag_second_moment(X_initial, M1_y, M2_y, L, sigma2, gamma, lr, momentum, max_iter):
    """
    Run Nesterov Accelerated Gradient descent (NAG) optimization for image recovery from up to
    the second moment of the measurements.
    Args:
        X_initial : Initial guess for the recovered image (passed as a 1-D array).
        M1_y : First moment of the measurements.
        M2_y : Second moment of the measurements.
        L : Image dimension (L x L)
        sigma2 : Noise variance of the measurements.
        gamma : Density factor.
        lr : Learning rate for gradient descent.
        momentum : Momentum coefficient for NAG.
        max_iter : Maximum number of iterations.
    Returns:
        The estimated image as a 1-D array.
    """
    X_updated = X_initial.copy()
    v_t = 0
    for it in range(max_iter):
        cost_val, grad_val = cost_grad_calc_second_moment(X_updated + momentum * v_t, M1_y, M2_y, sigma2, L, gamma)
        v_t = momentum * v_t - lr * grad_val
        X_updated = X_updated + v_t

    return X_updated


def run_nag_second_moment_prior(X_initial, M1_y, M2_y, L, sigma2, gamma, lr, momentum, max_iter):
    """
    Runs Nesterov Accelerated Gradient descent (NAG) optimization for image recovery from up to
    the second moment of the measurements. The function performs optimization using a pre-trained
    score network as prior.
    Parameters:
        X_initial: Initial guess for the recovered image (passed as a 1-D array).
        M1_y: First moment of the measurements.
        M2_y: Second moment of the measurements.
        L: Image dimension (L x L)
        sigma2: Noise variance of the measurements.
        gamma: Density factor.
        lr: Learning rate for gradient descent.
        momentum: Momentum coefficient for NAG.
        max_iter: Maximum number of iterations
    Returns:
       The estimated image as a 1-D array.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device: " + str(device))

    # Initialize network
    score_network = UnetScoreNetwork()
    checkpoint_path = "../checkpoints/mnist_prior_14X14.pth"

    # Load prior model weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    score_network.load_state_dict(state_dict)
    score_network = score_network.to(device)
    score_network.eval()

    X_updated = X_initial.copy()
    v_t = 0
    for it in range(max_iter):
        cost_val, grad_val = cost_grad_calc_second_moment(X_updated + momentum * v_t, M1_y, M2_y, sigma2, L, gamma)
        grad_norm = np.linalg.norm(grad_val)
        score_in = X_updated + momentum * v_t
        score_cur = score_network(torch.from_numpy(score_in.flatten()).unsqueeze(0).float().to(device), device=device).detach().cpu().numpy()
        score_norm = np.linalg.norm(score_cur.flatten())
        scaled_score = (grad_norm * score_cur.flatten() / score_norm)

        v_t = momentum * v_t - lr * (grad_val - scaled_score)
        X_updated = X_updated + v_t

    return X_updated


def run_nag_third_moment(X_initial, M1_y, M2_y, M3_y, L, sigma2, gamma, lr, momentum, max_iter):
    """
    Run Nesterov Accelerated Gradient descent (NAG) optimization for image recovery from up to
    the third moment of the measurements.
    Args:
        X_initial : Initial guess for the recovered image (passed as a 1-D array).
        M1_y : First moment of the measurements.
        M2_y : Second moment of the measurements.
        M3_y : Third moment of the measurements.
        L : Image dimension (L x L)
        sigma2 : Noise variance of the measurements.
        gamma : Density factor.
        lr : Learning rate for gradient descent.
        momentum : Momentum coefficient for NAG.
        max_iter : Maximum number of iterations.
    Returns:
        The estimated image as a 1-D array.
    """
    X_updated = X_initial.copy()
    N_mat = calcN_mat(L)
    v_t = 0
    for it in range(max_iter):
        cost_val, grad_val = cost_grad_calc_third_moment(X_updated + momentum * v_t, M1_y, M2_y,
                                                                             M3_y, sigma2, L, N_mat, gamma)
        v_t = momentum * v_t - lr * grad_val
        X_updated = X_updated + v_t

    return X_updated


def run_nag_third_moment_prior(X_initial, M1_y, M2_y, M3_y, L, sigma2, gamma, lr, momentum, max_iter):
    """
    Runs Nesterov Accelerated Gradient descent (NAG) optimization for image recovery from up to
    the third moment of the measurements. The function performs optimization using a pre-trained
    score network as prior.
    Parameters:
        X_initial: Initial guess for the recovered image (passed as a 1-D array).
        M1_y: First moment of the measurements.
        M2_y: Second moment of the measurements.
        M3_y: Third moment of the measurements.
        L: Image dimension (L x L)
        sigma2: Noise variance of the measurements.
        gamma: Density factor.
        lr: Learning rate for gradient descent.
        momentum: Momentum coefficient for NAG.
        max_iter: Maximum number of iterations
    Returns:
       The estimated image as a 1-D array.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device: " + str(device))

    # Initialize network
    score_network = UnetScoreNetwork()
    checkpoint_path = "../checkpoints/mnist_prior_14X14.pth"

    # Load prior model weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    score_network.load_state_dict(state_dict)
    score_network = score_network.to(device)
    score_network.eval()

    X_updated = X_initial.copy()
    N_mat = calcN_mat(L)
    v_t = 0
    for it in range(max_iter):
        cost_val, grad_val = cost_grad_calc_third_moment(X_updated + momentum * v_t, M1_y, M2_y, M3_y, sigma2, L, N_mat, gamma)
        grad_norm = np.linalg.norm(grad_val)
        score_in = X_updated + momentum * v_t
        score_cur = score_network(torch.from_numpy(score_in.flatten()).unsqueeze(0).float().to(device), device=device).detach().cpu().numpy()
        score_norm = np.linalg.norm(score_cur.flatten())
        scaled_score = (grad_norm * score_cur.flatten() / score_norm)

        v_t = momentum * v_t - lr * (grad_val - scaled_score)
        X_updated = X_updated + v_t


    return X_updated


def run_nag_sr(X_initial, M1_y, M2_y, M3_y, L_high, L_low, sigma2, gamma, lr, momentum, max_iter):
    """
    Runs Nesterov's Accelerated Gradient descent (NAG) optimization for super resolution image recovery from up to
    the third moment of the measurements.
    Parameters:
        X_initial: Initial guess for the recovered image (passed as a 1-D array).
        M1_y: First-order moment of the measurements
        M2_y: Second-order moment of the measurements
        M3_y: Third-order moment of the measurements
        L_high: High-resolution image dimension (L_high x L_high)
        L_low: Low-resolution image dimension (L_low x L_low)
        sigma2: Noise variance of the measurements.
        gamma: Density factor.
        lr: Learning rate
        momentum: Momentum coefficient for NAG
        max_iter: Maximum number of iterations
    Returns:
        X_updated_high: Optimized solution at high resolution
    """
    # construct sampling matrix
    base_mat = np.zeros((int(L_high / L_low), int(L_high / L_low)))
    base_mat[0, 0] = 1
    sample_mat = np.kron(np.ones((L_low, L_low)), base_mat)
    mask_sample = sample_mat.flatten() == 1

    X_updated_high = X_initial.copy()
    N_mat = calcN_mat(L_low)
    v_t =  np.zeros_like(X_updated_high)
    for it in range(max_iter):
        X_updated_low = X_updated_high[mask_sample]
        v_t_low = v_t[mask_sample]
        obj_func_in = X_updated_low + momentum * v_t_low
        cost_val, grad_val_low = cost_grad_calc_third_moment(obj_func_in, M1_y, M2_y,
                                                                             M3_y, sigma2, L_low, N_mat, gamma)

        grad_val_high = np.zeros_like(X_updated_high)
        grad_val_high[mask_sample] = grad_val_low
        v_t = momentum * v_t - lr * grad_val_high
        X_updated_high = X_updated_high + v_t

    return X_updated_high


def run_nag_sr_prior(X_initial,
                       M1_y, M2_y, M3_y, L_high, L_low, sigma2, gamma, lr, momentum, score_factor, max_iter):
    """
    Runs Nesterov's Accelerated Gradient descent (NAG) optimization for super resolution image recovery from up to
    the third moment of the measurements. The function performs optimization using a pre-trained score network as prior.
    Parameters:
        X_initial: Initial guess for the recovered image (passed as a 1-D array).
        M1_y: First-order moment of the measurements
        M2_y: Second-order moment of the measurements
        M3_y: Third-order moment of the measurements
        L_high: High-resolution image dimension (L_high x L_high)
        L_low: Low-resolution image dimension (L_low x L_low)
        sigma2: Noise variance of the measurements.
        gamma: Density factor.
        lr: Learning rate
        momentum: Momentum coefficient for NAG
        score_factor: Scaling factor for the score prior at non-sampled locations
        max_iter: Maximum number of iterations
    Returns:
        X_updated_high: Optimized solution at high resolution
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device: " + str(device))

    # Initialize network
    score_network = UnetScoreNetwork_SR()

    checkpoint_path = "../checkpoints/mnist_prior_28X28.pth"

    print("Load checkpoint")
    # Load prior model weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    score_network.load_state_dict(state_dict)
    score_network = score_network.to(device)
    score_network.eval()

    # construct sampling matrix
    base_mat = np.zeros((int(L_high / L_low), int(L_high / L_low)))
    base_mat[0, 0] = 1
    sample_mat = np.kron(np.ones((L_low, L_low)), base_mat)
    mask_sample = sample_mat.flatten() == 1
    mask_sample_coml = sample_mat.flatten() == 0 # the complementary samples

    X_updated_high = X_initial.copy()
    N_mat = calcN_mat(L_low)
    v_t =  np.zeros_like(X_updated_high)
    for it in range(max_iter):
        X_updated_low = X_updated_high[mask_sample]
        v_t_low = v_t[mask_sample]
        obj_func_in = X_updated_low + momentum * v_t_low
        cost_val, grad_val_low = cost_grad_calc_third_moment(obj_func_in, M1_y, M2_y,
                                                                             M3_y, sigma2, L_low, N_mat, gamma)
        grad_norm = np.linalg.norm(grad_val_low)
        score_in = X_updated_high + momentum * v_t
        score_cur = score_network(torch.from_numpy(score_in.flatten()).unsqueeze(0).float().to(device),
                                  device=device).detach().cpu().numpy()
        score_norm = np.linalg.norm(score_cur.flatten())

        scaled_score = np.zeros_like(X_updated_high)
        scaled_score[mask_sample] = (grad_norm * score_cur.flatten()[mask_sample] / score_norm)
        scaled_score[mask_sample_coml] = score_cur.flatten()[mask_sample_coml] * score_factor

        grad_val_high = np.zeros_like(X_updated_high)
        grad_val_high[mask_sample] = grad_val_low
        v_t = momentum * v_t - lr * (grad_val_high - scaled_score)
        X_updated_high = X_updated_high + v_t

    return X_updated_high
