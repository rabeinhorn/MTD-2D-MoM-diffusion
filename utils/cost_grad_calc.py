import numpy as np
from utils.moments_calc import calc_M3_signal_wrap, calc_M2_signal_wrap


def cost_grad_calc_third_moment(X, M1_y_given, M2_y_given, M3_y_given, sigma2, L, N_mat, gamma):
    """
    Calculate cost function and its gradient using first, second and third moments.

    Args:
        X : Input signal as 1-D array.
        M1_y_given : First moment of the measurements.
        M2_y_given : Second moment of the measurements.
        M3_y_given : Third moment of the measurements.
        sigma2 : Noise variance of the measurements.
        L : Image dimension (L x L).
        N_mat : sparse matrix of the influence of the noise.
        gamma : Density factor.

    Returns:
        tuple: (f_tot, g_tot) where
                f_tot : cost function value
                g_tot : gradient of the cost function
    """
    X = np.reshape(X, (L, L))  # X is passed as a 1-D array
    signal_length = L ** 2
    M1_x_calc, M2_x_calc, M3_x_calc, gM1_x, gM2_x, gM3_x = calc_M3_signal_wrap(L, X)
    gM2_x = np.reshape(gM2_x, (L ** 2, signal_length))
    gM3_x = np.reshape(gM3_x, (L ** 4, signal_length))
    w1 = 1 / 2
    w2 = 1 / (2 * L ** 2)
    w3 = 1 / (2 * L ** 4)

    R1 = gamma * M1_x_calc - M1_y_given
    R2 = gamma * M2_x_calc - M2_y_given
    R2[0, 0] += sigma2 # Add noise element

    M3_x_calc += M1_x_calc * sigma2 * np.reshape(N_mat.toarray(), np.shape(M3_x_calc)) # Add noise element
    R3 = gamma * M3_x_calc - M3_y_given

    # %% cost and grad functions calculation
    f1 = w1 * R1 ** 2
    f2 = w2 * np.sum(R2 ** 2)
    f3 = w3 * np.sum(R3 ** 2)
    f_tot = f1 + f2 + f3

    g_1 = 2 * w1 * gamma * gM1_x * R1
    g_2 = 2 * w2 * gamma * R2.flatten() @ gM2_x
    g_3 = 2 * w3 * gamma * R3.flatten() @ gM3_x
    g_tot = g_1 + g_2 + g_3

    return f_tot, g_tot


def cost_grad_calc_second_moment(X, M1_y_given, M2_y_given, sigma2, L, gamma):
    """
    Calculate cost and gradient using first and second-order moments.

    Args:
        X : Input signal as 1-D array
        M1_y_given : First moment of the measurements.
        M2_y_given : Second moment of the measurements.
        sigma2 : Noise variance of the measurements.
        L : Image dimension (L x L)
        gamma : Density factor.

    Returns:
        tuple: (f_tot, g_tot) where
            f_tot : cost function value
            g_tot : gradient of the cost function
    """
    X = np.reshape(X, (L, L))  # X is passed as a 1-D array
    signal_length = L ** 2
    M1_x_calc, M2_x_calc, gM1_x, gM2_x = calc_M2_signal_wrap(L, X)
    gM2_x = np.reshape(gM2_x, (L ** 2, signal_length))
    w1 = 1 / 2
    w2 = 1 / (2 * L ** 2)

    R1 = gamma * M1_x_calc - M1_y_given
    R2 = gamma * M2_x_calc - M2_y_given
    R2[0, 0] += sigma2  # Add noise element

    # %% cost and grad functions calculation
    f1 = w1 * R1 ** 2
    f2 = w2 * np.sum(R2 ** 2)
    f_tot = f1 + f2

    g_1 = 2 * w1 * gamma * gM1_x * R1
    g_2 = 2 * w2 * gamma * R2.flatten() @ gM2_x
    g_tot = g_1 + g_2

    return f_tot, g_tot