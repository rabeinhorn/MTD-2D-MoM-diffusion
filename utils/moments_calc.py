import numpy as np
import multiprocessing as mp
from utils.micrograph_generation import generate_clean_micrograph


def calc_M2_micrographs_wrap(L, sigma2, gamma, X, W, N, NumMicrographs):
    """
    Calculate first and second moments for multiple micrographs in parallel.
    This function wraps the calculation of first and second moments for multiple
    micrographs using parallel processing to improve performance.

    Args:
        L : Image dimension (L x L).
        sigma2 : Noise variance of the measurements.
        gamma : Density factor.
        X : Image to be placed in the micrographs.
        W : Width of reserved area around Image instance in micrograph.
        N : Measurement dimension (N x N).
        NumMicrographs : Number of micrographs (i.e. measurements).

    Returns:
        tuple
            M1_ys : First moments for each micrograph.
            M2_ys : Second moments for each micrograph.

    Notes
    -----
    The function uses multiprocessing to parallelize calculations across available CPU cores,
    using approximately 1/3 of available cores for optimal performance.
    """
    M1_ys = np.zeros((NumMicrographs,))
    M2_ys = np.zeros((NumMicrographs, L, L))
    M3_ys = np.zeros((NumMicrographs, L, L, L, L))

    num_cpus = mp.cpu_count()
    sub_cpus = int(num_cpus / 3)
    pool = mp.Pool(sub_cpus)
    Ms_parallel = pool.starmap(calc_M2_micrograph, [[L, X, W, N, gamma, sigma2, ii] for ii in range(NumMicrographs)])
    pool.close()
    pool.join()

    for ii in range(NumMicrographs):
        M1_ys[ii] = Ms_parallel[ii][0]
        M2_ys[ii, :, :] = Ms_parallel[ii][1]

    return M1_ys, M2_ys


def calc_M2_micrograph(L, X, W, N, gamma, sigma2, sd):
    """
    Generate a clean micrograph, add noise, and calculate its first and second moments.

    Args:
        L : Image dimension (L x L).
        X : Image to be placed in the micrographs.
        W : Width of reserved area around Image instance in micrograph.
        N : Micrograph dimension (N x N).
        gamma : Density factor.
        sigma2 : Noise variance of the micrograph.
        sd : Random seed.

    Returns:
        tuple: A pair (M1_y, M2_y) where:
            - M1_y : First moment of the micrograph.
            - M2_y : Second moment of the micrograph.
    """
    y_clean, _, _ = generate_clean_micrograph(X, W, L, N, gamma * (N / L) ** 2, seed=sd)
    y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    yy = np.zeros((N, N, 1))
    yy[:, :, 0] = y

    M1_y = np.mean(y)

    M2_y = np.zeros((L, L))

    for i1 in range(L):
        for j1 in range(L):
            M2_y[i1, j1] = M2_2d(yy, (i1, j1))


    return M1_y, M2_y


def calc_M3_micrographs_wrap(L, sigma2, gamma, X, W, N, NumMicrographs):
    """
    Calculate up to the third moment for multiple micrographs in parallel.
    This function wraps the calculation the moments for multiple
    micrographs using parallel processing to improve performance.

    Args:
        L : Image dimension (L x L).
        sigma2 : Noise variance of the measurements.
        gamma : Density factor.
        X : Image to be placed in the micrographs.
        W : Width of reserved area around Image instance in micrograph.
        N : Measurement dimension (N x N).
        NumMicrographs : Number of micrographs (i.e. measurements).

    Returns:
        tuple
            M1_ys : First moments for each micrograph.
            M2_ys : Second moments for each micrograph.
            M3_ys : Third moments for each micrograph.

    Notes
    -----
    The function uses multiprocessing to parallelize calculations across available CPU cores,
    using approximately 1/3 of available cores for optimal performance.
    """
    M1_ys = np.zeros((NumMicrographs,))
    M2_ys = np.zeros((NumMicrographs, L, L))
    M3_ys = np.zeros((NumMicrographs, L, L, L, L))

    num_cpus = mp.cpu_count()
    sub_cpus = int(num_cpus / 3)
    pool = mp.Pool(sub_cpus)
    Ms_parallel = pool.starmap(calc_M3_micrograph, [[L, X, W, N, gamma, sigma2, ii] for ii in range(NumMicrographs)])
    pool.close()
    pool.join()

    for ii in range(NumMicrographs):
        M1_ys[ii] = Ms_parallel[ii][0]
        M2_ys[ii, :, :] = Ms_parallel[ii][1]
        M3_ys[ii, :, :, :, :] = Ms_parallel[ii][2]

    return M1_ys, M2_ys, M3_ys


def calc_M3_micrograph(L, X, W, N, gamma, sigma2, sd):
    """
    Generate a clean micrograph, add noise, and calculate up to its third moment.

    Args:
        L : Image dimension (L x L).
        X : Image to be placed in the micrographs.
        W : Width of reserved area around Image instance in micrograph.
        N : Micrograph dimension (N x N).
        gamma : Density factor.
        sigma2 : Noise variance of the micrograph.
        sd : Random seed.

    Returns:
        tuple: A triplet (M1_y, M2_y, M3_y) where:
            - M1_y : First moment of the micrograph.
            - M2_y : Second moment of the micrograph.
            - M3_y : Third moment of the micrograph.
    """
    y_clean, _, _ = generate_clean_micrograph(X, W, L, N, gamma * (N / L) ** 2, seed=sd)
    y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    yy = np.zeros((N, N, 1))
    yy[:, :, 0] = y

    M1_y = np.mean(y)

    M2_y = np.zeros((L, L))

    for i1 in range(L):
        for j1 in range(L):
            M2_y[i1, j1] = M2_2d(yy, (i1, j1))


    M3_y = np.zeros((L, L, L, L))

    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    M3_y[i1, j1, i2, j2] = M3_2d(yy, (i1, j1), (i2, j2))


    return M1_y, M2_y, M3_y


def calc_M3_signal_wrap(L, X):
    """
    Calculate first, second and third order moments of a given signal and the corresponding gradients using parallel processing.
    This function wraps the calculation of moments by distributing the workload across multiple CPU cores.
    It checks for available CPU cores and terminates if insufficient resources are available.

    Args:
        L : Image dimension (L x L).
        X : Input image.

    Returns:
        M1_x : First order moment (mean) of the signal.
        M2_x : Second order moment matrix of shape (L, L)
        M3_x : Third order moment tensor of shape (L, L, L, L)
        g1_x : First order gradient vector of length L^2
        g2_x : Second order gradient tensor of shape (L, L, L^2)
        g3_x : Third order gradient tensor of shape (L, L, L, L, L^2)
    """
    num_cpus = mp.cpu_count()
    if num_cpus < L:
        print(" Not enough CPUs available, terminate program")
        exit(0)
    pool = mp.Pool(L)

    all_parallel = pool.starmap(calc_M3_signal, [[L, X, i1] for i1 in range(L)])
    pool.close()
    pool.join()

    M1_x = np.mean(X)
    M2_x = np.reshape(np.concatenate([M2 for M2, M3, g2, g3 in all_parallel]),(L, L))
    M3_x = np.reshape(np.concatenate([M3 for M2, M3, g2, g3 in all_parallel]), (L, L, L, L))
    signal_length = L ** 2
    g1_x = np.full(signal_length, 1 / signal_length)
    g2_x = np.reshape(np.concatenate([g2 for M2, M3, g2, g3 in all_parallel]), (L, L, signal_length))
    g3_x = np.reshape(np.concatenate([g3 for M2, M3, g2, g3 in all_parallel]), (L, L, L, L, signal_length))

    return M1_x, M2_x, M3_x, g1_x, g2_x, g3_x


def calc_M2_signal_wrap(L, X):
    """
    Calculate first and second moments of a given signal and the corresponding gradients using parallel processing.
    This function wraps the calculation of moments by distributing the workload across multiple CPU cores.
    It checks for available CPU cores and terminates if insufficient resources are available.

    Args:
        L : Image dimension (L x L).
        X : Input image.

    Returns:
        M1_x : First order moment (mean) of the signal.
        M2_x : Second order moment matrix of shape (L, L)
        g1_x : First order gradient vector of length L^2
        g2_x : Second order gradient tensor of shape (L, L, L^2)
    """
    num_cpus = mp.cpu_count()
    if num_cpus < L:
        print(" Not enough CPUs available, terminate program")
        exit(0)
    pool = mp.Pool(L)

    all_parallel = pool.starmap(calc_M2_signal, [[L, X, i1] for i1 in range(L)])
    pool.close()
    pool.join()

    M1_x = np.mean(X)
    M2_x = np.reshape(np.concatenate([M2 for M2, g2 in all_parallel]),(L, L))

    signal_length = L ** 2
    g1_x = np.full(signal_length, 1 / signal_length)
    g2_x = np.reshape(np.concatenate([g2 for M2, g2 in all_parallel]), (L, L, signal_length))

    return M1_x, M2_x, g1_x, g2_x


def calc_M3_signal(L, X, i1):
    """
    Calculate the second and third moments of a given signal and their gradients.

    Args:
        L : Image dimension (L x L).
        X : Input image (i.e. signal).
        i1 : Fixed index for moment calculations.

    Returns:
        tuple:
            - M2_x : Second order moments array of shape (L,)
            - M3_x : Third order moments array of shape (L,L,L)
            - gS2_x : Second order moment gradients of shape (L, L^2)
            - gS3_x : Third order moment gradients of shape (L,L,L,L^2)
    """
    xx = np.zeros((L, L, 1))
    xx[:, :, 0] = X

    M2_x = np.zeros((L))

    for j1 in range(L):
        M2_x[j1] = M2_2d(xx, (i1, j1))

    M3_x = np.zeros((L, L, L))

    for j1 in range(L):
        for i2 in range(L):
            for j2 in range(L):
                M3_x[j1, i2, j2] = M3_2d(xx, (i1, j1), (i2, j2))

    signal_length = L ** 2

    gS2_x = np.zeros((L, signal_length))

    for j1 in range(L):
        gS2_x[j1, :] = calcS2_x_grad(X, L, (i1, j1))

    gS2_x = gS2_x / signal_length

    gS3_x = np.zeros((L, L, L, signal_length))

    for j1 in range(L):
        for i2 in range(L):
            for j2 in range(L):
                gS3_x[j1, i2, j2, :] = calcS3_x_grad(X, L, (i1, j1), (i2, j2))

    gS3_x = gS3_x / signal_length

    return M2_x, M3_x, gS2_x, gS3_x


def calc_M2_signal(L, X, i1):
    """
    Calculate the second moment of a given signal and its gradient.

    Args:
        L : Image dimension (L x L).
        X : Input image (i.e. signal).
        i1 : Fixed index for moment calculations.

    Returns:
        tuple:
            - M2_x : Second order moments array of shape (L,)
            - gS2_x : Second order moment gradients of shape (L, L^2)
    """
    xx = np.zeros((L, L, 1))
    xx[:, :, 0] = X

    M2_x = np.zeros((L))

    for j1 in range(L):
        M2_x[j1] = M2_2d(xx, (i1, j1))

    signal_length = L ** 2

    gS2_x = np.zeros((L, signal_length))

    for j1 in range(L):
        gS2_x[j1, :] = calcS2_x_grad(X, L, (i1, j1))

    gS2_x = gS2_x / signal_length

    return M2_x, gS2_x


def M2_2d(A, shift1):
    """ Calculate second-order autocorrelation of A for shift1.

    Args:
        A: Input image.
        shift1: a tuple containing the shift.
    Returns:
        second-order autocorrelation of A for shift1
    """
    dim1, dim2, _ = np.shape(A)

    shift1y = -shift1[0]
    shift1x = -shift1[1]

    valsy1 = [0, -shift1y]
    valsx1 = [0, -shift1x]

    rangey = list(range(max(valsy1), dim1 + min(valsy1)))
    rangey1 = [x + shift1y for x in rangey]
    rangex = list(range(max(valsx1), dim2 + min(valsx1)))
    rangex1 = [x + shift1x for x in rangex]

    return np.sum(A[min(rangey):max(rangey) + 1, min(rangex):max(rangex) + 1, :] * A[min(rangey1):max(rangey1) + 1,
                                                                                   min(rangex1):max(rangex1) + 1, :],
                  axis=(0, 1)) / (dim1 * dim2)


def M3_2d(A, shift1, shift2):
    """ Calculate third-order autocorrelation of A for shift1, shift2.

    Args:
        A: Input image.
        shift1, shift2: tuples containing the shifts.

    Returns:
        third-order autocorrelation of A for shift1, shift2
    """
    dim1, dim2, _ = np.shape(A)

    shift1y = -shift1[0]
    shift1x = -shift1[1]

    shift2y = -shift2[0]
    shift2x = -shift2[1]

    valsy1 = [0, -shift1y, -shift2y]
    valsx1 = [0, -shift1x, -shift2x]

    rangey = list(range(max(valsy1), dim1 + min(valsy1)))
    if rangey == []: return 0
    rangey1 = [x + shift1y for x in rangey]
    rangey2 = [x + shift2y for x in rangey]

    rangex = list(range(max(valsx1), dim2 + min(valsx1)))
    if rangex == []: return 0
    rangex1 = [x + shift1x for x in rangex]
    rangex2 = [x + shift2x for x in rangex]

    return np.sum(A[min(rangey):max(rangey) + 1, min(rangex):max(rangex) + 1, :] * A[min(rangey1):max(rangey1) + 1,
                                                                                   min(rangex1):max(rangex1) + 1,
                                                                                   :] * A[min(rangey2):max(rangey2) + 1,
                                                                                        min(rangex2):max(rangex2) + 1,
                                                                                        :], axis=(0, 1)) / (dim1 * dim2)


def calcS2_x_grad(X, L, shift1):
    """ Calculate the gradient of the second-order autocorrelation of X for shift1.

    Args:
        X: Input image.
        shift1: a tuple containing the shift.
    Returns:
        Gradient of the second-order autocorrelation of X for shift1
    """
    shift1y = shift1[0]
    shift1x = shift1[1]

    rangey = list(range(shift1y, L))
    rangey1 = [i - shift1y for i in rangey]
    rangex = list(range(shift1x, L))
    rangex1 = [i - shift1x for i in rangex]

    T1 = np.zeros((L, L))
    T2 = np.zeros((L, L))

    T1[min(rangey):max(rangey) + 1, min(rangex):max(rangex) + 1] = X[min(rangey1):max(rangey1) + 1,
                                                                      min(rangex1):max(rangex1) + 1]
    T2[min(rangey1):max(rangey1) + 1, min(rangex1):max(rangex1) + 1] = X[min(rangey):max(rangey) + 1,
                                                                   min(rangex):max(rangex) + 1]
    gS2_x = T1 + T2

    return gS2_x.flatten()


def calcS3_x_grad(X, L, shift1, shift2):
    """ Calculate the gradient of the third-order autocorrelation of X for shift1, shift2.

    Args:
        X: Input image.
        shift1, shift2: tuples containing the shifts.

    Returns:
        Gradient of the third-order autocorrelation of X for shift1, shift2
    """
    shift1y = shift1[0]
    shift1x = shift1[1]
    shift2y = shift2[0]
    shift2x = shift2[1]

    shifts_y = [shift1y, shift2y]
    shifts_x = [shift1x, shift2x]

    rangey = list(range(max(shifts_y), L))
    rangey1 = [i - shift1y for i in rangey]
    rangey2 = [i - shift2y for i in rangey]
    rangex = list(range(max(shifts_x), L))
    rangex1 = [i - shift1x for i in rangex]
    rangex2 = [i - shift2x for i in rangex]

    T1 = np.zeros((L, L))
    T2 = np.zeros((L, L))
    T3 = np.zeros((L, L))

    T1[min(rangey):max(rangey) + 1, min(rangex):max(rangex) + 1] = X[min(rangey1):max(rangey1) + 1,min(rangex1):max(rangex1) + 1] * \
                                                                   X[min(rangey2):max(rangey2) + 1,min(rangex2):max(rangex2) + 1]
    T2[min(rangey1):max(rangey1) + 1, min(rangex1):max(rangex1) + 1] = X[min(rangey):max(rangey) + 1,min(rangex):max(rangex) + 1] * \
                                                                       X[min(rangey2):max(rangey2) + 1,min(rangex2):max(rangex2) + 1]
    T3[min(rangey2):max(rangey2) + 1, min(rangex2):max(rangex2) + 1] = X[min(rangey):max(rangey) + 1,min(rangex):max(rangex) + 1] * \
                                                                       X[min(rangey1):max(rangey1) + 1,min(rangex1):max(rangex1) + 1]
    gS3_x = T1 + T2 + T3

    return gS3_x.flatten()