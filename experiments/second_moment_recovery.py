import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.moments_calc import calc_M2_micrographs_wrap
from utils.gradient_descent_opt import run_nag_second_moment_prior, run_nag_second_moment


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--N', type=int, default=4000, help='Micrograph width')
    parser.add_argument('--num_micro', type=int, default=10, help='Number of micrographs')
    parser.add_argument('--gamma', type=float, default=0.1,  help='Density factor')
    parser.add_argument('--snr', type=float, default=0.5, help='SNR of measurement')
    parser.add_argument('--num_iter', type=int, default=4000, help='Number of iterations')
    parser.add_argument('--lr', type=float, default=18000, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.993, help='momentum')
    parser.add_argument("--disable_prior", action="store_true", help="Disable score prior (default is enabled)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    X_gt = plt.imread(f"../images/mnist_digit_3_crop_14X14.png")

    L = np.shape(X_gt)[0]
    W = 2 * L - 1

    sigma2 = np.linalg.norm(X_gt) ** 2 / (args.snr * np.pi * (L // 2) ** 2)

    print("Simulate micrographs and calculate their moments")
    M1_ys, M2_ys = calc_M2_micrographs_wrap(L, sigma2, args.gamma, X_gt, W, args.N, args.num_micro)
    sum_M1_ys = np.sum(M1_ys)
    sum_M2_ys = np.sum(M2_ys, 0)

    M1_y = sum_M1_ys / args.num_micro
    M2_y = sum_M2_ys / args.num_micro

    print("Solve optimization problem")
    X_initial = np.random.rand(L, L)
    X_initial = np.linalg.norm(X_gt) * X_initial / np.linalg.norm(X_initial)
    X_initial = X_initial.flatten()  # initial guess should be a 1-D array
    if args.disable_prior:
        estimate = run_nag_second_moment(X_initial, M1_y, M2_y, L, sigma2, args.gamma, lr=args.lr,
                                         momentum=args.momentum, max_iter=args.num_iter)
    else:
        estimate = run_nag_second_moment_prior(X_initial, M1_y, M2_y, L, sigma2, args.gamma, lr=args.lr,
                                               momentum=args.momentum, max_iter=args.num_iter)

    X_est = np.reshape(estimate, (L, L))  # X recovered is passed as a 1-D array
    X_E = np.linalg.norm(X_est - X_gt) / np.linalg.norm(X_gt)
    print("estimation error is ", X_E)


if __name__ == "__main__":
    main()