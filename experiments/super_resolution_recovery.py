import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.moments_calc import calc_M3_micrographs_wrap
from utils.gradient_descent_opt import run_nag_sr_prior, run_nag_sr


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--N', type=int, default=4000, help='Micrograph width')
    parser.add_argument('--num_micro', type=int, default=10, help='Number of micrographs')
    parser.add_argument('--gamma', type=float, default=0.1,  help='Density factor')
    parser.add_argument('--snr', type=float, default=0.5, help='SNR of measurement')
    parser.add_argument('--num_iter', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--lr', type=float, default=18000, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.993, help='Momentum')
    parser.add_argument('--score_factor', type=float, default=1e-10, help='Score factor')
    parser.add_argument("--disable_prior", action="store_true", help="Disable score prior (default is enabled)")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    X_gt_high = plt.imread(f"../images/mnist_digit_3_crop_28X28.png")
    # Down sample to get low resolution image
    L_low = 14
    L_high = 28
    base_mat = np.zeros((int(L_high / L_low), int(L_high / L_low)))
    base_mat[0, 0] = 1
    sample_mat = np.kron(np.ones((L_low, L_low)), base_mat)
    sample_C = sample_mat.flatten() == 1
    X_gt_low = np.reshape(X_gt_high.flatten()[sample_C], (L_low, L_low))

    W = 2 * L_low - 1

    sigma2 = np.linalg.norm(X_gt_low) ** 2 / (args.snr * np.pi * (L_low // 2) ** 2)

    print("Simulate micrographs and calculate their moments")
    M1_ys, M2_ys, M3_ys = calc_M3_micrographs_wrap(L_low, sigma2, args.gamma, X_gt_low, W, args.N, args.num_micro)
    sum_M1_ys = np.sum(M1_ys)
    sum_M2_ys = np.sum(M2_ys, 0)
    sum_M3_ys = np.sum(M3_ys, 0)
    M1_y = sum_M1_ys / args.num_micro
    M2_y = sum_M2_ys / args.num_micro
    M3_y = sum_M3_ys / args.num_micro


    print("Solve optimization problem")
    X_initial = np.random.rand(L_high, L_high)
    X_initial = np.linalg.norm(X_gt_high) * X_initial / np.linalg.norm(X_initial)
    X_initial = X_initial.flatten()  # initial guess should be a 1-D array
    if args.disable_prior:
        estimate = run_nag_sr(X_initial, M1_y, M2_y, M3_y, L_high, L_low, sigma2, args.gamma, lr=args.lr,
                              momentum=args.momentum, max_iter=args.num_iter)
    else :
        estimate = run_nag_sr_prior(X_initial, M1_y, M2_y, M3_y, L_high, L_low, sigma2, args.gamma, lr=args.lr,
                                    momentum=args.momentum, score_factor=args.score_factor, max_iter=args.num_iter)


    X_est = np.reshape(estimate, (L_high, L_high))  # X recovered is passed as a 1-D array
    X_E = np.linalg.norm(X_est - X_gt_high) / np.linalg.norm(X_gt_high)
    print("estimation error is ", X_E)


if __name__ == "__main__":
    main()