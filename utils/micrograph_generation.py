import numpy as np


def generate_clean_micrograph(X, W, L, N, m, p=np.array([1]), seed=None):
    """
    Generate a clean micrograph by placing signals at random non-overlapping positions.

    Parameters:
        X: Signal to be placed
        W: Width of reserved area around signal
        L: Size of signal
        N: Size of output micrograph
        m: Number of signals to place
        p: Number of signal instances to place
        seed: Random seed for reproducibility

    Returns:
        Y: Generated micrograph
        placed_list: Count of placed signals per type
        locations: List of signal placement coordinates
    """
    if seed != None:
        np.random.seed(seed)
    m = round(m)
    mask = np.zeros((N, N))
    # The locations table records the chosen signal locations.
    locations = [[] for i in range(m)]
    # This counter records how many signals we successfully placed.
    placed = 0
    placed_list = [0 for i in range(1)]
    max_trials = 5*m

    Y = np.zeros((N, N))
    for _ in range(max_trials):
        # Pick a candidate location for the upper-left corner of the signal
        candidate = np.random.randint(N-W, size=2)
        # Check if there is enough room, taking the separation rule into
        # account. That is, a square of size WxW with upper-left corner
        # specified by the candidate must be entirely free.
        if not ((mask[candidate[0]:candidate[0] + W, candidate[1]:candidate[1] + W] == 1).any()):
            # Record the successful candidate
            locations[placed] = candidate
            # Mark the area as reserved
            mask[candidate[0]:candidate[0]+W, candidate[1]:candidate[1]+W] = 1
            index_rand = np.random.choice(1, p=p)
            placed_list[index_rand] = placed_list[index_rand] + 1
            placed = placed + 1
            Y[candidate[0] : candidate[0] + L, candidate[1] : candidate[1] + L] = X
            # Stop if we placed sufficiently many signals successfully.
            if placed >= m:
                break
    locations = locations[0:placed]
    return Y, placed_list, locations


