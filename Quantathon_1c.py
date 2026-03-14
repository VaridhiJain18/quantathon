import numpy as np

# ============================================================
# SHARED HELPERS
# ============================================================

def simulate_system_vectorized(alpha, sigma, N, T, rng):
    """
    Simulate the full (X, Y, A, S) system for n = 1..N across T trials.
    
    Returns
    -------
    S : ndarray, shape (T, N) — observed values S_1 .. S_N for all trials
    A : ndarray, shape (T, N) — unobserved A_1 .. A_N for all trials
    """
    coin1 = rng.choice([-1, 1], size=(T, N))
    coin2 = rng.choice([-1, 1], size=(T, N))

    X = np.cumsum(coin1, axis=1)
    A = (X % 5) - 2
    d = alpha * A + sigma * coin2
    S = np.cumsum(d, axis=1)

    return S, A

# ============================================================
# PART (c) - OPTIMAL HMM ESTIMATOR
# ============================================================

# Define the states of A_n
A_VALS = np.array([-2, -1, 0, 1, 2])

# Create the Markov Transition Matrix for A_n
# T_mat[i, j] = P(A_n = j | A_{n-1} = i)
T_mat = np.zeros((5, 5))
for i in range(5):
    T_mat[i, (i - 1) % 5] = 0.5  # Step left (mod 5)
    T_mat[i, (i + 1) % 5] = 0.5  # Step right (mod 5)

def estimator_c(S_batch):
    """
    Computes the exact MMSE estimate R = E[A_N | S_1..S_N] using a Forward HMM filter.
    Handles both a 1D array of shape (N,) and a 2D array of shape (T_trials, N).
    """
    # Ensure 2D for consistent vectorized processing
    is_1d = S_batch.ndim == 1
    if is_1d:
        S_batch = S_batch[np.newaxis, :]

    # Extract step-by-step differences d_n
    d_batch = np.diff(np.insert(S_batch, 0, 0, axis=1), axis=1)
    T_trials, N = d_batch.shape

    # Initial belief at n=0: We know A_0 = -2 with 100% certainty.
    # Index 0 corresponds to A_VALS = -2
    p = np.zeros((T_trials, 5))
    p[:, 0] = 1.0 

    # Run the forward filter through time 1 to N
    for n in range(N):
        d_n = d_batch[:, n] 
        
        # 1. Predict Step: Push current belief through the transition matrix
        p_pred = p @ T_mat
        
        # 2. Update Step: Multiply by the observation likelihood P(d_n | A_n)
        # Because sigma=1, d_n = A_n + y_n, where y_n is +1 or -1.
        # So the likelihood is 0.5 if |d_n - A_n| == 1, else 0.
        likelihood = (np.abs(d_n[:, None] - A_VALS) == 1).astype(float) * 0.5
        p_unnormalized = p_pred * likelihood
        
        # 3. Normalize probabilities to sum to 1
        row_sums = p_unnormalized.sum(axis=1, keepdims=True)
        # Safe division to prevent zero-division errors
        p = np.divide(p_unnormalized, row_sums, out=np.zeros_like(p_unnormalized), where=row_sums!=0)

    # p now contains the exact posterior probability distribution of A_N.
    # The MMSE estimate is the expected value of this distribution.
    expected_A_N = np.sum(p * A_VALS, axis=1)

    return expected_A_N[0] if is_1d else expected_A_N

def run_part_c(N=2026, T_trials=100000, seed=42):
    rng = np.random.default_rng(seed)
    
    # 1. Simulate Data
    S, A = simulate_system_vectorized(alpha=1.0, sigma=1.0, N=N, T=T_trials, rng=rng)
    
    # 2. True A_N values
    true_AN = A[:, -1]
    
    # 3. Estimate A_N
    estimated_AN = estimator_c(S)
    
    # 4. Calculate MSE
    sq_errors = (true_AN - estimated_AN)**2
    
    return {
        "N": N,
        "T": T_trials,
        "simulated_MSE": float(np.mean(sq_errors)),
        "simulated_SE": float(np.std(sq_errors) / np.sqrt(T_trials))
    }

# ============================================================
# MAIN — print results
# ============================================================

if __name__ == "__main__":
    np.set_printoptions(precision=6)
    
    print("=" * 60)
    print("PART (c)  —  Optimal HMM Estimator for A_N")
    print("  Estimator: R = E[A_N | S_1 .. S_N]")
    print("=" * 60)
    
    res = run_part_c(N=2026, T_trials=100000)
    
    print(f"  N                  : {res['N']}")
    print(f"  Monte Carlo trials : {res['T']:,}")
    print(f"  Simulated MSE      : {res['simulated_MSE']:.6f}  (± {2*res['simulated_SE']:.6f})")
    print("=" * 60)