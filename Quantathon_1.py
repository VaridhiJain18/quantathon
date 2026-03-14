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
    """
    # Generate all coin flips for all trials at once. Shape: (T, N)
    coin1 = rng.choice([-1, 1], size=(T, N))
    coin2 = rng.choice([-1, 1], size=(T, N))

    # Calculate X and A across the N dimension (axis=1)
    X = np.cumsum(coin1, axis=1)
    A = (X % 5) - 2

    # dY is simply coin2. Calculate increments of S directly.
    # alpha has shape (T, 1) to broadcast correctly across (T, N)
    d = alpha * A + sigma * coin2
    
    # S_n = cumulative sum of the increments
    S = np.cumsum(d, axis=1)

    return S


# ============================================================
# PART (b)
# ============================================================

def estimator_b(S):
    """
    U_N: MMSE-consistent estimator of alpha^2 from S_1 .. S_N.
    sigma = 1 is assumed known.
    Handles both 1D arrays (single trial) and 2D arrays (multiple trials).
    """
    # Calculate differences along the last axis (-1) to support both 1D and 2D arrays
    d = np.diff(S, prepend=0.0, axis=-1)
    
    # E[d_n^2] -> 2 * alpha^2 + 1
    mean_d2 = np.mean(d ** 2, axis=-1)
    
    # Solve for alpha^2
    alpha2_hat = np.maximum(0.0, (mean_d2 - 1.0) / 2.0)
    
    return alpha2_hat


def run_part_b(N_values=(50, 200, 500, 2000, 5000, 20000), T=30_000, seed=42):
    rng = np.random.default_rng(seed)
    results = []

    for N in N_values:
        # Generate true alpha ~ U(0,1) for T trials. Shape: (T, 1)
        alpha = rng.uniform(0.0, 1.0, size=(T, 1))
        
        # Target is alpha^2 (Shape: T, 1)
        target = alpha ** 2 
        
        # Simulate all T paths at once
        S = simulate_system_vectorized(alpha, sigma=1.0, N=N, T=T, rng=rng)
        
        # Estimate alpha^2 for all T paths at once (Shape: T)
        UN = estimator_b(S)
        
        # Calculate MSE across the T trials
        # Reshape target to (T,) to match UN's shape
        sq_errors = (target.flatten() - UN) ** 2
        
        results.append({
            "N"       : N,
            "mean_MSE": float(np.mean(sq_errors)),
            "std_MSE" : float(np.std(sq_errors) / np.sqrt(T)),
        })

    return results


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(42)
    T = 10000
    N_values = [50,200,500,2000,2026,5000]

    print("="*60)
    print("PART (b)  —  Estimating alpha^2")
    print("Estimator: U_N = max(0, mean(d_n^2) - 1)/2")
    print("Theoretical MSE limit as N->inf: 0")
    print("="*60)
    print(f"{'N':>8}  {'Mean MSE':>12}  {'±2 SE':>12}")
    print("-"*40)

    for N in N_values:
        alpha = rng.uniform(0.0,1.0,(T,1))
        # Simulate
        coin1 = rng.choice([-1,1], size=(T,N))
        coin2 = rng.choice([-1,1], size=(T,N))
        X = np.cumsum(coin1, axis=1)
        A = (X % 5) - 2
        d = alpha * A + coin2
        S = np.cumsum(d, axis=1)

        # Estimator
        dS = np.diff(np.insert(S,0,0,axis=1), axis=1)
        UN = np.maximum(0,(np.mean(dS**2, axis=1)-1)/2)

        # Target
        target = (alpha**2).flatten()
        errors = (UN-target)**2

        mean_MSE = np.mean(errors)
        se_MSE = np.std(errors)/np.sqrt(T)

        print(f"{N:>8}  {mean_MSE:>12.6f}  {2*se_MSE:>12.6f}")