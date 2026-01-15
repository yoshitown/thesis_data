import os
import sys
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
import warnings

# Path setup to import from src
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
sys.path.insert(0, SRC_DIR)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    # 1. Load Data
    csv_path = os.path.abspath(os.path.join(THIS_DIR, "../../preprocessed_data/selected_scores.csv"))
    
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract features (r, s) and target (B_score)
    X_all = df[['r', 's']].values
    y_all = df['B_score'].values
    
    # Store indices to simulate "observed" vs "unobserved"
    all_indices = np.arange(len(df))
    
    # 2. Initialization: Use ALL points from the CSV
    observed_indices = list(all_indices)
    
    print(f"--- Initialization ---")
    print(f"Loading all {len(observed_indices)} points from CSV as initial data.")
    # Train a Gaussian Process on all available CSV data to provide a prediction
    # (this model is only used to show an estimated B_score before asking for true value)
    kernel = RBF(length_scale=0.3) + WhiteKernel(noise_level=1e-2)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
    try:
        gp.fit(X_all, y_all)
    except Exception:
        # Fallback: if GP training fails, set gp to None and skip predictions
        gp = None
    
    # 3. Optimization Loop using GP + EI (optimize acquisition with L-BFGS-B)
    n_iterations = 5
    print(f"\n--- Starting GP+EI Optimization Loop ({n_iterations} iterations) ---")

    # Use raw individual trial data (do not aggregate duplicates)
    X_candidates = X_all
    y_candidates = y_all

    # Start training data with all individual observations
    X_train = X_candidates.copy()
    y_train = y_candidates.copy()

    # Bounds for optimization
    bounds = [(float(X_all[:, 0].min()), float(X_all[:, 0].max())),
              (float(X_all[:, 1].min()), float(X_all[:, 1].max()))]

    def expected_improvement(x, gp, y_best, xi=0.0):
        x = np.atleast_2d(x)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = sigma.reshape(-1)
        mu = mu.reshape(-1)
        with np.errstate(divide='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def propose_location(gp, bounds, n_restarts=10):
        dim = 2
        best_x = None
        best_acq = -np.inf

        def min_obj(x):
            # negative EI for minimizer
            return -expected_improvement(x.reshape(1, -1), gp, y_train.max())[0]

        # multiple random restarts
        for _ in range(n_restarts):
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if not res.success:
                continue
            acq_val = -res.fun
            if acq_val > best_acq:
                best_acq = acq_val
                best_x = res.x

        # fallback: grid search over candidate points if optimizer failed
        if best_x is None:
            ei_vals = expected_improvement(X_candidates, gp, y_train.max())
            idx = int(np.argmax(ei_vals))
            return X_candidates[idx]

        return best_x

    for i in range(n_iterations):
        # Train GP on current observations
        kernel = RBF(length_scale=0.3) + WhiteKernel(noise_level=1e-4)
        gp_loop = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp_loop.fit(X_train, y_train)

        # Propose next point by maximizing EI
        x_next = propose_location(gp_loop, bounds, n_restarts=20)

        # Show GP estimate at proposed point
        mu, sigma = gp_loop.predict(np.atleast_2d(x_next), return_std=True)
        print(f"\nIteration {i+1} Proposed (continuous): r={x_next[0]:.4f}, s={x_next[1]:.4f}")
        print(f"  GP estimate -> mean={mu[0]:.4f}, std={sigma[0]:.4f}")

        # Map to nearest available CSV row (individual trial) to get true B_score
        dists = np.sum((X_candidates - x_next) ** 2, axis=1)
        nearest_idx = int(np.argmin(dists))
        true_x = X_candidates[nearest_idx]
        true_y = y_candidates[nearest_idx]
        print(f"  Nearest CSV point: r={true_x[0]:.2f}, s={true_x[1]:.2f} -> true B_score={true_y:.4f}")

        # Add observation and continue
        X_train = np.vstack([X_train, true_x])
        y_train = np.concatenate([y_train, [true_y]])

    # Final result
    best_idx = int(np.argmax(y_train))
    print(f"\n--- Optimization Finished ---")
    print(f"Best observed in training set: r={X_train[best_idx,0]:.2f}, s={X_train[best_idx,1]:.2f} -> B_score={y_train[best_idx]:.4f}")

if __name__ == "__main__":
    main()
