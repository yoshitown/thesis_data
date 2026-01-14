import os
import sys
import numpy as np
import pandas as pd
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
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
    
    # 2. Initialization: Randomly select 3 points to start
    np.random.seed(42)  # For reproducibility
    initial_indices = np.random.choice(all_indices, size=3, replace=False)
    observed_indices = list(initial_indices)
    
    # Keep track of observed (r, s) values to avoid duplicates
    # Using a set of tuples for efficient lookup
    observed_points = set()
    for idx in observed_indices:
        observed_points.add((X_all[idx, 0], X_all[idx, 1]))

    print(f"--- Initialization ---")
    print(f"Selecting {len(observed_indices)} initial random points:")
    for idx in observed_indices:
        print(f"  Index {idx}: r={X_all[idx,0]:.2f}, s={X_all[idx,1]:.2f} -> B_score={y_all[idx]:.4f}")

    # 3. Optimization Loop using Optuna
    n_iterations = 5
    print(f"\n--- Starting Optuna Optimization Loop ({n_iterations} iterations) ---")

    # Define the objective function for Optuna
    def objective(trial):
        # In this simulation, we want Optuna to suggest (r, s)
        # Optuna usually optimizes continuous spaces, but we only have discrete points.
        # Approach: Allow Optuna to suggest continuous r, s, then map to nearest available point.
        # Alternatively, since we want to pick FROM the candidates, we can ask Optuna 
        # to pick an INDEX, but standard Optuna bayes optimization works best on continuous parameters.
        # 
        # Better Approach for "Discrete" selection with standard TPE/GP in Optuna:
        # Suggest r and s in their min-max range.
        # Find nearest point in *unobserved* candidates.
        # Return that point's B_score.
        # 
        # HOWEVER, the user wants us to "propose a point" -> "return true value".
        # We need to actively manage the 'observed' set outside the objective function 
        # or use `study.ask()` interface for manual loop control.
        # The `study.ask()` interface is perfect for "Human-in-the-loop" or simulation styles.
        
        # We will use the imperative "Ask-and-Tell" interface.
        pass 

    # We use Gaussian Process sampler to match the previous logic's intent (though TPE is default)
    # Optuna's GP support comes via `optuna.integration.BoTorchSampler` or standard `TPESampler`.
    # Based on the user request to "use optuna", TPE is standard, but if they want GP explicitly
    # we would need BoTorch. Given successful install of just 'optuna' and 'scikit-learn',
    # we will stick to default TPE or use skopt integration if available. 
    # Let's use standard Optuna TPE sampler which is robust.
    
    # Note: If we really want GP, we need `botorch` (which failed to install on py3.8). 
    # So we will use default Optuna sampler (TPE). It is also a statistical model (Bayesian).
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Pre-populate study with initial random points
    for idx in observed_indices:
        study.add_trial(
            optuna.trial.create_trial(
                params={"r": X_all[idx, 0], "s": X_all[idx, 1]},
                value=y_all[idx],
                distributions={
                    "r": optuna.distributions.FloatDistribution(0.6, 0.8), # Approx range
                    "s": optuna.distributions.FloatDistribution(0.1, 0.5)
                }
            )
        )

    r_min, r_max = X_all[:, 0].min(), X_all[:, 0].max()
    s_min, s_max = X_all[:, 1].min(), X_all[:, 1].max()

    for i in range(n_iterations):
        # 1. Ask Optuna for parameters
        trial = study.ask(fixed_distributions={
            "r": optuna.distributions.FloatDistribution(r_min, r_max),
            "s": optuna.distributions.FloatDistribution(s_min, s_max)
        })
        
        r_suggested = trial.params['r']
        s_suggested = trial.params['s']
        
        # 2. Find the nearest UN-OBSERVED point in our CSV data
        candidate_indices = np.setdiff1d(all_indices, observed_indices)
        
        if len(candidate_indices) == 0:
            print("No more candidates.")
            break
            
        X_candidates = X_all[candidate_indices]
        
        # Calculate distances to suggested point
        # Normalize/Scale if necessary, but simple Euclidean here for demo
        dists = np.sum((X_candidates - np.array([r_suggested, s_suggested]))**2, axis=1)
        nearest_rel_idx = np.argmin(dists)
        nearest_abs_idx = candidate_indices[nearest_rel_idx]
        
        # Check if this "nearest" logic keeps hitting the same point despite 'observed' logic
        # (It shouldn't because we remove observed from candidates)
        
        r_actual = X_all[nearest_abs_idx, 0]
        s_actual = X_all[nearest_abs_idx, 1]
        y_actual = y_all[nearest_abs_idx]
        
        # 3. "Tell" Optuna the result
        # IMPORTANT: We tell Optuna the *actual* parameters we evaluated, not just what it suggested.
        # This helps the model learn the true landscape of discrete points better.
        # But `study.tell` expects to match the `ask`. 
        # If we evaluate a different point, strictly speaking we should add a new trial.
        # However, for simulation, we can just report the value for the suggested point 
        # OR (better) interpret the suggestion as "Look in this area" and report the finding.
        # Let's simply report the value obtained at the nearest neighbor.
        
        # To be precise in tracking:
        print(f"\nIteration {i+1}:")
        print(f"  Optuna Suggested: r={r_suggested:.4f}, s={s_suggested:.4f}")
        print(f"  Nearest Candidate: r={r_actual:.2f}, s={s_actual:.2f} (Index {nearest_abs_idx})")
        print(f"  True B_score: {y_actual:.4f}")
        
        study.tell(trial, y_actual)
        
        # Update observed
        observed_indices.append(nearest_abs_idx)

    # Final Recap
    print(f"\n--- Optimization Finished ---")
    print(f"Best trial value: {study.best_value:.4f}")
    print(f"Best trial params: {study.best_params}")
    
    # Find the row in DataFrame that matches best params (approx)
    # Since we might have reported slightly different params than actuals to Optuna (due to nearest neighbor),
    # let's find the best from our tracked observed indices.
    best_observed_idx = observed_indices[np.argmax(y_all[observed_indices])]
    print(f"Best observed point in Data: r={X_all[best_observed_idx,0]:.2f}, s={X_all[best_observed_idx,1]:.2f} -> B_score={y_all[best_observed_idx]:.4f}")

if __name__ == "__main__":
    main()
