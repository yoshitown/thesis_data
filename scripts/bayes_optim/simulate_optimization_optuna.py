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
        print(f"\nIteration {i+1}:")
        print(f"  Optuna Suggested: r={r_suggested:.4f}, s={s_suggested:.4f}")

        # Show GP estimate if available
        if gp is not None:
            try:
                mean, std = gp.predict(np.array([[r_suggested, s_suggested]]), return_std=True)
                print(f"  GP estimate -> mean={mean[0]:.4f}, std={std[0]:.4f}")
            except Exception:
                print("  GP prediction unavailable.")
        else:
            print("  No GP model available for prediction.")

        # Interactive Input: ask user for the true B_score
        while True:
            try:
                user_input = input(f"  Enter true B_score for r={r_suggested:.4f}, s={s_suggested:.4f}: ")
                y_actual = float(user_input)
                break
            except ValueError:
                print("  Invalid input. Please enter a valid number.")
        
        # 3. "Tell" Optuna the result
        study.tell(trial, y_actual)
        
        # Update observed (For consistency with previous logic, though strictly not needed for Optuna if we don't care about CSV matching)
        # We can add a dummy index or just skip the CSV tracking part since this is manual input now.
        # But to keep the "best observed" logic at the end working if possible, we'll just track the value.
        
    # Final Recap
    print(f"\n--- Optimization Finished ---")
    print(f"Best trial value: {study.best_value:.4f}")
    print(f"Best trial params: {study.best_params}")

if __name__ == "__main__":
    main()
