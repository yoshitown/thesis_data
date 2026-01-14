import os
import sys
import numpy as np
import pandas as pd

# Path setup to import from src
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
sys.path.insert(0, SRC_DIR)

from gp_optimizer import train_gp, suggest_next_point_discrete

def main():
    # 1. Load Data
    # Assuming the script is in scripts/bayes_optim/
    csv_path = os.path.abspath(os.path.join(THIS_DIR, "../../preprocessed_data/selected_scores.csv"))
    
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract features and target
    X_all = df[['r', 's']].values
    y_all = df['B_score'].values
    
    # Store used indices to keep track of observed points
    all_indices = np.arange(len(df))
    
    # 2. Initialization: Randomly select 3 points to start
    np.random.seed(42) # Fixed seed for reproducibility
    initial_indices = np.random.choice(all_indices, size=3, replace=False)
    
    observed_indices = list(initial_indices)
    
    print(f"--- Initialization ---")
    print(f"Selecting {len(observed_indices)} initial random points:")
    for idx in observed_indices:
        print(f"  Index {idx}: r={X_all[idx,0]:.2f}, s={X_all[idx,1]:.2f} -> B_score={y_all[idx]:.4f}")
        
    # 3. Optimization Loop
    n_iterations = 5
    print(f"\n--- Starting Optimization Loop ({n_iterations} iterations) ---")
    
    for i in range(n_iterations):
        # Prepare training data
        train_indices = np.array(observed_indices)
        X_train = X_all[train_indices]
        y_train = y_all[train_indices]
        
        # Prepare candidate pool (indices not yet observed)
        candidate_indices = np.setdiff1d(all_indices, train_indices)
        
        if len(candidate_indices) == 0:
            print("No more candidates to explore.")
            break

        X_candidates = X_all[candidate_indices]
            
        # Train GP
        # Note: Depending on 'r' and 's' scaling, length_scale might need tuning. 
        # Here we use default from gp_optimizer (length_scale=0.3).
        gp = train_gp(X_train, y_train)
        
        # Suggest next point
        # suggest_next_point_discrete returns index relative to X_candidates
        best_candidate_rel_idx = suggest_next_point_discrete(gp, X_candidates)
        best_candidate_abs_idx = candidate_indices[best_candidate_rel_idx]
        
        # "Measure" the true value
        suggested_X = X_all[best_candidate_abs_idx]
        actual_y = y_all[best_candidate_abs_idx]
        
        print(f"\nIteration {i+1}:")
        print(f"  Proposed Point: r={suggested_X[0]:.2f}, s={suggested_X[1]:.2f} (Index {best_candidate_abs_idx})")
        print(f"  True B_score: {actual_y:.4f}")
        
        # Update observed set
        observed_indices.append(best_candidate_abs_idx)

    # Final result
    # Re-calculate best from all observed (including the last one)
    train_indices = np.array(observed_indices)
    y_train = y_all[train_indices]
    
    best_idx_in_train = np.argmax(y_train)
    best_abs_idx = train_indices[best_idx_in_train]
    
    print(f"\n--- Optimization Finished ---")
    print(f"Best found point: r={X_all[best_abs_idx,0]:.2f}, s={X_all[best_abs_idx,1]:.2f} -> B_score={y_all[best_abs_idx]:.4f}")

if __name__ == "__main__":
    main()
