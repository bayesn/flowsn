import pandas as pd
import numpy as np
from pathlib import Path
from utils import TEMPLATE_M0, SNANA_KEYS

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def cuts(df):
    """Applies JLA-style selection cuts."""
    df = df[np.logical_and(df['x1'] >= -3, df['x1'] <= 3)]
    df = df[np.logical_and(df['c'] >= -0.3, df['c'] <= 0.3)]
    df = df[df['x1ERR'] <= 1.0]
    df = df[df['cERR'] <= 1.0]
    df = df[df['PKMJDERR'] <= 2.0]
    df = df[np.logical_and(df['FITPROB'] >= 0.05, df['FITPROB'] <= 1.0)]
    return df

# 1. Load and concatenate all training files
dataframes = []

# Standard Training files
for i in range(3):
    path = BASE_DIR / 'SNANA_files' / 'training_files' / f'TRAINING{i+1}.FITRES'
    if path.exists():
        loaded_df = pd.read_csv(path, comment="#", sep='\s+')
        dataframes.append(cuts(loaded_df.sample(frac=1))[:3_000_000])
        print(f"Loaded TRAINING{i+1}")

# Flat Training file
flat_path = BASE_DIR / 'SNANA_files' / 'training_files' / 'TRAINING_FLAT.FITRES'
if flat_path.exists():
    loaded_df = pd.read_csv(flat_path, comment="#", sep='\s+')
    dataframes.append(cuts(loaded_df.sample(frac=1))[:1_000_000])
    print("Loaded TRAINING_FLAT")

# Final combined dataframe
df = pd.concat(dataframes).sample(frac=1)
print(f"Total Supernovae in combined set: {len(df)}")

# 2. Build the array column-by-column using SNANA_KEYS
train_list = []

for key, (coeff, const) in SNANA_KEYS.items():
    if key in ['COV_c_x0', 'COV_x1_x0']:
        # Correct COV math: (coeff * COV / x0) + const
        val = (coeff * df[key] / df['x0'] + const).values
        
    elif key == 'SIM_DLMAG':
        val = (df[key] + df['SIM_MUSHIFT'] + const).values
        
    else:
        # Standard math for redshifts, mB, c, x1, errors, alpha, beta
        val = (coeff * df[key] + const).values
    
    train_list.append(val.reshape(-1, 1))

# 3. Stack into final array and save
train_arr = np.hstack(train_list)

save_path = Path(__file__).resolve().parent / 'SNANA_training_set.npy'
save_path.parent.mkdir(parents=True, exist_ok=True)

np.save(save_path, train_arr)
print(f"Successfully saved training set to {save_path}")
print(f"Final shape: {train_arr.shape}")
