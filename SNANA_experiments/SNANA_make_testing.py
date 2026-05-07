import pandas as pd
import numpy as np
import os
import argparse
import copy
from utils import TEMPLATE_M0, SNANA_KEYS

parser = argparse.ArgumentParser()
parser.add_argument("--cosmo", type=int, default=1)
args = parser.parse_args()
cosmo = 'cosmo' + str(args.cosmo)
print(f"Processing: {cosmo}")

# JLA cuts
def cuts(df):
    df = df[np.logical_and(df['x1'] >= -3, df['x1'] <= 3)]
    df = df[np.logical_and(df['c'] >= -0.3, df['c'] <= 0.3)]
    df = df[df['x1ERR'] < 1.0]
    df = df[df['cERR'] < 1.0]
    df = df[df['PKMJDERR'] < 2.0]
    df = df[np.logical_and(df['FITPROB'] >= 0.05, df['FITPROB'] <= 1.0)]
    return df

os.makedirs('testing_sets/' + cosmo, exist_ok=True)

# Loop through files
for i in range(100):
    dir_ = '../../SNANA_files/test_files/' + cosmo + '/TESTING' + str(i + 1) + '.FITRES'
    
    # Try to load file, skip if missing
    if not os.path.exists(dir_):
        continue
        
    df = cuts(pd.read_csv(dir_, comment="#", sep='\s+').sample(frac=1))
    
    test_arr = np.empty((len(df), 0), dtype=np.float64)

    # LOOP THROUGH KEYS WITHOUT MODIFYING THE DICTIONARY
    for key, (coeff, const) in SNANA_KEYS.items():
        if key in ['COV_c_x0', 'COV_x1_x0']:
            # Apply the division by x0 locally for THIS file only
            # Math: (-2.5/log(10)) * COV_from_file / x0_from_file
            val = (coeff * df[key] / df['x0'] + const).values.reshape(-1, 1)
        else:
            # Standard math for all other columns
            val = (coeff * df[key] + const).values.reshape(-1, 1)
            
        test_arr = np.append(test_arr, val, axis=1)

    # Sample fake masses (appended last, index -1)
    fake_mass = np.random.uniform(8, 12, len(df)).reshape(-1, 1)
    test_arr = np.append(test_arr, fake_mass, axis=1)

    print(f"File {i+1} | Supernovae: {len(test_arr)} | Shape: {test_arr.shape}")
    np.save('testing_sets/' + cosmo + '/SNANA_test' + str(i) + '.npy', test_arr)
