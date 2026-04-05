import numpy as np
import pandas as pd
import os

# Correct column names based on PhysioNet documentation
column_names = [
    'elapsed_time',           # col 1  — seconds
    'left_stride_interval',   # col 2  — seconds
    'right_stride_interval',  # col 3  — seconds
    'left_swing_sec',         # col 4  — seconds
    'right_swing_sec',        # col 5  — seconds
    'left_swing_pct',         # col 6  — % of stride
    'right_swing_pct',        # col 7  — % of stride
    'left_stance_sec',        # col 8  — seconds
    'right_stance_sec',       # col 9  — seconds
    'left_stance_pct',        # col 10 — % of stride
    'right_stance_pct',       # col 11 — % of stride
    'double_support_sec',     # col 12 — seconds
    'double_support_pct',     # col 13 — % of stride
]

def load_subject_file(filepath, label):
    try:
        data = np.loadtxt(filepath)
        df   = pd.DataFrame(data, columns=column_names)
        df = df.drop(columns=['elapsed_time'])

        # Per subject summary
        # Each subject becomes ONE row
        # We compute mean and std for every feature
        # std captures variability which is key for Parkinson's
        row = {}

        for col in df.columns:
            row[f'{col}_mean'] = round(df[col].mean(), 4)
            row[f'{col}_std']  = round(df[col].std(),  4)

        #  Derived asymmetry features
        # Difference between left and right sides
        # Parkinson's often affects one side more than the other

        row['stride_asymmetry'] = round(abs(
            df['left_stride_interval'].mean() -
            df['right_stride_interval'].mean()
        ), 4)

        row['swing_asymmetry_sec'] = round(abs(
            df['left_swing_sec'].mean() -
            df['right_swing_sec'].mean()
        ), 4)

        row['swing_asymmetry_pct'] = round(abs(
            df['left_swing_pct'].mean() -
            df['right_swing_pct'].mean()
        ), 4)

        row['stance_asymmetry_sec'] = round(abs(
            df['left_stance_sec'].mean() -
            df['right_stance_sec'].mean()
        ), 4)

        row['stance_asymmetry_pct'] = round(abs(
            df['left_stance_pct'].mean() -
            df['right_stance_pct'].mean()
        ), 4)

        #  Overall stride variability
        # Average variability across both feet
        # Higher = more irregular = stronger Parkinson's signal
        row['overall_stride_variability'] = round((
            df['left_stride_interval'].std() +
            df['right_stride_interval'].std()
        ) / 2, 4)

        #  Double support variability
        # How consistent is the double support phase
        row['double_support_variability'] = round(
            df['double_support_pct'].std(), 4
        )

        # Number of strides recorded
        row['num_strides'] = len(df)

        # Metadata
        row['subject'] = os.path.basename(filepath).replace('.ts', '')
        row['label']   = label  # 1 = Parkinson's, 0 = Healthy

        return row

    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None


def build_dataset(folder, label):
    rows  = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.ts')])

    print(f"\nProcessing {len(files)} files from {folder}...")

    for filename in files:
        filepath = os.path.join(folder, filename)
        row      = load_subject_file(filepath, label)
        if row:
            rows.append(row)
            print(f"  OK: {filename} — {row['num_strides']} strides")
        else:
            print(f"  FAILED: {filename}")

    return pd.DataFrame(rows)


# Run

park_df = build_dataset('data/parkinsons', label=1)
ctrl_df = build_dataset('data/healthy',    label=0)

full_df = pd.concat([park_df, ctrl_df], ignore_index=True)

# Save
full_df.to_csv('gait_dataset.csv', index=False)

# Summary
print(f"\n{'='*50}")
print(f"Saved gait_dataset.csv")
print(f"{'='*50}")
print(f"Total subjects:         {len(full_df)}")
print(f"Parkinson's subjects:   {len(full_df[full_df.label == 1])}")
print(f"Healthy subjects:       {len(full_df[full_df.label == 0])}")
print(f"Features per subject:   {len(full_df.columns) - 2}")
print(f"\nColumns in dataset:")
for col in full_df.columns:
    print(f"  {col}")
print(f"\nFirst 3 rows:")
print(full_df.head(3).to_string())