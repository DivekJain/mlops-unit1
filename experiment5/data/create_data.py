# create_dataset.py

from sklearn.datasets import load_diabetes
import pandas as pd

# Load dataset
data = load_diabetes()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df['target'] = data.target

# Save to CSV
df.to_csv("diabetes.csv", index=False)

print("diabetes.csv created successfully!")
