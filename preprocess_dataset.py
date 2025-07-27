import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Load the Dataset
df = pd.read_csv("Updated_Synthetic_Dataset.csv")

# Handle Missing Values
print("Checking for missing values:\n", df.isnull().sum())
df = df.dropna(subset=['eligible_products'])  # Drop rows with missing eligible_products
print("\nAfter dropping missing values, shape:", df.shape)

# Inspect Distribution of recommended_product
print("\nDistribution of recommended_product:")
print(df['recommended_product'].value_counts())

# Filter out rare classes (fewer than 10 instances)
min_instances = 10
class_counts = df['recommended_product'].value_counts()
valid_classes = class_counts[class_counts >= min_instances].index
df = df[df['recommended_product'].isin(valid_classes)]
print("\nAfter filtering rare classes, shape:", df.shape)
print("\nFiltered distribution of recommended_product:")
print(df['recommended_product'].value_counts())

# Verify Data Integrity
print("\nUnique customer IDs:", df['customer_id'].nunique(), "out of", len(df))

# Define Prompt Template
def create_prompt(row):
    prompt = f"""### Customer Profile ###
Age: {row['age']}
Income: {row['income']}
Credit Score: {row['credit_score']}
Account Balance: {row['account_balance']}
Transaction Frequency: {row['transaction_freq']}
Transaction Amount: {row['transaction_amount']}
Employment Status: {row['employment_status']}
Customer Segment: {row['customer_segment']}
Health Status: {row['health_status']}
Risk Tolerance: {row['risk_tolerance']}
Investment Goals: {row['investment_goals']}
Debt-to-Income Ratio: {row['debt_to_income_ratio']}
Collateral Value: {row['collateral_value']}
### Output ###
Eligible Products: {row['eligible_products']}
Recommended Product: {row['recommended_product']}"""
    return prompt

# Apply Prompt Template
df['prompt'] = df.apply(create_prompt, axis=1)

# Split the Dataset (70% train, 20% val, 10% test)
train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df['recommended_product']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.33, random_state=42, stratify=temp_df['recommended_product']
)

# Save as JSON for Unsloth
def save_as_json(df, filename):
    data = [{"text": prompt} for prompt in df['prompt']]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

save_as_json(train_df, 'train_prompts.json')
save_as_json(val_df, 'val_prompts.json')
save_as_json(test_df, 'test_prompts.json')

print("\nPreprocessing complete! Files saved: train_prompts.json, val_prompts.json, test_prompts.json")
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")