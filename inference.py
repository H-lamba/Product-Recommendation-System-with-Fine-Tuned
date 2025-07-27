import torch
from unsloth import FastLanguageModel
import re
import json

# Load the fine-tuned model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained("./fine_tuned_model/final_model")
model = FastLanguageModel.for_inference(model)

# Define a new customer profile
new_customer = {
    'age': 35,
    'income': 50000,
    'credit_score': 700,
    'account_balance': 10000,
    'transaction_freq': 20,
    'transaction_amount': 500,
    'employment_status': 'Employed',
    'customer_segment': 'Young Professional',
    'health_status': 'Good',
    'risk_tolerance': 'Medium',
    'investment_goals': 'Long-term',
    'debt_to_income_ratio': 0.3,
    'collateral_value': 20000
}

# Create the prompt
prompt = f"""### Customer Profile ###
Age: {new_customer['age']}
Income: {new_customer['income']}
Credit Score: {new_customer['credit_score']}
Account Balance: {new_customer['account_balance']}
Transaction Frequency: {new_customer['transaction_freq']}
Transaction Amount: {new_customer['transaction_amount']}
Employment Status: {new_customer['employment_status']}
Customer Segment: {new_customer['customer_segment']}
Health Status: {new_customer['health_status']}
Risk Tolerance: {new_customer['risk_tolerance']}
Investment Goals: {new_customer['investment_goals']}
Debt-to-Income Ratio: {new_customer['debt_to_income_ratio']}
Collateral Value: {new_customer['collateral_value']}
### Output ###"""

# Generate prediction
output = model.generate(
    **tokenizer(prompt, return_tensors="pt").to("cuda"),
    max_new_tokens=50
)
generated = tokenizer.decode(output[0], skip_special_tokens=True)

# Extract eligible products and recommended product
eligible_match = re.search(r"Eligible Products: ([^\n]+)", generated)
predicted_eligible = eligible_match.group(1).strip() if eligible_match else ""
recommended_match = re.search(r"Recommended Product: ([^\n]+)", generated)
predicted_recommended = recommended_match.group(1).strip() if recommended_match else ""

# Format as JSON for production
response = {
    "eligible_products": predicted_eligible.split(", ") if predicted_eligible else [],
    "recommended_product": predicted_recommended
}

print(json.dumps(response, indent=2))