from datasets import load_dataset
from transformers import TextGenerationPipeline
from sklearn.metrics import accuracy_score, f1_score
import re

# Load the fine-tuned model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained("./fine_tuned_model/final_model")
model = FastLanguageModel.for_inference(model)

# Load the test dataset
test_dataset = load_dataset("json", data_files="/home/ubuntu/data/test_prompts.json")["train"]

# Generate predictions
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=50)
predictions_eligible = []
predictions_recommended = []
for prompt in test_dataset["text"]:
    generated = pipe(prompt)[0]["generated_text"]
    # Extract eligible products
    eligible_match = re.search(r"Eligible Products: ([^\n]+)", generated)
    predicted_eligible = eligible_match.group(1).strip() if eligible_match else ""
    # Extract recommended product
    recommended_match = re.search(r"Recommended Product: ([^\n]+)", generated)
    predicted_recommended = recommended_match.group(1).strip() if recommended_match else ""
    predictions_eligible.append(predicted_eligible)
    predictions_recommended.append(predicted_recommended)

# Extract true labels
true_eligible = [re.search(r"Eligible Products: ([^\n]+)", prompt).group(1).strip() for prompt in test_dataset["text"]]
true_recommended = [re.search(r"Recommended Product: ([^\n]+)", prompt).group(1).strip() for prompt in test_dataset["text"]]

# Calculate metrics
eligible_accuracy = accuracy_score(true_eligible, predictions_eligible)
eligible_f1 = f1_score(true_eligible, predictions_eligible, average="weighted")
recommended_accuracy = accuracy_score(true_recommended, predictions_recommended)
recommended_f1 = f1_score(true_recommended, predictions_recommended, average="weighted")

print(f"Eligible Products Accuracy: {eligible_accuracy:.4f}")
print(f"Eligible Products F1 Score: {eligible_f1:.4f}")
print(f"Recommended Product Accuracy: {recommended_accuracy:.4f}")
print(f"Recommended Product F1 Score: {recommended_f1:.4f}")
print("\nSample Prediction:")
print(f"True Eligible: {true_eligible[0]}")
print(f"Predicted Eligible: {predictions_eligible[0]}")
print(f"True Recommended: {true_recommended[0]}")
print(f"Predicted Recommended: {predictions_recommended[0]}")