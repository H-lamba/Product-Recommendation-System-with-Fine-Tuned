import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import os

# Configuration
max_seq_length = 2048
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
load_in_4bit = True
output_dir = "/home/ubuntu/data/finetuned_model"
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-4
warmup_steps = 5
logging_steps = 10
save_steps = 100
eval_steps = 100

# Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Deepseek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Load Dataset
dataset = load_dataset("json", data_files={
    "train": "/home/ubuntu/data/train_prompts.json",
    "validation": "/home/ubuntu/data/val_prompts.json"
})

# Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    logging_steps=logging_steps,
    save_steps=save_steps,
    eval_steps=eval_steps,
    evaluation_strategy="steps",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    report_to="none",
)

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)

# Train
trainer.train()

# Save Model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")