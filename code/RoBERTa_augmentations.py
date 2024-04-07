import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from huggingface_hub import HfFolder, notebook_login
from tqdm import tqdm

import wandb
import os
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


HF_TOKEN = "put your token here"
WANDB_TOKEN = "put your token here"

wandb.login(key=WANDB_TOKEN)

model_id = "roberta-base"
username = "ImperialIndians23"
downsampled = False
processed = True
augmented = True
synonym_augmented = True

dataset_id = "ImperialIndians23/nlp_cw_data"
run_name = f"roberta - keyword"

if processed:
    dataset_id += "_processed"
    run_name += " processed "
else:
    dataset_id += "_unprocessed"
    run_name += " unprocessed "

if downsampled:
    dataset_id += "_downsampled"
    run_name += "downsampled "

if augmented:
    dataset_id += "_augmented"
    run_name += "augmented "

if synonym_augmented:
    dataset_id += "_synonym"
    run_name += "synonym "


training_config = {
    "learning_rate": 1e-5,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 2,
    "lr_scheduler_type": "inverse_sqrt",
    "warmup_steps": 500,
    "load_best_model_at_end": True,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "metric_for_best_model": "f1",
}

run_name += f"- {training_config['num_train_epochs']} epochs"

repository_id = "ImperialIndians23/RobertaBaseUnprocessedAugmentedSynonym"

dataset = load_dataset(dataset_id)

print("Processing the dataset...")

# Training
train_dataset = dataset["train"]

# Validation dataset
val_dataset = dataset["valid"]

# Preprocessing
tokenizer = RobertaTokenizerFast.from_pretrained(model_id)


# This function tokenizes the input text using the RoBERTa tokenizer.
# It applies padding and truncation to ensure that all sequences have the same length (512 tokens).
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)


train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

num_labels = 2
class_names = [0, 1]
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

id2label = {i: label for i, label in enumerate(class_names)}

# Update the model's configuration with the id2label mapping
config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": id2label})


model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

training_args = TrainingArguments(
    output_dir=repository_id,
    logging_dir=f"{repository_id}/logs",
    logging_steps=10,
    save_total_limit=3,
    push_to_hub=True,
    report_to="wandb",
    **training_config,
)

run = wandb.init(
    project="nlp_cw",
    name=run_name,
    # Track hyperparameters and run metadata
    config=training_config,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

# Save the tokenizer and create a model card
tokenizer.save_pretrained(repository_id)
trainer.create_model_card()

# Push the results to the hub
trainer.push_to_hub()
