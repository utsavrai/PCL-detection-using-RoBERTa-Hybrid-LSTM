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
from transformers import DataCollatorWithPadding

from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW
from torch import nn, optim
from torch.nn import functional as F


class RobertaClassifier(nn.Module):
    def __init__(self, num_labels, config):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base", config=config)
        self.mapper = nn.Linear(
            self.roberta.config.hidden_size * 2, self.roberta.config.hidden_size * 2
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size * 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.config = config

    def forward(
        self,
        input_ids,
        attention_mask,
        community_input_ids,
        community_attention_mask,
        labels=None,
    ):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = outputs[1]

        # Process community keywords
        community_outputs = self.roberta(
            input_ids=community_input_ids, attention_mask=community_attention_mask
        )
        community_embedding = community_outputs[1]

        # Concatenate
        combined_embedding = torch.cat((text_embedding, community_embedding), dim=1)

        mapped_embedding = self.mapper(combined_embedding)
        # Apply dropout to the output of the mapper
        dropped_embedding = self.dropout(mapped_embedding)

        # Pass the result through the classifier to get logits
        logits = self.classifier(dropped_embedding)

        # Compute loss if labels are provided (during training)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        else:
            return logits


class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):

        batch = super().__call__(features)

        community_input_ids = torch.stack(
            [feature["community_input_ids"] for feature in features]
        )
        community_attention_mask = torch.stack(
            [feature["community_attention_mask"] for feature in features]
        )

        # Pad the community_input_ids and attention_mask to the longest sequence in the batch
        padded_community_input_ids = self.pad_tensors(
            community_input_ids, self.tokenizer.pad_token_id
        )
        padded_community_attention_mask = self.pad_tensors(community_attention_mask, 0)

        batch["community_input_ids"] = padded_community_input_ids
        batch["community_attention_mask"] = padded_community_attention_mask

        return batch

    def pad_tensors(self, tensors, pad_token_id):
        # Find the longest tensor
        max_length = max(t.size(0) for t in tensors)
        # Pad each tensor to match the longest one
        padded = torch.stack(
            [
                torch.cat(
                    [
                        t,
                        torch.full(
                            (max_length - t.size(0),), pad_token_id, dtype=t.dtype
                        ),
                    ]
                )
                for t in tensors
            ]
        )
        return padded


def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
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


HF_TOKEN = "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"
WANDB_TOKEN = "26b0c0ff3251f094fd91c1472199ea71e4edaa45"

# WANDB_TOKEN = userdata.get('WANDB_TOKEN')
wandb.login(key=WANDB_TOKEN)

model_id = "roberta-base"
username = "ImperialIndians23"
downsampled = True
processed = True

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


repository_id = "ImperialIndians23/RobertaBaseUnprocessedDownsampled"

dataset = load_dataset(dataset_id)

print("Processing the dataset...")

tokenizer = RobertaTokenizerFast.from_pretrained(model_id)


def tokenize_data(batch):
    # Tokenize the main text
    text_encoding = tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Tokenize the community keyword
    community_encoding = tokenizer(
        batch["community"],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    return {
        "input_ids": text_encoding["input_ids"],
        "attention_mask": text_encoding["attention_mask"],
        "community_input_ids": community_encoding["input_ids"],
        "community_attention_mask": community_encoding["attention_mask"],
    }


train_dataset = dataset["train"].map(tokenize_data, batched=True)
valid_dataset = dataset["valid"].map(tokenize_data, batched=True)


train_dataset.set_format(
    "torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label",
        "community_input_ids",
        "community_attention_mask",
    ],
)
valid_dataset.set_format(
    "torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label",
        "community_input_ids",
        "community_attention_mask",
    ],
)

num_labels = 2
class_names = [0, 1]
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

id2label = {i: label for i, label in enumerate(class_names)}

# Update the model's configuration with the id2label mapping
config = AutoConfig.from_pretrained(model_id)
config.update({"id2label": id2label})

model = RobertaClassifier(num_labels=2, config=config)


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


data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

# Save the tokenizer and create a model card
tokenizer.save_pretrained(repository_id)
trainer.create_model_card()

# Push the results to the hub
trainer.push_to_hub()

model_save_path = f"./nlp/{repository_id}/final_model.pth"

torch.save(trainer.model.state_dict(), model_save_path)

loaded_model = RobertaClassifier(num_labels=2, config=config)

# Load the model state
model_state_dict = torch.load(model_save_path)

# Update the model instance with the loaded state dictionary
loaded_model.load_state_dict(model_state_dict)

loaded_model.eval()
print("loaded..")

from torch.utils.data import DataLoader

# Assuming dataset["valid"] is already tokenized and ready for input
valid_dataloader = DataLoader(
    dataset["valid"],
    batch_size=1,
    collate_fn=data_collator,
    shuffle=False,
)

from tqdm import tqdm

loaded_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(valid_dataloader):
        # Move batch to the same device as the loaded_model
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels").detach().cpu().numpy()
        outputs = loaded_model(**batch)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()

        # Accumulate predictions and labels
        all_preds.append(preds)
        all_labels.append(labels)

# Calculate metrics

gold = dataset["valid"]["label"]
t1p = precision_score(gold, all_preds)
t1r = recall_score(gold, all_preds)
t1f = f1_score(gold, all_preds)
print("Precision:", t1p)
print("Recall:", t1r)
print("F1:", t1f)
print("-" * 40)
