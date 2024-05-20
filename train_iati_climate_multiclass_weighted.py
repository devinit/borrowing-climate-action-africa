# ! pip install datasets evaluate transformers accelerate huggingface_hub --quiet

# from huggingface_hub import login

# login()

import types
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
import evaluate
import numpy as np
import pandas as pd
from collections import Counter
import math

from typing import Optional, Tuple, Union


card = "alex-miller/ODABert"
tokenizer = AutoTokenizer.from_pretrained(card, model_max_length=512)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def weighted_forward_bert(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    pooled_output = outputs[1]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights)
            loss = loss_fct(logits, labels)
    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

dataset = load_dataset("devinitorg/iati-policy-markers", split="train")

dataset = dataset.filter(lambda example: example["climate_adaptation"] or example["climate_mitigation"])
dataset = dataset.filter(lambda example: example["climate_adaptation_sig"] in [0, 1, 2] or example["climate_mitigation_sig"] in [0, 1, 2])
dataset = dataset.filter(lambda example: example["text"] != "" and example["text"] is not None and len(example["text"]) > 10)
cols_to_remove = dataset.column_names
cols_to_remove.remove("text")
cols_to_remove.remove("climate_adaptation_sig")
cols_to_remove.remove("climate_mitigation_sig")
dataset = dataset.remove_columns(cols_to_remove)

# De-duplicate
df = pd.DataFrame(dataset)
print(df.shape)
df = df.drop_duplicates(subset=['text'])
print(df.shape)
dataset = Dataset.from_pandas(df, preserve_index=False)

def create_class_labels(example):
    adaptation_label = int(example['climate_adaptation_sig'])
    mitigation_label = int(example['climate_mitigation_sig'])
    if adaptation_label > 2:
        adaptation_label = 2
    if mitigation_label > 2:
        mitigation_label = 2
    example['class_labels'] = '{}{}'.format(adaptation_label, mitigation_label)
    return example

# Add removable column to stratify
dataset = dataset.map(create_class_labels)
count = Counter()
count.update(dataset['class_labels'])
print(count)

# Remove some of the full negative examples
full_negative = dataset.filter(lambda example: example['class_labels'] == '00')
some_positive = dataset.filter(lambda example: example['class_labels'] != '00')
full_negative = full_negative.shuffle(seed=42).select(range(math.floor(some_positive.num_rows)))
dataset = concatenate_datasets([full_negative, some_positive])
count = Counter()
count.update(dataset['class_labels'])
print(count)

# Stratify and split
dataset = dataset.class_encode_column('class_labels').train_test_split(
    test_size=0.2,
    stratify_by_column="class_labels",
    shuffle=True,
    seed=42
)
dataset.remove_columns("class_labels")

unique_labels = [
    "Climate adaptation - significant objective",
    "Climate adaptation - principal objective",
    "Climate mitigation - significant objective",
    "Climate mitigation - principal objective"
]
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {id2label[i]: i for i in id2label.keys()}

def preprocess_function(example):
    adaptation_label = example['climate_adaptation_sig']
    mitigation_label = example['climate_mitigation_sig']
    labels = [0. for i in range(len(unique_labels))]
    if adaptation_label == 1:
        label = "Climate adaptation - significant objective"
        labels[label2id[label]] = 1.
    elif adaptation_label >= 2:
        label = "Climate adaptation - significant objective"
        labels[label2id[label]] = 1.
        label = "Climate adaptation - principal objective"
        labels[label2id[label]] = 1.

    if mitigation_label == 1:
        label = "Climate mitigation - significant objective"
        labels[label2id[label]] = 1.
    elif mitigation_label >= 2:
        label = "Climate mitigation - significant objective"
        labels[label2id[label]] = 1.
        label = "Climate mitigation - principal objective"
        labels[label2id[label]] = 1.

    example = tokenizer(example['text'], truncation=True)
    example['labels'] = labels
    return example

dataset = dataset.map(preprocess_function, remove_columns=['climate_adaptation_sig', 'climate_mitigation_sig'])

weight_list = list()
total_rows = dataset['train'].num_rows + dataset['test'].num_rows
print("Weights:")
for label in unique_labels:
    label_idx = label2id[label]
    positive_filtered_dataset = dataset.filter(lambda example: example['labels'][label_idx] == 1.)
    negative_filtered_dataset = dataset.filter(lambda example: example['labels'][label_idx] == 0.)
    pos_label_rows = positive_filtered_dataset['train'].num_rows + positive_filtered_dataset['test'].num_rows
    neg_label_rows = negative_filtered_dataset['train'].num_rows + negative_filtered_dataset['test'].num_rows
    label_weight = neg_label_rows / pos_label_rows
    weight_list.append(label_weight)
    print("{}: {}".format(label, label_weight))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
weights = torch.tensor(weight_list)
weights = weights.to(device)

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
   return 1/(1 + np.exp(-x))


def compute_metrics(eval_pred):
   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(int).reshape(-1)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


model = AutoModelForSequenceClassification.from_pretrained(
    card,
    num_labels=len(id2label.keys()), 
    id2label=id2label,
    label2id=label2id, 
    problem_type="multi_label_classification"
)
model.forward = types.MethodType(weighted_forward_bert, model)
model.class_weights = weights

training_args = TrainingArguments(
    'iati-climate-multi-classifier-weighted2',
    learning_rate=2e-6, # This can be tweaked depending on how loss progresses
    per_device_train_batch_size=24, # These should be tweaked to match GPU VRAM
    per_device_eval_batch_size=24,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    load_best_model_at_end=True,
    push_to_hub=True,
    save_total_limit=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()