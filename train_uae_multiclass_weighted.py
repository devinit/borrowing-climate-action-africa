# ! pip install datasets evaluate transformers accelerate huggingface_hub --quiet

# from huggingface_hub import login

# login()

import types
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
import evaluate
import numpy as np

from typing import Optional, Tuple, Union


DEV = False


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


def weighted_forward_distilbert(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    distilbert_output = self.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
    pooled_output = hidden_state[:, 0]  # (bs, dim)
    pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
    pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
    pooled_output = self.dropout(pooled_output)  # (bs, dim)
    logits = self.classifier(pooled_output)  # (bs, num_labels)

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
        output = (logits,) + distilbert_output[1:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=distilbert_output.hidden_states,
        attentions=distilbert_output.attentions,
    )



card = "alex-miller/ODABert"
tokenizer = AutoTokenizer.from_pretrained(card, model_max_length=512)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


unique_labels = [
    'Significant adaptation',
    'Principal adaptation',
    'Significant mitigation',
    'Principal mitigation'
]
id2label = {i: label for i, label in enumerate(unique_labels)}
label2id = {id2label[i]: i for i in id2label.keys()}

def adapt_mit_to_labels(example):
    labels = [0. for i in range(len(unique_labels))]
    if example['climate_adaptation'] == 1:
        label = 'Significant adaptation'
        label_id = label2id[label]
        labels[label_id] = 1.
    if example['climate_adaptation'] == 2:
        label = 'Significant adaptation'
        label_id = label2id[label]
        labels[label_id] = 1.
        label = 'Principal adaptation'
        label_id = label2id[label]
        labels[label_id] = 1.
    if example['climate_mitigation'] == 1:
        label = 'Significant mitigation'
        label_id = label2id[label]
        labels[label_id] = 1.
    if example['climate_mitigation'] == 2:
        label = 'Significant mitigation'
        label_id = label2id[label]
        labels[label_id] = 1.
        label = 'Principal mitigation'
        label_id = label2id[label]
        labels[label_id] = 1.
    
    example = tokenizer(example['text'], truncation=True)
    example['labels'] = labels
    example['class_labels'] = max(labels)
    return example

dataset = load_dataset("csv", data_files="data/uae_for_nlp.csv", split="train")
dataset = dataset.map(adapt_mit_to_labels, num_proc=8)

if DEV:
    dataset = dataset.class_encode_column('class_labels').train_test_split(
        test_size=0.3,
        stratify_by_column="class_labels",
        shuffle=True,
        seed=42
    )
    dataset = dataset['test'].train_test_split(
        test_size=0.2,
        stratify_by_column="class_labels",
        shuffle=True,
        seed=42
    )
else:
    dataset = dataset.class_encode_column('class_labels').train_test_split(
        test_size=0.1,
        stratify_by_column="class_labels",
        shuffle=True,
        seed=42
    )
dataset = dataset.remove_columns(['unique_id', 'text', 'climate_adaptation', 'climate_mitigation', 'class_labels'])


weight_list = list()
print("Weights:")
for label_pos in range(len(unique_labels)):
    label = unique_labels[label_pos]
    positive_filtered_dataset = dataset.filter(lambda example: example['labels'][label_pos] == 1.)
    negative_filtered_dataset = dataset.filter(lambda example: example['labels'][label_pos] == 0.)
    pos_rows = positive_filtered_dataset['train'].num_rows + positive_filtered_dataset['test'].num_rows
    neg_rows = negative_filtered_dataset['train'].num_rows + negative_filtered_dataset['test'].num_rows
    label_weight = neg_rows / pos_rows
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
    problem_type="multi_label_classification",
    classifier_dropout=0.2 # distilbert
    # seq_classif_dropout=0.2 # bert
)
model.forward = types.MethodType(weighted_forward_bert, model)
model.class_weights = weights

training_args = TrainingArguments(
    'uae-climate-multi-classifier-weighted',
    learning_rate=1e-7, # This can be tweaked depending on how loss progresses
    per_device_train_batch_size=8, # These should be tweaked to match GPU VRAM
    per_device_eval_batch_size=8,
    num_train_epochs=10 if DEV else 20,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    load_best_model_at_end=True,
    push_to_hub=not DEV,
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
if not DEV:
    trainer.push_to_hub()
