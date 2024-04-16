from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from scipy.special import softmax
import math
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
card = "alex-miller/iati-climate-classifier"
model = AutoModelForSequenceClassification.from_pretrained(card)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(card, model_max_length=512)
positive_class='1.0'


def chunk_by_tokens(input_text, model_max_size=512):
    chunks = list()
    tokens = tokenizer.encode(input_text)
    token_length = len(tokens)
    if token_length <= model_max_size:
        return [input_text]
    desired_number_of_chunks = math.ceil(token_length / model_max_size)
    calculated_chunk_size = math.ceil(token_length / desired_number_of_chunks)
    for i in range(0, token_length, calculated_chunk_size):
        chunks.append(tokenizer.decode(tokens[i:i + calculated_chunk_size]))
    return chunks


def inference(example):
    climate_predicted = False
    final_positive_class_confidence = 0
    text_chunks = chunk_by_tokens(example['text'])
    for text_chunk in text_chunks:
        inputs = tokenizer(text_chunk, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            predictions = model(**inputs)
        
        logits = predictions.logits.cpu().detach().numpy()[0]
        predicted_confidences = softmax(logits, axis=0)
        predicted_class_id = np.argmax(logits)
        predicted_class = model.config.id2label[predicted_class_id]
        climate_predicted = climate_predicted or predicted_class == positive_class
        positive_class_id = model.config.label2id[positive_class]
        positive_class_confidence = predicted_confidences[positive_class_id]
        final_positive_class_confidence = max(final_positive_class_confidence, positive_class_confidence)

    example['climate_predicted'] = climate_predicted
    example['climate_confidence'] = final_positive_class_confidence
    return example


def main():
    dataset = load_dataset('csv', data_files='data/filtered_gcdf_au_loan_2022.csv')
    dataset = dataset['train'].map(inference)
    dataset.to_csv('data/climate_classified_gcdf.csv')


if __name__ == '__main__':
    main()