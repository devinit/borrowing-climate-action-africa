from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"
card = "./uae-climate-multi-classifier-weighted"
model = AutoModelForSequenceClassification.from_pretrained(card)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(card, model_max_length=512)


def sigmoid(x):
   return 1/(1 + np.exp(-x))


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


def inference(text):
    text_chunks = chunk_by_tokens(text)
    cumulative_results = np.zeros(shape=(len(text_chunks),model.num_labels),dtype=np.float32)
    for i, text_chunk in enumerate(text_chunks):
        inputs = tokenizer(text_chunk, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            predictions = model(**inputs).logits.cpu().numpy()
        logits = sigmoid(predictions).reshape(-1)
        cumulative_results[i,] = logits
    
    mean_results = cumulative_results.mean(axis=0)
    predicted_classes = (mean_results > 0.5)
    results = dict()
    for i, label in enumerate(model.config.label2id.keys()):
        results[label] = (mean_results[i], predicted_classes[i])
    return results


def main():
    res = inference("Renewable energy projects RENEWABLE ENERGY PROJECTS Renewable energy projects Energy generation, renewable sources - multiple technologies")
    print(res)

if __name__ == '__main__':
    main()