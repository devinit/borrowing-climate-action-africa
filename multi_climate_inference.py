from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
import math
import re


global TOKENIZER
global DEVICE
global MODEL
TOKENIZER = AutoTokenizer.from_pretrained('alex-miller/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = AutoModelForSequenceClassification.from_pretrained("alex-miller/iati-climate-multi-classifier-weighted2")
MODEL = MODEL.to(DEVICE)

climate_keywords = [
    'adaptation',
    'adaptive',
    'adaptative',
    'adapt',
    'afforestation',
    'agro',
    'agroecology',
    'agri',
    'agricole',
    'agricoles',
    'agriculture',
    'agroecological',
    'agroforestry',
    'anthracnose',
    'aquaponics',
    'bio',
    'biodiversity',
    'biodiversité',
    'bioenergy',
    'biomass',
    'bioremediation',
    'carbon',
    'carbone',
    'ccnucc',
    'cement',
    'cgiar',
    'charcoal',
    'climate',
    'climatic',
    'coastal',
    'coffee',
    'cook',
    'cooking',
    'compost',
    'composting',
    'conservation',
    'consistently',
    'coping',
    'decarbonization',
    'desert',
    'desertification',
    'depletion',
    'depleted',
    'desalination',
    'dessalement',
    'disasters',
    'disaster',
    'diversifying',
    'diverse',
    'diversified',
    'drm',
    'drr',
    'disaster risk',
    'drought',
    'durable',
    'durables',
    'early warning',
    'ecologique',
    'ecologiques',
    'écologiques',
    'écologique',
    'electricité',
    'electrique',
    'électrique',
    'ecological',
    'ecology',
    'ecosystem',
    'ecosystemen',
    'environnement',
    'environment',
    'environmental',
    'env',
    'exhaust',
    'elephant',
    'elephants',
    'electricity',
    'electric',
    'electrification',
    'elektronicznego',
    'elektrycznej',
    'energy',
    'energies',
    'energi',
    'energia',
    'énergétique',
    'énergie',
    'efficiency',
    'farm',
    'farms',
    'farmer',
    'farmers',
    'flloca',
    'flood',
    'floods',
    'flooding',
    'fotowoltaiczne',
    'forest',
    'forests',
    'forestal',
    'forestry',
    'forêts',
    'fuel',
    'gas',
    'gases',
    'gazów',
    'gcf',
    'ghg',
    'grazing',
    'green',
    'greenhouse',
    'greening',
    'harvest',
    'harvests',
    'hydro',
    'hydropower',
    'hydroélectrique',
    'hydroelectric',
    'ifad',
    'iucn',
    'interconnexion',
    'interconnection',
    'land',
    'lcf',
    'lowlands',
    'mangrove',
    'mangroves',
    'marine',
    'météorologiques',
    'météorologique',
    'meteorologiques',
    'meteorologique',
    'meteorology',
    'meteorological',
    'mitigación',
    'mitigating',
    'mitigation',
    'mitigated',
    'montane',
    'nature',
    'natural',
    'ocean',
    'odpadów',
    'odnawialne',
    'organic',
    'pastorales',
    'permaculture',
    'photovoltaic',
    'plant',
    'plantacji',
    'plants',
    'planting',
    'plantation',
    'plantations',
    'power',
    'preparedness',
    'pv',
    'rains',
    'recycle',
    'recycled',
    'recycling',
    'reforestation',
    'remediation',
    'remote sensing',
    'renewable',
    'renewables',
    'renouvelables',
    'renouvelable',
    'resilience',
    'resilient',
    'résilience',
    'restoration',
    'retrofitting',
    'reuse',
    'rice',
    'risk reduction',
    'rio',
    'rolnej',
    'rolniczej',
    'satellite',
    'sea',
    'seascape',
    'season',
    'sequestration',
    'species',
    'spalinowych',
    'soil',
    'solar',
    'solaire',
    'soneczn',
    'solarization',
    'sustainability',
    'sustainable',
    'territorial',
    'territory',
    'tolerant',
    'transmission',
    'transmisión',
    'tropical',
    'uicn',
    'unfccc',
    'upcycling',
    'vegetation',
    'verte',
    'waste',
    'water',
    'watershed',
    'weatherization',
    'weather',
    'wildlife',
    'wind',
    'windpower',
    'zielona',
    'écosystémiques',
]
climate_regex_string = '|'.join([r'\b%s\b' % word for word in climate_keywords])
CLIMATE_REGEX = re.compile(climate_regex_string, re.I)


def remove_string_special_characters(s):
    # removes special characters with ' '
    stripped = re.sub(r'[^\w\s]', ' ', s)

    # Change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
        return stripped.lower()


def sigmoid(x):
   return 1/(1 + np.exp(-x))


def chunk_by_tokens(input_text, model_max_size=512):
    chunks = list()
    tokens = TOKENIZER.encode(input_text)
    token_length = len(tokens)
    if token_length <= model_max_size:
        return [input_text]
    desired_number_of_chunks = math.ceil(token_length / model_max_size)
    calculated_chunk_size = math.ceil(token_length / desired_number_of_chunks)
    for i in range(0, token_length, calculated_chunk_size):
        chunks.append(TOKENIZER.decode(tokens[i:i + calculated_chunk_size]))
    return chunks


def inference(model, inputs):
    predictions = model(**inputs)

    logits = predictions.logits.cpu().detach().numpy()[0]
    predicted_confidences = sigmoid(logits)
    predicted_classes = (predicted_confidences > 0.5)

    return predicted_classes, predicted_confidences

def map_columns(example):
    textual_data_list = [
        example['project_title'],
        example['short_description'],
        example['long_description']
    ]
    textual_data_list = [str(textual_data) for textual_data in textual_data_list if textual_data is not None]
    text = remove_string_special_characters(" ".join(textual_data_list))

    predictions = {
        "Climate adaptation - significant objective": [False, 0],
        "Climate adaptation - principal objective": [False, 0],
        "Climate mitigation - significant objective": [False, 0],
        "Climate mitigation - principal objective": [False, 0]
    }

    text_chunks = chunk_by_tokens(text)
    for text_chunk in text_chunks:
        inputs = TOKENIZER(text_chunk, return_tensors="pt", truncation=True).to(DEVICE)
        model_pred, model_conf = inference(MODEL, inputs)
        predictions['Climate adaptation - significant objective'][0] = predictions['Climate adaptation - significant objective'][0] or model_pred[0]
        predictions['Climate adaptation - significant objective'][1] = max(predictions['Climate adaptation - significant objective'][1], model_conf[0])
        predictions['Climate adaptation - principal objective'][0] = predictions['Climate adaptation - principal objective'][0] or model_pred[1]
        predictions['Climate adaptation - principal objective'][1] = max(predictions['Climate adaptation - principal objective'][1], model_conf[1])
        predictions['Climate mitigation - significant objective'][0] = predictions['Climate mitigation - significant objective'][0] or model_pred[2]
        predictions['Climate mitigation - significant objective'][1] = max(predictions['Climate mitigation - significant objective'][1], model_conf[2])
        predictions['Climate mitigation - principal objective'][0] = predictions['Climate mitigation - principal objective'][0] or model_pred[3]
        predictions['Climate mitigation - principal objective'][1] = max(predictions['Climate mitigation - principal objective'][1], model_conf[3])


    example['Climate adaptation - significant objective predicted'] = predictions['Climate adaptation - significant objective'][0]
    example['Climate adaptation - significant objective confidence'] = predictions['Climate adaptation - significant objective'][1]
    example['Climate adaptation - principal objective predicted'] = predictions['Climate adaptation - principal objective'][0]
    example['Climate adaptation - principal objective confidence'] = predictions['Climate adaptation - principal objective'][1]
    example['Climate mitigation - significant objective predicted'] = predictions['Climate mitigation - significant objective'][0]
    example['Climate mitigation - significant objective confidence'] = predictions['Climate mitigation - significant objective'][1]
    example['Climate mitigation - principal objective predicted'] = predictions['Climate mitigation - principal objective'][0]
    example['Climate mitigation - principal objective confidence'] = predictions['Climate mitigation - principal objective'][1]
    example['Climate keyword match'] = CLIMATE_REGEX.search(text) is not None
    return example

def main():
    text_cols = ['project_title', 'short_description', 'long_description']
    dataset = pd.read_csv("large_data/crs_2022.csv")
    dataset = dataset.drop_duplicates(subset=['project_title'])
    dataset_text = dataset[text_cols]
    dataset_text = Dataset.from_pandas(dataset_text)
    dataset_text = dataset_text.map(map_columns, remove_columns=text_cols)
    dataset_text = pd.DataFrame(dataset_text)
    dataset = pd.concat([dataset.reset_index(drop=True), dataset_text.reset_index(drop=True)], axis=1)
    dataset.to_csv('large_data/crs_2022_predictions.csv', index=False)


if __name__ == '__main__':
    main()

