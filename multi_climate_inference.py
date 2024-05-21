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
    'adapt',
    'adaptation',
    'adaptative',
    'adaptive',
    'afforestation',
    'agri',
    'agricole',
    'agricoles',
    'agricultural',
    'agriculture',
    'agro',
    'agroecological',
    'agroecology',
    'agroforestry',
    'agrícola',
    'ambiental',
    'anthracnose',
    'aquaponics',
    'baterias',
    'batería',
    'batterie',
    'batteries',
    'battery',
    'bio',
    'biodiversity',
    'biodiversité',
    'bioenergy',
    'biomasa',
    'biomass',
    'biomasse',
    'bioremediation',
    'bosque',
    'bosques',
    'carbon',
    'carbone',
    'catastrophe',
    'catastrophes',
    'catástrofe',
    'catástrofes',
    'ccnucc',
    'cement',
    'cgiar',
    'charcoal',
    'chemin de fer',
    'chemins de fer',
    'clean',
    'climat',
    'climate',
    'climatic',
    'climatico',
    'climatique',
    'climatiques',
    'climático',
    'coastal',
    'coffee',
    'compost',
    'composting',
    'conservation',
    'consistently',
    'contribución determinada nacional',
    'contribution determinee nationale',
    'cook',
    'cooking',
    'coping',
    'crop',
    'crops',
    'cultivo',
    'cultivos',
    'culture',
    'cultures',
    "d'électricité",
    'decarbonization',
    'deforestación',
    'deforestation',
    'degradation',
    'depleted',
    'depletion',
    'desalination',
    'desert',
    'desertification',
    'desierto',
    'dessalement',
    'disaster',
    'disaster risk',
    'disasters',
    'diverse',
    'diversified',
    'diversifying',
    'drm',
    'drought',
    'drr',
    'dryland',
    'drylands',
    'durable',
    'durablement',
    'durables',
    'déchets',
    'déforestation',
    'désert',
    'early warning',
    'ecological',
    'ecologique',
    'ecologiques',
    'ecology',
    'ecosistema',
    'ecosistemas',
    'ecosystem',
    'ecosystemen',
    'ecosystems',
    'efficiency',
    'electric',
    'electricity',
    'electricité',
    'electrification',
    'electrique',
    'elektronicznego',
    'elektrycznej',
    'elephant',
    'elephants',
    'emision',
    'emisiones',
    'emisión',
    'emission',
    'emissions',
    'energetique',
    'energi',
    'energia',
    'energie',
    'energies',
    'energy',
    'env',
    'environment',
    'environmental',
    'environnement',
    'environnementales',
    'environnementaux',
    'eolica',
    'eolienne',
    'exhaust',
    'eólica',
    'farm',
    'farmer',
    'farmers',
    'farms',
    'ferrocarril',
    'flloca',
    'flood',
    'flooding',
    'floods',
    'forest',
    'forestal',
    'forestiere',
    'forestry',
    'forests',
    'forêt',
    'forêts',
    'fotowoltaiczne',
    'fuel',
    'gas',
    'gases',
    'gaz',
    'gazów',
    'gcf',
    'geotermia',
    'geothermal',
    'ghg',
    'grazing',
    'green',
    'greenhouse',
    'greening',
    'grid',
    'grids',
    'grille',
    'géothermique',
    'harvest',
    'harvests',
    'hydro',
    'hydroelectric',
    'hydroelectriques',
    'hydropower',
    'hydroélectrique',
    'ifad',
    'interconnection',
    'interconnexion',
    'iucn', "l'électricité", 'land',
    'lcf',
    'land',
    'lands',
    'lignes',
    'limpio',
    'lines',
    'lowlands',
    'líneas',
    'mangrove',
    'mangroves',
    'marine',
    'meteorological',
    'meteorologique',
    'meteorologiques',
    'meteorology',
    'meteorológica',
    'meteorológicos',
    'mini-reseaux',
    'mitigación',
    'mitigated',
    'mitigating',
    'mitigation',
    'montane',
    'mw',
    'météorologique',
    'météorologiques',
    'nationally determined contribution',
    'nationally determined contributions',
    'natural',
    'nature',
    'naturelles',
    'ndc',
    'ocean',
    'odnawialne',
    'odpadów',
    'organic',
    'pastorales',
    'pays sec',
    'permaculture',
    'photovoltaic',
    'plant',
    'plantacji',
    'plantation',
    'plantations',
    'planting',
    'plants',
    'power',
    'preparedness',
    'propre',
    'pv',
    'railway',
    'railways',
    'rains',
    'recycle',
    'recycled',
    'recycling',
    'redd',
    'reforestation',
    'remediation',
    'remote sensing',
    'renewable',
    'renewables',
    'renouvelable',
    'renouvelables',
    'residuos',
    'resilience',
    'resilient',
    'restoration',
    'retrofitting',
    'reuse',
    'rice',
    'rio',
    'risk reduction',
    'rolnej',
    'rolniczej',
    'réseau',
    'réseaux',
    'résilience',
    'satellite',
    'sea',
    'seascape',
    'season',
    'sequestration',
    'soil',
    'solaire',
    'solaires',
    'solar',
    'solarization',
    'sols',
    'soneczn',
    'spalinowych',
    'species',
    'suelos',
    'sustainability',
    'sustainable',
    'sustainably',
    'sustentablemente',
    'sequía',
    'sécheresse',
    'terre',
    'terres',
    'territorial',
    'territory',
    'tierra',
    'tierras',
    'tolerant',
    'transmisión',
    'transmission',
    'tropical',
    'uicn',
    'unfccc',
    'upcycling',
    'vegetation',
    'verte',
    'vias ferreas',
    'waste',
    'water',
    'watershed',
    'weather',
    'weatherization',
    'wildlife',
    'wind',
    'windpower',
    'zielona',
    'zone',
    'zones',
    'écologique',
    'écologiques',
    'écosystème',
    'écosystèmes',
    'écosystémiques',
    'électrique',
    'émission',
    'émissions',
    'énergie',
    'énergétique',
    'éolienne'
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
    keyword_match = False

    if text is not None:
        keyword_match = CLIMATE_REGEX.search(text) is not None
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
    example['Climate keyword match'] = keyword_match
    return example

def main():
    text_cols = ['project_title', 'short_description', 'long_description']
    dataset = pd.read_csv("large_data/crs_2022.csv")
    dataset_screened = dataset[dataset['climate_adaptation'].isin([0, 1, 2]) | dataset['climate_mitigation'].isin([0, 1, 2])]
    dataset_unscreened = dataset[dataset['climate_adaptation'].isnull() & dataset['climate_mitigation'].isnull()]
    dataset_text = dataset_unscreened[text_cols]
    dataset_text = Dataset.from_pandas(dataset_text)
    dataset_text = dataset_text.map(map_columns, remove_columns=text_cols)
    dataset_text = pd.DataFrame(dataset_text)
    dataset_unscreened = pd.concat([dataset_unscreened.reset_index(drop=True), dataset_text.reset_index(drop=True)], axis=1)
    dataset = pd.concat([dataset_screened, dataset_unscreened])
    dataset.to_csv('large_data/crs_2022_predictions.csv', index=False)


if __name__ == '__main__':
    main()

