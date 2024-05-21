from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd


def remove_string_special_characters(s):
    # removes special characters with ' '
    stripped = re.sub(r'[^\w\s]', ' ', s)

    # Change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
            return stripped.lower()
    
def remove_stop_words(s):
    stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish')) | set(stopwords.words('french')) | set(stopwords.words('german'))
    try:
        return ' '.join([word for word in nltk.word_tokenize(s) if word not in stop_words])
    except TypeError:
         return ' '
    

def features(vectorizer, result):
    feature_array = np.array(vectorizer.get_feature_names_out())
    flat_values = result.toarray().flatten()
    zero_indices = np.where(flat_values == 0)[0]
    tfidf_sorting = np.argsort(flat_values)[::-1]
    masked_tfidf_sorting = np.delete(tfidf_sorting, zero_indices, axis=0)

    return feature_array[masked_tfidf_sorting].tolist()


def concat_text(example):
    text_vec = [example['project_title'], example['short_description'], example['long_description']]
    text_vec = [string for string in text_vec if string is not None]
    text = " ".join(text_vec)
    example['text'] = remove_stop_words(remove_string_special_characters(text))
    return example


text_cols = ['project_title', 'short_description', 'long_description', 'climate_adaptation', 'climate_mitigation']
dataset = pd.read_csv("large_data/crs_2022.csv")
dataset = dataset[text_cols]
dataset = Dataset.from_pandas(dataset)
dataset = dataset.map(concat_text, num_proc=8)

not_climate = dataset.filter(lambda example: example['climate_mitigation'] == 0 and example['climate_adaptation'] == 0)
climate = dataset.filter(lambda example: example['climate_adaptation'] == 2 or example['climate_mitigation'] == 2)

vectorizer = TfidfVectorizer(ngram_range=(1, 1))
vectorizer.fit(dataset['text'])

not_climate_result = vectorizer.transform([" ".join(not_climate['text'])])
top_not_climate = features(vectorizer, not_climate_result)
top_not_climate = top_not_climate[:250]

climate_result = vectorizer.transform([" ".join(climate['text'])])
top_climate = features(vectorizer, climate_result)
top_climate = [word for word in top_climate if word not in top_not_climate]
top_climate = top_climate[:500]
# top_climate.sort()

print(top_climate)