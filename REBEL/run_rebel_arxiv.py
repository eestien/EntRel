import spacy
from rebel import spacy_component
import json
import pandas as pd
import time
import re
from tqdm import tqdm
from text_preprocessing import *
from nltk.stem import 	WordNetLemmatizer
# import dask.bag as db
import nltk
from numpy.random import default_rng

nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load("en_core_web_lg")
print("S")
nlp.add_pipe("rebel", after="senter", config={
    'device':-1, # Number of the GPU, -1 if want to use CPU
    'model_name':'Babelscape/rebel-large'} # Model used, will default to 'Babelscape/rebel-large' if not given
    )

papers = pd.read_pickle('../data/Arxiv/arxiv_papers_processed.pickle')

out = {}
start_time = time.time()


wordnet_lemmatizer = WordNetLemmatizer()

rng = default_rng(seed=42)
random_articles = rng.choice(len(papers['abstract'].values), size=250000, replace=False)


for idx in tqdm(random_articles, total=len(random_articles)):
    try:
        input_text = papers.iloc[idx, 1]
        input_text = process_sent(input_text)
        input_text = input_text.split('.')
        
        text = ''
        for i in input_text:
            if i !='':
                sent = ''
                for j in i.split(' '):
                    sent+= f'{wordnet_lemmatizer.lemmatize(j)} '
                text+=f"{sent}."
        input_text = text
        doc = nlp(input_text)
        

        for value, rel_dict in doc._.rel.items():
            out[f"{input_text}->{value}"] = {'relation': rel_dict['relation'], 'head': str(rel_dict['head_span']), 'tail': str(rel_dict['tail_span'])}
    except Exception as e:
        print(e)
        pass
exec_time = time.time() - start_time
out['time'] = exec_time
with open('arxiv_rel_extracted_250K.json', 'w') as fp:
    json.dump(out, fp)