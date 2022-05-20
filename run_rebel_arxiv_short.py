import spacy
from rebel import spacy_component
import json
import pandas as pd
import time
import re
from tqdm import tqdm
from text_preprocessing import *
# import dask.bag as db

nlp_stem = spacy.load("en_core_web_sm")

nlp = spacy.load("en_core_web_lg")
print("S")
nlp.add_pipe("rebel", after="senter", config={
    'device':-1, # Number of the GPU, -1 if want to use CPU
    'model_name':'Babelscape/rebel-large'} # Model used, will default to 'Babelscape/rebel-large' if not given
    )

papers = pd.read_pickle('../data/Arxiv/arxiv_papers_processed.pickle')

out = {}
start_time = time.time()

for idx, input_text in tqdm(enumerate(papers['abstract'].iloc[:2000].values), total=len(papers['abstract'].iloc[:2000].values)):
    try:
        
        input_text = process_sent(input_text)                 
        # print(input_sentence)
        doc = nlp_stem(input_text)
        doc = nlp(" ".join([token.lemma_ for token in doc]))

        for value, rel_dict in doc._.rel.items():
            out[f"{input_text}->{value}"] = {'relation': rel_dict['relation'], 'head': str(rel_dict['head_span']), 'tail': str(rel_dict['tail_span'])}
    except Exception as e:
        print(e)
        pass
exec_time = time.time() - start_time
out['time'] = exec_time
with open('arxiv_rel_extracted_small.json', 'w') as fp:
    json.dump(out, fp)