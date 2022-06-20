import pandas as pd
import spacy
from rebel import spacy_component
from text_preprocessing import *
import json
from tqdm import tqdm
import time
CLEANR = re.compile('<.*?>') 
medium_data = pd.read_csv('../data/medium_data.csv')

medium_data['title'] = medium_data['title'].apply(lambda x: x.replace(u'\xa0',u' '))
medium_data['title'] = medium_data['title'].apply(lambda x: x.replace('\u200a',' '))

nlp = spacy.load("en_core_web_lg")
print("S")
nlp.add_pipe("rebel", after="senter", config={
    'device':-1, # Number of the GPU, -1 if want to use CPU
    'model_name':'Babelscape/rebel-large'} # Model used, will default to 'Babelscape/rebel-large' if not given
    )

out = {}
start_time = time.time()

for idx, input_sentence in tqdm(enumerate(medium_data['title'].values), total=len(medium_data['title'].values)):
    try:
        input_sentence = re.sub(CLEANR, '', input_sentence)
        input_sentence = process_sent(input_sentence)                 
        doc = nlp(input_sentence)

        for value, rel_dict in doc._.rel.items():
            out[f"{input_sentence}->{value}"] = {'relation': rel_dict['relation'], 'head': str(rel_dict['head_span']), 'tail': str(rel_dict['tail_span'])}
    except Exception as e:
        pass
    if idx == 15:
        break
exec_time = time.time() - start_time
out['time'] = exec_time
with open('medium_rel_extracted_from_cmd_small.json', 'w') as fp:
    json.dump(out, fp)