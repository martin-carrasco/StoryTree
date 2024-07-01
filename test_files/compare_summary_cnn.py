from evaluate import load
import random
from tqdm import tqdm
import pandas as pd
import evaluate
import numpy as np
from hierarchy_node import HierarchyNode
import summarization
import yaml
from tree_encoding import TreeEncoding
import tensorflow_datasets as tfds
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tensorflow as tf
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

tokenizer = AutoTokenizer.from_pretrained("sileod/roberta-base-discourse-marker-prediction")
model = AutoModelForSequenceClassification.from_pretrained("sileod/roberta-base-discourse-marker-prediction", output_hidden_states=True)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    input_embs = config['Summary']['Embeddings']
    input_text = config['Summary']['InputText']
    input_comp = config['Summary']['CompText']
    scale = config['Visualization']['Scale']
    shortness = config['Visualization']['Short_sents_std']
    closeness = config['Visualization']['Closeness']

    #run(input_text, input_embs, input_comp, scale, shortness, closeness)
    run_all_datasets()



def run(input_text, input_embs, input_comp, scale, shortness, closeness):
    
    with open(input_text, 'r') as f:
        text = f.readlines()[0]
    text = text.split('.')[:-1]

    embs = torch.load(input_embs)
    rouge = evaluate.load('rouge')

    with open(input_comp, 'r') as f:
        real_summary = f.readlines()[0]
    references = [real_summary]

    hierarchy = HierarchyNode(embs)
    hierarchy.calculate_persistence()
    adjacency = hierarchy.h_nodes_adj
    n_leaves = np.min(list(adjacency.keys()))
    n_nodes = np.max(list(adjacency.keys())) + 1
    trimming_summary, trimmed, important = summarization.get_hierarchy_summary_ids(embs)
    summary_1 = summarization.get_k_center_summary(summary_length=len(trimming_summary), embs=embs, sents=text)
    summary_1 = ['.'.join(summary_1)]

    output = rouge.compute(predictions=summary_1, references=references)


    summary_2 = ['.'.join(summarization.get_hierarchy_summary(embs, text)[0])]
    output_2 = rouge.compute(predictions=summary_2, references=references)

    summary_random = ['.'.join(np.random.choice(text, 5))]
    output_3 = rouge.compute(predictions=summary_random, references=references)

    print(f'k-center: {output}')
    print(f'STL k-center: {output_2}')
    print(f'random: {output_3}')



def generate_embeddings(row):
    input_ids = []
    attention_masks = []
    sent = row['sentence']
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 85,           # Pad & truncate all sentences.
                        padding='max_length',
                        truncation=True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                )

    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0).squeeze()
    attention_masks = torch.cat(attention_masks, dim=0).squeeze()
    #labels = torch.tensor(labels)

    return {'input_ids': input_ids, 'attention_masks': attention_masks }
def evaluate_model(dataset):

    model.eval()
    with torch.no_grad():
        outputs = model(dataset['input_ids'], token_type_ids=None, attention_mask=dataset['attention_masks'])
        hidden_states = torch.stack(outputs.hidden_states)
        mean_pooled = hidden_states.sum(axis=2) / dataset['attention_masks'].sum(axis=-1).unsqueeze(-1)
        mean_pooled = torch.mean(mean_pooled[:2], dim=0)
        #print(mean_pooled.shape)
        embs = mean_pooled
        '''
        hidden_states = outputs.hidden_states
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings =  token_embeddings.permute(1, 2, 0, 3)

        print(token_embeddings.size())

        embs = make_word_embeddings_1(token_embeddings)
        '''
        return embs.detach().numpy()

def eval_score(text, summary, rouge):
    sents = text.split('.')[:-1]
    references = [summary.strip().replace('\n', '')]
    ds_sents = datasets.Dataset.from_dict({'sentence': sents})
    ds_sents.set_format(type='torch', columns=['sentence'])
    ds_embs = ds_sents.map(generate_embeddings)
    embs = evaluate_model(ds_embs)
    #embs = np.array(embs).reshape(-1, 1)
    hierarchy = HierarchyNode(embs)
    hierarchy.calculate_persistence()
    adjacency = hierarchy.h_nodes_adj
    #n_leaves = np.min(list(adjacency.keys()))
    #n_nodes = np.max(list(adjacency.keys())) + 1
    trimming_summary, trimmed, important = summarization.get_hierarchy_summary_ids(embs)


    summary_1 = summarization.get_k_center_summary(summary_length=len(trimming_summary), embs=embs, sents=sents)
    summary_1 = ['.'.join(summary_1)]

    output = rouge.compute(predictions=summary_1, references=references)


    summary_2 = ['.'.join(summarization.get_hierarchy_summary(embs, sents)[0])]
    output_2 = rouge.compute(predictions=summary_2, references=references)

    summary_random = ['.'.join(np.random.choice(sents, 5))]
    output_3 = rouge.compute(predictions=summary_random, references=references)

    summary_4 = ['.'.join(summarization.get_k_center_summary_after_trimming(summary_length=len(trimming_summary), adjacency=adjacency, embs=embs, sents=sents, trimmed=trimmed))]
    output_4 = rouge.compute(predictions=summary_4, references=references)


    return output, output_2, output_3, output_4, (summary_1, summary_2, summary_random, summary_4)



def run_all_datasets():
    rouge = evaluate.load('rouge')
    cnn_data = load_dataset("cnn_dailymail", '1.0.0')
    cnn_data_test = cnn_data['test']
    cnn_data_validation = cnn_data['validation']
    rouge = evaluate.load('rouge')
    eval_methods = {
        'Method': [],
        'Summary'
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'rougeLsum': []
    }
    cnt = 0
    N = 10 
    index = random.sample(range(0, len(cnn_data_validation) - 1), N)
    small_DS = cnn_data_validation.select(index)
    # DS = small_DS.map(generate_embeddings)
    # DS.set_format(type='torch', columns=['input_ids', 'attention_masks', 'article', 'highlights'])
    # embeddings = evaluate_model(DS)
    # df_pandas = pd.DataFrame(DS)
    # df_pandas['embeddings'] = embeddings.tolist()
    # print('HELLO')
    for element in tqdm(small_DS):
        text = element['article']
        summary = element['highlights']
        k_center, stl_k_center, random_e, k_center_trim, summaries = eval_score(text, summary, rouge)

        # k_center
        eval_methods['Method'].append('K Center')
        eval_methods['Summary'].append(summaries[0])
        for k in list(k_center.keys()):
            eval_methods[k].append(k_center[k])

        # st_k_center
        eval_methods['Method'].append('Salient STL')
        eval_methods['Summary'].append(summaries[1])
        for k in list(stl_k_center.keys()):
            eval_methods[k].append(stl_k_center[k])

        # random
        eval_methods['Method'].append('Random')
        eval_methods['Summary'].append(summaries[2])
        for k in list(random_e.keys()):
            eval_methods[k].append(random_e[k])

        # K Center Trimmed
        eval_methods['Method'].append('K Center STL')
        eval_methods['Summary'].append(summaries[3])
        for k in list(k_center_trim.keys()):
            eval_methods[k].append(k_center_trim[k])
        cnt += 1
        print(cnt)

    df = pd.DataFrame.from_dict(eval_methods)
    df.to_csv('scoring_roberta_1.csv')




if __name__ == "__main__":
    main()


'''
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
'''