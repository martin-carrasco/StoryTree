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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import tensorflow as tf
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

# model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained("sileod/roberta-base-discourse-marker-prediction")
model = AutoModelForSequenceClassification.from_pretrained("sileod/roberta-base-discourse-marker-prediction", output_hidden_states=True)
#tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
#model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", output_hidden_states=True)

def mean_pooling(model_output, attention_mask):
    
    #token_embeddings = model_output[0]
    #token_embeddings = torch.stack(model_output.hidden_states)
    token_embeddings = model_output.hidden_states[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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




def generate_embeddings(dataset: datasets.Dataset):
    input_ids = []
    attention_masks = []
    sent= dataset['sentence']
    encoded_dict = tokenizer.batch_encode_plus(
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

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    #labels = torch.tensor(labels)

    return {'input_ids': input_ids, 'attention_masks': attention_masks}

def evaluate_model(dataset):

    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256)
    with torch.no_grad():
        final_embs = []
        for data in tqdm(data_loader):
            outputs = model(data['input_ids'], token_type_ids=None, attention_mask=data['attention_masks'])
            print('Evaluated')
            embs = mean_pooling(outputs, data['attention_masks'])
            print('Pooled')
            '''
            hidden_states = outputs.hidden_states
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings =  token_embeddings.permute(1, 2, 0, 3)

            print(token_embeddings.size())

            embs = make_word_embeddings_1(token_embeddings)
            '''
            final_embs += embs.detach().numpy().tolist()
        return final_embs

def gen_emebddings(ds: datasets.Dataset):
    print('Generating embeddings')
    output_dict = generate_embeddings(ds)

    ds = ds.add_column('input_ids', output_dict['input_ids'].tolist())
    ds_embs = ds.add_column('attention_masks', output_dict['attention_masks'].tolist())

    print('Evaluating the model')
    embs = evaluate_model(ds_embs)
    ds_embs = ds_embs.add_column('embeddings', embs)
    return ds_embs

def eval_score(sents, embs, summary, rouge):
    references = [summary.strip().replace('\n', '')]

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

def split_in_sents(row):

    sentences = []
    article_ids = []
    highlights = []
    for i, art in enumerate(row['article']):
        c_sentences = art.split('.')[:-1]
        sentences += c_sentences
        article_ids += [row['id'][i]] * len(c_sentences)
        highlights += [row['highlights'][i]] * len(c_sentences)


    return {'article_id': article_ids, 'sentence': sentences, 'highlight': highlights}





def run_all_datasets(data_generated=False, embeddings_generated=False):
    rouge = evaluate.load('rouge')
    cnn_data = load_dataset("cnn_dailymail", '2.0.0', data_dir='/var/scratch/mca305')
    cnn_data_test = cnn_data['test']
    cnn_data_validation = cnn_data['validation']

    
    rouge = evaluate.load('rouge')
    eval_methods = {
        'Method': [],
        'Summary': [],
        'Label': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'rougeLsum': []
    }
    cnt = 0

    if not embeddings_generated:
        if not data_generated:
            N = 500
            #small_news = cnn_data_validation.filter(lambda x: len(x['article']) < 800)
            index = random.sample(range(0, len(cnn_data_validation) - 1), N)
            small_DS = cnn_data_validation.select(index)

            small_DS = small_DS.map(split_in_sents, remove_columns=small_DS.column_names, batched=True)
            small_DS.save_to_disk("data_sent_parsed")
            exit()
        else:
            small_DS = datasets.load_from_disk("data_parsed_roberta_edus")
            small_DS = cnn_data_validation.filter(lambda x: x['id'] in small_DS['article_id'])
            small_DS = small_DS.map(split_in_sents, remove_columns=small_DS.column_names, batched=True)
        small_DS.set_format(type='torch', columns=['sentence', 'highlight', 'article_id'])
        small_DS = gen_emebddings(small_DS)
        small_DS.save_to_disk("embeddings_berta_base")
    else:
        small_DS = datasets.load_from_disk("embeddings_berta_base")

    print('Embeddings extracted... now calculating STs')
    unique_ids = list(set(small_DS['article_id']))
    for u_id in tqdm(unique_ids):
        current_ds = small_DS.filter(lambda x: x['article_id'] == u_id)

        sents = current_ds['sentence']
        summary = current_ds['highlight'][0]

        # Eval scores BERT
        k_center, stl_k_center, random_e, k_center_trim, summaries = eval_score(sents, current_ds['embeddings'], summary, rouge)

        # k_center
        eval_methods['Method'].append('K Center')
        eval_methods['Summary'].append(summaries[0])
        eval_methods['Label'].append(summary)
        for k in list(k_center.keys()):
            eval_methods[k].append(k_center[k])

        # st_k_center
        eval_methods['Method'].append('Salient STL')
        eval_methods['Summary'].append(summaries[1])
        eval_methods['Label'].append(summary)
        for k in list(stl_k_center.keys()):
            eval_methods[k].append(stl_k_center[k])

        # random
        eval_methods['Method'].append('Random')
        eval_methods['Summary'].append(summaries[2])
        eval_methods['Label'].append(summary)
        for k in list(random_e.keys()):
            eval_methods[k].append(random_e[k])

        # K Center Trimmed
        eval_methods['Method'].append('K Center STL')
        eval_methods['Summary'].append(summaries[3])
        eval_methods['Label'].append(summary)
        for k in list(k_center_trim.keys()):
            eval_methods[k].append(k_center_trim[k])
        cnt += 1

        ## Eval scores ROBERTA
        print(cnt)

    df = pd.DataFrame.from_dict(eval_methods)
    df.to_csv('scoring_roberta_base_1.csv')




if __name__ == "__main__":
    main()

