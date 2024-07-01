from typing import Dict, Iterable, List
import torch

import numpy as np
from sentence_transformers import SentenceTransformer
import util
import w2v_embedding as w2v_emb
from hierarchy_node import HierarchyNode
import summarization
import yaml
from tree_encoding import TreeEncoding
from transformers import BertTokenizer, BertModel

import torch
import nltk.data
import nltk
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    input = config['Visualization']['Input']
    output = config['Visualization']['Output']
    scale = config['Visualization']['Scale']
    shortness = config['Visualization']['Short_sents_std']
    closeness = config['Visualization']['Closeness']

    run(input, output, scale, shortness, closeness)

MODE = 0
def run(input, output, scale, shortness, closeness):
    
    with open(input, 'r') as file:
        text = file.read()
        
    if not MODE:
        model = SentenceTransformer('all-mpnet-base-v2')
        #sentences = nltk.sent_tokenize(text)
        sentences = text.split('\n')
        embs = model.encode(sentences)
    if MODE:
        embs = torch.load('input/marcus_EDU.pt').detach().numpy()




    hierarchy = HierarchyNode(embs)
    hierarchy.calculate_persistence()
    adjacency = hierarchy.h_nodes_adj
    n_leaves = np.min(list(adjacency.keys()))
    n_nodes = np.max(list(adjacency.keys())) + 1
    trimming_summary, trimmed, important = summarization.get_hierarchy_summary_ids(embs)
    kcenter_summary = summarization.get_k_center_summary_ids(summary_length=len(trimming_summary), embs=embs)
    
    TE = TreeEncoding(adjacency=adjacency, births=hierarchy.birth_time, n_leaves=n_leaves, n_nodes=n_nodes, trimming_summary=trimming_summary,
                kcenter_summary=kcenter_summary, trimmed=trimmed, important=important, SCALE=scale)
    TE.draw_tree(output)

if __name__ == "__main__":
    main()
