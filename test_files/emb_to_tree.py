from typing import Dict, Iterable, List
import torch

import numpy as np
from hierarchy_node import HierarchyNode
import summarization
import yaml
from tree_encoding import TreeEncoding

import torch
import nltk
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    input = config['BERT']['InputStoryTree']
    output = config['BERT']['OutputStoryTree']
    scale = config['Visualization']['Scale']
    shortness = config['Visualization']['Short_sents_std']
    closeness = config['Visualization']['Closeness']

    run(input, output, scale, shortness, closeness)

def run(input, output, scale, shortness, closeness):
    embs = torch.load(input)
    print(len(embs))

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
