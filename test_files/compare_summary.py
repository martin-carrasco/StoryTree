from evaluate import load
import evaluate
import numpy as np
from hierarchy_node import HierarchyNode
import summarization
import yaml
from tree_encoding import TreeEncoding

import torch


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    input_embs = config['Summary']['Embeddings']
    input_text = config['Summary']['InputText']
    input_comp = config['Summary']['CompText']
    scale = config['Visualization']['Scale']
    shortness = config['Visualization']['Short_sents_std']
    closeness = config['Visualization']['Closeness']

    run(input_text, input_embs, input_comp, scale, shortness, closeness)



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






if __name__ == "__main__":
    main()