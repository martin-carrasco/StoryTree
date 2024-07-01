import numpy as np
import util
#import w2v_embedding as w2v_emb
import torch
from hierarchy_node import HierarchyNode
import summarization
import yaml
from tree_encoding import TreeEncoding


def make_word_embeddings_1(token_embeddings):

    sentence_vecs = []
    # `token_embeddings` is a [30 x 64 x 13 x 768] tensor.
    # For each embeddings...
    for sent in token_embeddings:
        token_vecs_cat = []
        # For each token in the sentence...
        for token in sent:
            
            # `token` is a [13 x 768] tensor

            # Concatenate the vectors (that is, append them together) from the last 
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            
            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)
        token_tensor = torch.stack(token_vecs_cat, dim=0)
        sentence_vecs.append(token_tensor)
    sentence_tensor = torch.stack(sentence_vecs, dim=0)
    return sentence_tensor
def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    input = config['Visualization']['Input']
    output = config['Visualization']['Output']
    scale = config['Visualization']['Scale']
    shortness = config['Visualization']['Short_sents_std']
    closeness = config['Visualization']['Closeness']

    run(input, output, scale, shortness, closeness)


def run(input, output, scale, shortness, closeness):
    
    with open(input, 'r') as file:
        text = file.read()
        
    par_sent_dict, sents, prep_sents = util.get_text_data(text, module='nltk')
    original_sents = sents
    embs = w2v_emb.get_doc_embedding(prep_sents)
    prep_sents, sents, merged = util.cleanup_sentences(embs, prep_sents, sents, threshold=closeness, n_std=shortness)
    embs = w2v_emb.get_doc_embedding(prep_sents)

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