import yaml
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

import torch

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    input = config['BERT']['InputBase'] #config['Visualization']['Input']
    output = config['BERT']['OutputSentBERT'] #config['Visualization']['Output']
    output_ipe = output.split('.')[0] + '.ipe'
    scale = config['Visualization']['Scale']
    shortness = config['Visualization']['Short_sents_std']
    closeness = config['Visualization']['Closeness']

    run(input, output, output_ipe, scale, shortness, closeness)



def run(input, output, output_ipe, scale, shortness, closeness):
    
    with open(input, 'r') as file:
        text = file.read()
        
    model = SentenceTransformer('all-mpnet-base-v2')
    sentences = text.split('.')[:-1]
    embs = model.encode(sentences)
    print(embs.shape)
    #torch.save(embs, output)


if __name__ == "__main__":
    main()