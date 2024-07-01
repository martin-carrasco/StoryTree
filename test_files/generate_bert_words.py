import yaml
from transformers import BertTokenizer, BertModel

import torch


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

    input = config['BERT']['InputBase'] #config['Visualization']['Input']
    output = config['BERT']['OutputWordBERT'] #config['Visualization']['Output']
    output_ipe = output.split('.')[0] + '.ipe'
    scale = config['Visualization']['Scale']
    shortness = config['Visualization']['Short_sents_std']
    closeness = config['Visualization']['Closeness']

    run(input, output, output_ipe, scale, shortness, closeness)



def run(input, output, output_ipe, scale, shortness, closeness):
    
    with open(input, 'r') as file:
        text = file.read()
        
    input_ids = []
    attention_masks = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sentences = text.split('.')
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
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

    model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings =  token_embeddings.permute(1, 2, 0, 3)

        embs = make_word_embeddings_1(token_embeddings)
        torch.save(embs, output)


if __name__ == "__main__":
    main()