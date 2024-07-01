from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import yaml

tokenizer = AutoTokenizer.from_pretrained("sileod/roberta-base-discourse-marker-prediction")
model = AutoModelForSequenceClassification.from_pretrained("sileod/roberta-base-discourse-marker-prediction", output_hidden_states=True)

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

    input = config['roBERTa']['InputBase'] #config['Visualization']['Input']
    output = config['roBERTa']['OutputWordBERT'] #config['Visualization']['Output']
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
    sentences = text.split('.')[:-1]
    print(f'Num sentences: {len(sentences)}')
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

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
        hidden_states = torch.stack(outputs.hidden_states)
        mean_pooled = hidden_states.sum(axis=2) / attention_masks.sum(axis=-1).unsqueeze(-1)
        mean_pooled = torch.mean(mean_pooled[:2], dim=0)
        print(mean_pooled.shape)
        embs = mean_pooled
        '''
        hidden_states = outputs.hidden_states
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings =  token_embeddings.permute(1, 2, 0, 3)

        print(token_embeddings.size())

        embs = make_word_embeddings_1(token_embeddings)
        '''
        print(embs.shape)
        torch.save(embs.detach().numpy(), output)
    
main()
