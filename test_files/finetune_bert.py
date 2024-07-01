#import datasets
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from torch import nn

from allennlp.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor


#dataset = datasets.load_dataset('discovery', 'discovery')

BERT_MODEL = 'bert-base-uncased'

'''
class DiscoveryDataset(Dataset):
    def __init__(self, df, tokenizer, labeled=True):
        self.labeled = labeled
        if labeled:
            tmp = df[["A-coref", "B-coref"]].copy()
            tmp["target"] = 0
            tmp.loc[tmp['B-coref']  == 1, "target"] = 1
            tmp.loc[~(tmp['A-coref'] | tmp['B-coref']), "target"] = 2
            self.y = tmp.target.values.astype("uint8")
        
        self.offsets, self.tokens = [], []
        for _, row in df.iterrows():
            tokens, offsets = tokenize(row, tokenizer)
            self.offsets.append(offsets)
            self.tokens.append(tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + tokens + ["[SEP]"]))
        
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            return self.tokens[idx], self.offsets[idx], self.y[idx]
        return self.tokens[idx], self.offsets[idx], None
'''

class Head(nn.Module):
    """The MLP submodule"""
    def __init__(self, bert_hidden_size):
        super().__init__()
        self.bert_hidden_size = bert_hidden_size
        self.span_extractor = SelfAttentiveSpanExtractor(bert_hidden_size)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bert_hidden_size * 3),
            nn.Dropout(0.1),             
            nn.Linear(bert_hidden_size * 3, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),          
            nn.Linear(64, 3)
        )
        for i, module in enumerate(self.fc):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print("Initing batchnorm")
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                    print("Initing linear with weight normalization")
                    assert module[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                    print("Initing linear")
                nn.init.constant_(module.bias, 0)
                
    def forward(self, bert_outputs, offsets):
        assert bert_outputs.size(2) == self.bert_hidden_size
        spans_contexts = self.span_extractor(
            bert_outputs, 
            offsets[:, :4].reshape(-1, 2, 2)
        ).reshape(-1, 2 * self.bert_hidden_size)
        return self.fc(torch.cat([
            spans_contexts,
            torch.gather(
                bert_outputs, 1,
                offsets[:, 4:].unsqueeze(2).expand(-1, -1, self.bert_hidden_size)
            ).squeeze(1)
        ], dim=1))

class BSA(nn.Module):
    def __init__(self):
        super().__init__()
        if BERT_MODEL in ("bert-base-uncased", "bert-base-cased"):
            self.bert_hidden_size = 768
        elif BERT_MODEL in ("bert-large-uncased", "bert-large-cased"):
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.head = Head(self.bert_hidden_size)

    def forward(self, token_tensor, offsets):
        #token_tensor = token_tensor.to(self.device)
        bert_outputs, _ =  self.bert(
            token_tensor, attention_mask=(token_tensor > 0).long(), 
            token_type_ids=None, output_all_encoded_layers=False)
        head_outputs = self.head(bert_outputs, offsets)
        return head_outputs


def tokenize(row, tokenizer):
    break_points = sorted(
        [
            ("A", row["A-offset"], row["A"]),
            ("B", row["B-offset"], row["B"]),
            ("P", row["Pronoun-offset"], row["Pronoun"]),
        ], key=lambda x: x[0]
    )
    tokens, spans, current_pos = [], {}, 0
    for name, offset, text in break_points:
        tokens.extend(tokenizer.tokenize(row["Text"][current_pos:offset]))
        # Make sure we do not get it wrong
        assert row["Text"][offset:offset+len(text)] == text
        # Tokenize the target
        tmp_tokens = tokenizer.tokenize(row["Text"][offset:offset+len(text)])
        spans[name] = [len(tokens), len(tokens) + len(tmp_tokens) - 1] # inclusive
        tokens.extend(tmp_tokens)
        current_pos = offset + len(text)
    tokens.extend(tokenizer.tokenize(row["Text"][current_pos:offset]))
    assert spans["P"][0] == spans["P"][1]
    return tokens, (spans["A"] + spans["B"] + [spans["P"][0]])

tokenizer = BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=False,
    never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
)

df = pd.read_csv("../gap-development.tsv", delimiter="\t")

labeled = True
tmp = df[["A-coref", "B-coref"]].copy()
tmp["target"] = 0
tmp.loc[tmp['B-coref']  == 1, "target"] = 1
print(tmp)
tmp.loc[~(tmp['A-coref'] | tmp['B-coref']), "target"] = 2
print(tmp)
print(tmp[tmp['target'] == 2])
y = tmp.target.values.astype("uint8")

offsets, tokens = [], []
for _, row in df.iterrows():
    tokens, offsets = tokenize(row, tokenizer)
    print(tokens)
    offsets.append(offsets)
    tokens.append(tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + tokens + ["[SEP]"]))
    exit()


# Data loader preparation
train_ds = GAPDataset(df_train, tokenizer)
val_ds = GAPDataset(df_val, tokenizer)
test_ds = GAPDataset(df_test, tokenizer)
train_loader = DataLoader(
    train_ds,
    collate_fn = collate_examples,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)
val_loader = DataLoader(
    val_ds,
    collate_fn = collate_examples,
    batch_size=128,
    num_workers=2,
    pin_memory=True,
    shuffle=False
)
test_loader = DataLoader(
    test_ds,
    collate_fn = collate_examples,
    batch_size=128,
    num_workers=2,
    pin_memory=True,
    shuffle=False
)

# Training loop
model = GAPModel(BERT_MODEL, torch.device("cuda:0"))
# You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
set_trainable(model.bert, False)
set_trainable(model.head, True)
