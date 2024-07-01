from allennlp.modules.span_extractors.self_attentive_span_extractor import SelfAttentiveSpanExtractor
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

class Head(nn.Module):
    def __init__(self, bert_hidden_size: int):
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

def get_edu_spans(filename):
    spans = []
    start = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            spans.append((start, start+len(line)-2))
            start = start + len(line)-1
    return spans

bucket_widths = False
input_dim = 3072
num_width_embeddings = None
span_width_embedding_dim = None

sase = SelfAttentiveSpanExtractor(
    input_dim = input_dim,
    num_width_embeddings=num_width_embeddings,
    span_width_embedding_dim=span_width_embedding_dim,
    bucket_widths=bucket_widths
)
for param in sase.parameters():
    if param.grad is not None:
        print(param.grad.data)

sequence: torch.Tensor = torch.load('output/british_word_BERT.pt')

bottom_list = []
for b in sequence:
    bottom_list.append(torch.stack(b, dim=0))
sequence = torch.stack(bottom_list, dim=0)
sequence = sequence.unsqueeze(0)

spans = torch.LongTensor(get_edu_spans('input/british_base_text_edus.txt')).unsqueeze(0)

output = sase.forward(sequence, spans).squeeze(0)
print(output.size())
plt.plot(output[1].detach())
plt.show()
exit()

#print(spans.size())
#exit()

#np.save('input/british_EDU.npy', output.detach().numpy())
torch.save(output, 'input/british_EDU.pt')