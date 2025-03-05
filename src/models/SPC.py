import polars as pl
import torch
import numpy as np

from tqdm import tqdm
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainDataset(Dataset):

    def __init__(self, df, tokenizer):
        df = df.filter(pl.col("polarity") != "conflict")
        self.data = df
        self.label_mapping = {"negative": 0,
                              "neutral": 1,
                              "positive": 2}
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data[index]

        encoding = self.tokenizer(row['text'].item(), row['target'].item())
        label = self.label_mapping[row["polarity"].item()]
        encoding['labels'] = label
        return encoding

class BertForAspectClassification(nn.Module):
    def __init__(self, bert, num_labels=2, freeze_bert=False):
        super().__init__()
        self.bert = bert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, device=device),
                                        nn.ReLU(),
                                        nn.Linear(self.bert.config.hidden_size, num_labels, device=device))
        # self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels, device=device)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.pooler_output  # [batch_size, hidden_size] Выход из FNN для next sentence prediction
        # sequence_output = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size] Среднее всех токенов в предложении
        logits = self.classifier(sequence_output)  # [batch_size, num_labels]
        return logits


def collate_fn(batch_obj, pad_token=0):
    max_token_len = 0
    for sample in batch_obj:
        max_token_len = max(max_token_len, len(sample['input_ids']))

    input_ids = []
    token_type_ids = []
    attention_mask = []
    labels = []
    for sample in batch_obj:
        n_padding =  max(max_token_len - len(sample['input_ids']), 0)
        input_ids.append(sample['input_ids'] + [pad_token] * n_padding)
        token_type_ids.append(sample['token_type_ids'] + [0] * n_padding)
        attention_mask.append(sample['attention_mask'] + [0] * n_padding)
        labels.append(sample['labels'])

    result = {}
    result['input_ids'] = torch.tensor(input_ids).to(device, dtype = torch.long)
    result['token_type_ids'] = torch.tensor(token_type_ids).to(device, dtype = torch.long)
    result['attention_mask'] = torch.tensor(attention_mask).to(device, dtype = torch.long)
    result['labels'] = torch.tensor(labels).to(device, dtype = torch.long)
    return result

def test_model(model, dl: DataLoader):
    y_pred = np.array([])
    y_true = np.array([])
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl):
            out = model(**batch)
            y_pred = np.hstack((y_pred, out.argmax(dim=1).reshape(-1).detach().cpu().numpy()))
            y_true = np.hstack((y_true, batch['labels'].reshape(-1).detach().cpu().numpy()))
    print('\n')
    print(classification_report(y_true, y_pred))