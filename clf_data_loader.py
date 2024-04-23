import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

class SHROOMDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data.iloc[idx]['src']
        hyp = self.data.iloc[idx]['hyp']
        correct_pred = float(self.data.iloc[idx]['p(Hallucination)'])
        solar_pred = float(self.data.iloc[idx]['p(Hallucination)_solar'])
        if correct_pred >= 0.5 and solar_pred >= 0.5:
            target = tensor([1])
        elif correct_pred >= 0.5 and solar_pred < 0.5:
            target = tensor([0])
        elif correct_pred < 0.5 and solar_pred >= 0.5:
            target = tensor([0])
        elif correct_pred < 0.5 and solar_pred < 0.5:
            target = tensor([1])
        else:
            raise ValueError('Invalid target value')
        
        encoding = self.tokenizer(
            src,
            hyp,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        for k in encoding:
            encoding[k] = encoding[k].squeeze()
        
        return encoding, target

def get_data(batch_size=8, max_length=128):
    df_agnostic = pd.read_json(os.path.join('SHROOM_dev-v2', 'val.model-agnostic.json'))
    df_aware = pd.read_json(os.path.join('SHROOM_dev-v2', 'val.model-aware.v2.json'))
    
    orig_df = pd.concat([df_agnostic, df_aware], axis=0)
    orig_df.drop(columns=['model', 'labels', 'label'], inplace=True)
    
    solar_preds_agnostic = pd.read_json(os.path.join('solar_out', 'val.model-agnostic.json'))
    solar_preds_aware = pd.read_json(os.path.join('solar_out', 'val.model-aware.json'))
    
    solar_df = pd.concat([solar_preds_agnostic, solar_preds_aware], axis=0)
    solar_df.rename(columns={'p(Hallucination)': 'p(Hallucination)_solar'}, inplace=True)
    
    df = pd.concat([orig_df, solar_df], axis=1)
    bad_rows = df.loc[(df['p(Hallucination)_solar'] > 0.5) & (df['label'] == 'Not Hallucination')]
    df.drop(bad_rows.index, inplace=True)
    df.drop(columns=['label'], inplace=True)

    train, test = train_test_split(df, test_size=0.05, random_state=42)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    train_loader = DataLoader(
        SHROOMDataset(train, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        SHROOMDataset(test, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader