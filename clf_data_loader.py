import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

class SHROOMDataset(Dataset):
    def __init__(self, df, tokenizer, is_dev, max_length):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_dev = is_dev

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data.iloc[idx]['src']
        hyp = self.data.iloc[idx]['hyp']
        if self.is_dev:
            label = self.data.iloc[idx]['label']
        
        if src is None:
            src = ''
        
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
        
        if self.is_dev:
            return encoding, tensor([int(label)])
        else:
            return encoding

def get_train_data(batch_size=8, max_length=128, overfit=False, remove_unnecessary_cols=False):
    df_agnostic = pd.read_json(
        os.path.join('SHROOM_unlabeled-training-data-v2', 'train.model-agnostic.json')
    )
    df_aware = pd.read_json(
        os.path.join('SHROOM_unlabeled-training-data-v2', 'train.model-aware.v2.json')
    )
    
    if overfit:
        df_agnostic = df_agnostic.iloc[:10]
        df_aware = df_aware.iloc[:10]
    
    if remove_unnecessary_cols: 
        df_agnostic.drop(columns=['ref', 'task', 'model', 'tgt'], inplace=True)
        df_aware.drop(columns=['ref', 'task', 'model', 'tgt'], inplace=True)
        
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    is_dev = False
        
    agnostic_loader = DataLoader(
        SHROOMDataset(df_agnostic, tokenizer, is_dev, max_length),
        batch_size=batch_size,
        shuffle=False
    )
    
    aware_loader = DataLoader(
        SHROOMDataset(df_aware, tokenizer, is_dev, max_length),
        batch_size=batch_size,
        shuffle=False
    )
    
    return agnostic_loader, aware_loader

def get_dev_data(batch_size=8, max_length=128, overfit=False):
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
    
    df['label'] = (df['p(Hallucination)'] >= 0.5) & (df['p(Hallucination)_solar'] >= 0.5)
    df['label'] = df['label']

    train, test = train_test_split(df, test_size=0.05, random_state=42, stratify=df['label'])
    
    if overfit:
        train = train.iloc[:10]
        test = train.iloc[:10]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    is_dev = True
    
    train_loader = DataLoader(
        SHROOMDataset(train, tokenizer, is_dev, max_length),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        SHROOMDataset(test, tokenizer, is_dev, max_length),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader
