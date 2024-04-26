import os
import torch
import json
from torch.utils.data import DataLoader
from transformers import BertForNextSentencePrediction, BertTokenizer

from clf_data_loader import get_train_data

class DataFilter():
        
    def __init__(self, model_choice, threshold=0.6, max_length=512):
        
        """
        Initialize the DataFilter.
        
        Args:
            model_choice (int): Choice of model to use.
            threshold (float): Threshold for filtering.
            max_length (int): Maximum length of input sequences.
        """
        
        if model_choice not in (0, 1, 2):
    
            msg = 'Invalid choice. Please choose 0, 1, or 2.\n\n'
            msg += '0: best micro precision\n'
            msg += '1: best micro f1\n'
            msg += '2: best accuracy and micro recall\n'
            
            raise ValueError(msg)
        
        models = (
            os.path.join('lr_2e-05_wd_0.0001_bs_16_ml_512', 'epoch_0.pt'), # best micro prec
            os.path.join('lr_2e-05_wd_0.0001_bs_16_ml_512', 'epoch_12.pt'), # best micro f1
            os.path.join('lr_2e-05_wd_1e-06_bs_16_ml_512', 'epoch_7.pt') # best acc, micro recall
        )
        
        model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-cased')
        model.load_state_dict(torch.load(os.path.join('models', models[model_choice]))['model'])
        model.eval()
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.threshold = threshold
        self.max_length = max_length
    
    def process_batch(self, data):
        
        """
        Process a batch of data.
        
        Args:
            data (pd.DataFrame): Data to process.
        """
        
        src = data['src']
        hyp = data['hyp']
        
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
            
        probs = None
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            
            # gives probability for p(incorrect) and p(correct) for each item in a batch
            probs = torch.softmax(outputs.logits, dim=1)
        
        # if the model is confident one way or the other, return True, otherwise False, as a list of booleans
        return (probs.max(dim=1).values > self.threshold).tolist()
    
    def filter_in_batches(self, dataset: DataLoader):
        
        """
        Filter a DataFrame in batches.
        
        Args:
            dataset (SHROOMDataset): Dataset to filter.
        """
        
        keep = []
        
        for i, batch in enumerate(dataset):
            keep.extend(self.process_batch(batch))
            if i % 100 == 0:
                print(f'Batch {i} processed.')
        
        return keep

def main():
    
    batch_size = 16
    max_length = 512
    
    filters = (
        DataFilter(0, max_length=max_length),
        DataFilter(1, max_length=max_length),
        DataFilter(2, max_length=max_length)
    )
    
    results = {
        0: {'agnostic': None, 'aware': None},
        1: {'agnostic': None, 'aware': None},
        2: {'agnostic': None, 'aware': None}
    }
    
    agnostic_loader, aware_loader = get_train_data(batch_size, max_length, overfit=False, remove_unnecessary_cols=True)

    for i, filter in enumerate(filters):
        results[i]['agnostic'] = filter.filter_in_batches(agnostic_loader)
        results[i]['aware'] = filter.filter_in_batches(aware_loader)
        
    with open('filter_results.json', 'w+') as f:
        json.dump(results, f)
        
if __name__ == '__main__':
    main()