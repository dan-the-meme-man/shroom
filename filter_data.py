import os
import torch
import json
from torch.utils.data import DataLoader
from transformers import BertForNextSentencePrediction, BertTokenizer

from clf_data_loader import get_train_data

class DataFilter():
        
    def __init__(self, model_choice, device, threshold=0.6, max_length=512):
        
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
            os.path.join('lr_2e-05_wd_0.0001_bs_16_ml_512', 'epoch_7.pt'), # best acc, micro recall
            os.path.join('lr_2e-07_wd_0.0001_bs_16_ml_512', 'epoch_0.pt'), # best micro prec
            os.path.join('lr_2e-06_wd_0.0001_bs_16_ml_512', 'epoch_13.pt') # best micro f1
        )
        
        print(f'Loading model {models[model_choice]}...')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-cased')
        model.load_state_dict(torch.load(os.path.join('models', models[model_choice]))['model'])
        model.eval()
        self.model = model
        self.model.to(device)
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.threshold = threshold
        self.max_length = max_length
        print(f'Model loaded on device {self.device}.')
    
    def process_batch(self, encoding: dict):
        
        """
        Process a batch of data.
        
        Args:
            data (pd.DataFrame): Data to process.
        """
            
        probs = None
        
        with torch.no_grad():
            outputs = self.model(**encoding.to(self.device))
            
            # gives probability for p(incorrect) and p(correct) for each item in a batch
            probs = torch.softmax(outputs.logits, dim=1)
        
        # if the model is at least threshold confident about correct, return True, else False
        return [p[1].item() >= self.threshold for p in probs]
    
    def filter_in_batches(self, dataloader: DataLoader):
        
        """
        Filter a DataFrame in batches.
        
        Args:
            dataloader (DataLoader): DataLoader to use.
        """
        
        keep = []
        
        for i, batch in enumerate(dataloader):
            keep.extend(self.process_batch(batch))
            #print(keep)
            if i % 100 == 0:
                print(f'Batch {i} processed.')
        
        return keep

def main():
    
    batch_size = 16
    max_length = 512
    
    results = {
        0: {'agnostic': None, 'aware': None},
        1: {'agnostic': None, 'aware': None},
        2: {'agnostic': None, 'aware': None}
    }
    
    print('Loading data...')
    agnostic_loader, aware_loader = get_train_data(batch_size, max_length, overfit=False, remove_unnecessary_cols=True)
    print('Data loaded.')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(3):
        filter = DataFilter(i, device, threshold=0.6, max_length=max_length)
        results[i]['agnostic'] = filter.filter_in_batches(agnostic_loader)
        results[i]['aware'] = filter.filter_in_batches(aware_loader)
        
    with open('filter_results.json', 'w+') as f:
        json.dump(results, f)
        
if __name__ == '__main__':
    main()