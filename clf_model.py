import time
import logging
from os import mkdir
from os.path import join, exists
from itertools import product

import matplotlib.pyplot as plt
from torch import no_grad, save, argmax
from torch.cuda import is_available
from torch.optim import AdamW
from sklearn.metrics import classification_report
from transformers import BertForNextSentencePrediction

from clf_data_loader import get_dev_data, get_train_data
        
def main(batch_size, lr, wd, overfit=False):

    epochs     = 20         if not overfit else 20
    max_length = 512        if not overfit else 32
    batch_size = batch_size if not overfit else 16
    lr         = lr         if not overfit else 2e-5
    wd         = wd         if not overfit else 1e-4
    
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    device = 'cuda' if is_available() else 'cpu'
    train_loader, test_loader = get_dev_data(batch_size, max_length, overfit)
    model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-cased')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    print(f'Training with {device}. Using {len(train_loader)} batches of size {batch_size}.')
    print(f'Overfitting: {overfit}. Max length: {max_length}. Epochs: {epochs}.')
    
    running_losses_train = []
    times_train = []
    running_losses_dev = []
    times_dev = []
    
    start = time.time()
    
    for epoch in range(epochs):
        
        epoch_start = time.time()
        
        model.train()
        
        train_preds = []
        train_labels = []
        dev_preds = []
        dev_labels = []
        
        model_str = f'lr_{lr}_wd_{wd}_bs_{batch_size}_ml_{max_length}'
        
        for i, (encoding, target) in enumerate(train_loader):
            
            batch_start = time.time()
            
            optimizer.zero_grad()
            outputs = model(**encoding.to(device), labels=target.to(device))

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_preds.extend(argmax(outputs.logits, dim=1).tolist())
            train_labels.extend(target.flatten().tolist())
            
            batch_end = time.time()

            running_losses_train.append(loss.item())
            times_train.append(batch_end - batch_start)
            
            msg =  f'(train) Epoch: {epoch + 1}, Batch: {i + 1}/{len(train_loader)}, '
            msg += f'Loss: {running_losses_train[-1]:.4}, '
            msg += f'Avg Loss: {sum(running_losses_train) / len(running_losses_train):.4}, '
            msg += f'Average time per batch: '
            msg += f'{time.strftime("%H:%M:%S", time.gmtime(sum(times_train) / len(times_train)))}'
            print(msg)
        
        model.eval()
        
        with no_grad():
            for i, (encoding, target) in enumerate(test_loader):
                
                batch_start = time.time()
                
                outputs = model(**encoding.to(device), labels=target.to(device))
                loss = outputs.loss
                
                dev_preds.extend(argmax(outputs.logits, dim=1).tolist())
                dev_labels.extend(target.flatten().tolist())
                
                batch_end = time.time()

                running_losses_dev.append(loss.item())
                times_dev.append(batch_end - batch_start)
                
                msg =  f'(dev) Epoch: {epoch + 1}, Batch: {i + 1}/{len(test_loader)}, '
                msg += f'Loss: {running_losses_dev[-1]:.4}, '
                msg += f'Avg Loss: {sum(running_losses_dev) / len(running_losses_dev):.4}, '
                msg += f'Average time per batch: '
                msg += f'{time.strftime("%H:%M:%S", time.gmtime(sum(times_dev) / len(times_dev)))}'
                print(msg)
        
        train_clf_report = classification_report(train_labels, train_preds, output_dict=True)
        dev_clf_report = classification_report(dev_labels, dev_preds, output_dict=True)
        
        if not exists('reports'):
            mkdir('reports')
        if not exists(join('reports', model_str)):
            mkdir(join('reports', model_str))
        with open(join('reports', model_str, f'epoch_{epoch}.txt'), 'w+') as f:
            f.write('Train\n')
            f.write(classification_report(train_labels, train_preds))
            f.write('\nDev\n')
            f.write(classification_report(dev_labels, dev_preds))
            
        epoch_end = time.time()
        
        print(f'Epoch {epoch + 1} took {time.strftime("%H:%M:%S", time.gmtime(epoch_end - epoch_start))}.')
        
        if not exists('plots'):
            mkdir('plots')
        if not exists(join('plots', model_str)):
            mkdir(join('plots', model_str))
        plt.figure()
        plt.scatter(range(len(running_losses_train)), running_losses_train, s=2, c='blue', label='train')
        plt.legend()
        plt.title(f'Train Loss: Epoch {epoch + 1}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(join('plots', model_str, f'train_loss_epoch_{epoch}.png'))
        plt.close()
        
        plt.figure()
        plt.scatter(range(len(running_losses_dev)), running_losses_dev, s=2, c='red', label='dev')
        plt.legend()
        plt.title(f'Dev Loss: Epoch {epoch + 1}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(join('plots', model_str, f'dev_loss_epoch_{epoch}.png'))
        plt.close()
        
        if not exists('models'):
            mkdir('models')
        if not exists(join('models', model_str)):
            mkdir(join('models', model_str))
        save(
            {'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
            join('models', model_str, f'epoch_{epoch}.pt')
        )
        
    end = time.time()
    
    print(f'Training took {time.strftime("%H:%M:%S", time.gmtime(end - start))}.')
    
    return train_clf_report, dev_clf_report, epoch
    
if __name__ == '__main__':
    
    overfit = False
    
    hparam_grid = {
        'batch_size': [16], # 16
        'lr': [2e-5, 2e-6, 2e-7], # 2e-5
        'wd': [1e-4, 1e-5, 1e-6] # 1e-4
    }
    
    best_hparams_by_metric = dict.fromkeys(
        ('weighted_f1', 'weighted_precision', 'weighted_recall', 'accuracy')
    )
    for key in best_hparams_by_metric:
        best_hparams_by_metric[key] = dict.fromkeys(('batch_size', 'lr', 'wd', 'epoch'))
        
    best_metric_values = dict.fromkeys(
        ('weighted_f1', 'weighted_precision', 'weighted_recall', 'accuracy')
    )
    for key in best_metric_values:
        best_metric_values[key] = 0.0
    
    for batch_size, lr, wd in product(*hparam_grid.values()):
        
        train_clf_report, dev_clf_report, epoch = main(batch_size, lr, wd, overfit)
        
        weighted_f1 = dev_clf_report['weighted avg']['f1-score']
        weighted_precision = dev_clf_report['weighted avg']['precision']
        weighted_recall = dev_clf_report['weighted avg']['recall']
        accuracy = dev_clf_report['accuracy']
        
        if weighted_f1 > best_metric_values['weighted_f1']:
            best_metric_values['weighted_f1'] = weighted_f1
            best_hparams_by_metric['weighted_f1']['batch_size'] = batch_size
            best_hparams_by_metric['weighted_f1']['lr'] = lr
            best_hparams_by_metric['weighted_f1']['wd'] = wd
            best_hparams_by_metric['weighted_f1']['epoch'] = epoch
        
        if weighted_precision > best_metric_values['weighted_precision']:
            best_metric_values['weighted_precision'] = weighted_precision
            best_hparams_by_metric['weighted_precision']['batch_size'] = batch_size
            best_hparams_by_metric['weighted_precision']['lr'] = lr
            best_hparams_by_metric['weighted_precision']['wd'] = wd
            best_hparams_by_metric['weighted_precision']['epoch'] = epoch
            
        if weighted_recall > best_metric_values['weighted_recall']:
            best_metric_values['weighted_recall'] = weighted_recall
            best_hparams_by_metric['weighted_recall']['batch_size'] = batch_size
            best_hparams_by_metric['weighted_recall']['lr'] = lr
            best_hparams_by_metric['weighted_recall']['wd'] = wd
            best_hparams_by_metric['weighted_recall']['epoch'] = epoch
            
        if accuracy > best_metric_values['accuracy']:
            best_metric_values['accuracy'] = accuracy
            best_hparams_by_metric['accuracy']['batch_size'] = batch_size
            best_hparams_by_metric['accuracy']['lr'] = lr
            best_hparams_by_metric['accuracy']['wd'] = wd
            best_hparams_by_metric['accuracy']['epoch'] = epoch
        
        print('Best hyperparameters by metric:')
        for k in best_hparams_by_metric:
            print(f'{k}: {best_hparams_by_metric[k]}, {best_metric_values[k]}')
            
        if overfit:
            break