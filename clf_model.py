import logging
from os import mkdir
from os.path import join, exists
import time
import matplotlib.pyplot as plt
from torch import no_grad, save, argmax
from torch.cuda import is_available
from torch.optim import AdamW
from sklearn.metrics import classification_report
from transformers import BertForNextSentencePrediction

from clf_data_loader import get_data
        
if __name__ == '__main__':
    
    overfit = False
    batch_size = 8 if not overfit else 2
    max_length = 256 if not overfit else 32
    epochs = 16
    lr = 2e-5
    wd = 0.01
    
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    device = 'cuda' if is_available() else 'cpu'
    train_loader, test_loader = get_data(batch_size, max_length, overfit)
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
        
        model_str = f'epoch_{epoch}_lr_{lr}_wd_{wd}_bs_{batch_size}_ml_{max_length}'
        
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
                
                train_preds.extend(argmax(outputs.logits, dim=1).tolist())
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
                
        if not exists('reports'):
            mkdir('reports')
        with open(join('reports', f'report_{model_str}.txt'), 'w+') as f:
            f.write('Train\n')
            f.write(classification_report(train_labels, train_preds))
            f.write('\nDev\n')
            f.write(classification_report(dev_labels, dev_preds))
            
        epoch_end = time.time()
        
        print(f'Epoch {epoch + 1} took {time.strftime("%H:%M:%S", time.gmtime(epoch_end - epoch_start))}.')
        
        if not exists('plots'):
            mkdir('plots')
        
        plt.figure()
        plt.scatter(range(len(running_losses_train)), running_losses_train, s=2, c='blue', label='train')
        plt.legend()
        plt.title(f'Train Loss: Epoch {epoch + 1}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(join('plots', f'train_loss_{model_str}.png'))
        
        plt.figure()
        plt.scatter(range(len(running_losses_dev)), running_losses_dev, s=2, c='red', label='dev')
        plt.legend()
        plt.title(f'Dev Loss: Epoch {epoch + 1}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(join('plots', f'dev_loss_{model_str}.png'))
        
        if not exists('models'):
            mkdir('models')
        save(
            {'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
            join('models', f'model_{model_str}.pt')
        )
        
    end = time.time()
    
    print(f'Training took {time.strftime("%H:%M:%S", time.gmtime(end - start))}.')