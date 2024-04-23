import time
import matplotlib.pyplot as plt
from torch.cuda import is_available
from torch.optim import AdamW
from transformers import BertForNextSentencePrediction
from clf_data_loader import get_data
        
if __name__ == '__main__':
    
    batch_size = 8
    max_length = 256
    epochs = 3
    
    device = 'cuda' if is_available() else 'cpu'
    train_loader, test_loader = get_data(batch_size, max_length)
    model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-cased')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    running_losses = []
    times = []
    
    start = time.time()
    
    for epoch in range(epochs):
        
        epoch_start = time.time()
        
        for i, (encoding, target) in enumerate(train_loader):
            
            batch_start = time.time()
            
            optimizer.zero_grad()
            outputs = model(**encoding.to(device), labels=target.to(device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            batch_end = time.time()

            running_losses.append(loss.item())
            times.append(batch_end - batch_start)
            
            msg =  f'Epoch: {epoch + 1}, Batch: {i + 1}/{len(train_loader)}, '
            msg += f'Loss: {running_losses[-1]:.4}, '
            msg += f'Avg Loss: {sum(running_losses) / len(running_losses):.4}, '
            msg += f'Average time per batch: '
            msg += f'{time.strftime("%H:%M:%S", time.gmtime(sum(times) / len(times)))}'
            print(msg)
            
        epoch_end = time.time()
        
        print(f'Epoch {epoch + 1} took {time.strftime("%H:%M:%S", time.gmtime(epoch_end - epoch_start))}.')
            
    end = time.time()
    
    print(f'Training took {time.strftime("%H:%M:%S", time.gmtime(end - start))}.')
            
plt.scatter(range(len(running_losses)), running_losses, s=1, c='blue')
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')