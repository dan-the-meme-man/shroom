import matplotlib.pyplot as plt
from torch.cuda import is_available
from torch.optim import AdamW
from transformers import BertForNextSentencePrediction
from clf_data_loader import get_data

# class BertRegressor(Module):
#     def __init__(self, bert):
#         super(BertRegressor, self).__init__()
#         self.bert = bert
#         for param in self.bert.parameters():
#             param.requires_grad = False
#         self.fc = Linear(self.bert.config.hidden_size, 1)
        
#     def forward(self, input_ids, token_type_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#         last_hidden_state = outputs.last_hidden_state
#         cls_token = last_hidden_state[:, 0]
#         return self.fc(cls_token)
        
if __name__ == '__main__':
    
    batch_size = 8
    max_length = 128
    epochs = 3
    
    device = 'cuda' if is_available() else 'cpu'
    train_loader, test_loader = get_data(batch_size, max_length)
    model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-cased')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    running_losses = []
    
    for epoch in range(epochs):
        for i, (encoding, target) in enumerate(train_loader):
            
            optimizer.zero_grad()
            outputs = model(**encoding.to(device), labels=target.to(device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_losses.append(loss.item())
            msg = f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_losses[-1]:.4},'
            msg += f' Avg Loss: {sum(running_losses) / len(running_losses):.4}'
            print(msg)
            
plt.scatter(range(len(running_losses)), running_losses, s=1, c='blue')
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')