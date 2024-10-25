import pandas as pd
import json
import os
import swifter
import argparse
import seaborn as sns
import math

os.environ["DGLBACKEND"] = "pyth"
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from Github_preprocessing import max_size
from collections import Counter
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from Github_preprocessing import device
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence


#Define the class for dataset 
class IntegralDataset(Dataset):
    def __init__(self, dataframe, tokenizer, sub_algo):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe['prefix'].values
        
        algo_dict = {'default': 0, 
                    'derivativedivides': 1, 
                    'parts': 2,
                    'risch': 3,
                    'norman': 4,
                    'trager': 5,
                    'parallelrisch': 6,
                    'meijerg': 7, 
                    'elliptic': 8,
                    'pseudoelliptic':9, 
                    'lookup': 10,
                    'gosper': 11,
                    'orering':12}        
        self.labels = dataframe['label'].apply(lambda x: x[algo_dict[sub_algo]]).values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize and pad sequences
        tokens = self.tokenizer.texts_to_sequences([text])
        # tokens_pad = pad_sequences(tokens, maxlen=self.max_len, padding='pre', truncating='post')[0]

        return {
            'ids': th.tensor(tokens, dtype=th.long),
            'labels': th.tensor(self.labels[idx], dtype=th.float)
        }

        


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a long enough positional encoding matrix
        pe = th.zeros(max_len, model_dim)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as a buffer so it's not considered a model parameter, but will be moved to GPU if necessary
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + self.pe[:x.size(0), :]
        return x
    

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers, num_classes=1, dropout=0.4):
        super(TransformerBinaryClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        
        self.pos_encoding = PositionalEncoding(model_dim, max_len=max_size)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, dim_feedforward=2048)
        
        self.fc1 = nn.Linear(model_dim, 13)        
        
        # Transformer Encoder with multiple layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final linear layer for classification
        self.fc = nn.Linear(model_dim, num_classes)

        # Dropout after TransformerEncoder
        self.dropout = nn.Dropout(dropout)
        
        # Normalise Probabilities
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x, mask):
        
        # x should be of shape (seq_length, batch_size)
        # Apply embedding layer to convert indices to embeddings
        x = self.embedding(x) # (batch_size, seq_len, h_dim)
        
        x = x.transpose(0, 1) # (seq_len, batch_size, h_dim)
    
        x = self.pos_encoding(x)
        
        # Transformer expects input of shape (seq_length, batch_size, h_dim)
        x = self.transformer_encoder(x, src_key_padding_mask=mask) # mask ignores the padding when calculating attention        
        
        if th.isnan(x).any() or th.isinf(x).any():
            print("NaN or Inf detected in transformer_output!")
            print("Transformer Output:", x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply the final linear layer (we take the last token of the sequence)
        
        cls_output = x[0] # only get embedding from CLS token
        if th.isnan(cls_output).any() or th.isinf(cls_output).any():
            print("NaN or Inf detected in cls_token!")
        logits = self.fc(cls_output)
        # print("logits", logits)        
        
        return logits

    

def train_model(model, algo, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    
    multi_batch_flag = False
    if th.cuda.device_count() > 1:
        print(f"Using {th.cuda.device_count()} GPUs")
        multi_batch_flag = True
        model = nn.DataParallel(model)
    
    model = model.to(device)        
    # Initialize the GradScaler for mixed precision
    scaler = GradScaler()
        
    # Early stopping variables
    early_stopping_patience = 10  # Stop if no improvement x epochs
    best_val_loss = 10000  # Track the best validation loss (arbitrarily high loss)
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        
        if early_stop:
            print("Early stopping")
            break
        
        model.train()
        running_loss = 0.0
        accuracy = BinaryAccuracy().to(device)
        precision = BinaryPrecision().to(device)

        for batch in train_loader:
            inputs, masks, labels = batch['ids'], batch['masks'], batch['labels']
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            # print(inputs.shape, labels.shape, masks.shape)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with autocast():
                outputs = model(inputs, masks)
                outputs = outputs.to(th.float).squeeze()
            
                if multi_batch_flag:              
                    outputs = outputs.view(-1)
                    
                    # Compute loss
                    # print(outputs)
                    # print(labels)
                loss = criterion(outputs, labels.float())
        
            # Backward pass and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()              
            
            running_loss += loss.item()

            # Apply sigmoid to outputs before computing metrics
            preds = th.sigmoid(outputs)
            
            # Update metrics
            accuracy.update(preds.squeeze(), labels.int())
            precision.update(preds.squeeze(), labels.int())                        

        
        # Compute metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = accuracy.compute()
        epoch_precision = precision.compute()

        # Print training metrics
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.8f}')
        
        # Start validation metrics
        
        model.eval()
        val_loss = 0.0
        
        with th.no_grad():
            accuracy = BinaryAccuracy().to(device)
            precision = BinaryPrecision().to(device)
            
            for batch in val_loader:
                inputs, masks, labels = batch['ids'], batch['masks'], batch['labels']
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                
                outputs = model(inputs, masks)
                
                outputs = outputs.to(th.float).squeeze()
            
                if multi_batch_flag:              
                    outputs = outputs.view(-1)
                
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                
                # Apply sigmoid to outputs before computing metrics
                preds = th.sigmoid(outputs)
            
                # Update metrics
                accuracy.update(preds.squeeze(), labels.int())
                precision.update(preds.squeeze(), labels.int())
                
        # Step the scheduler with the epoch loss        
        val_loss /= len(val_loader)
        scheduler.step(val_loss) # type: ignore
        
        val_accuracy = accuracy.compute()
        val_precision = precision.compute()
        
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Precision: {val_precision:.8f} \n')
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update the best validation loss
            epochs_no_improve = 0  # Reset patience counter
            #th.save(model.state_dict(), '/home/path_to_folder/' + algo + '.pth')
            
            print(f"Model saved at epoch {epoch} with loss {best_val_loss:.4f}\n")
            
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                early_stop = True  # Trigger early stopping        
         
    return model


def collate_fn(batch):
    # Do dynamic padding for each batch
    
    # Extract 'inputs' and 'labels' from the batch
    ids = [item['ids'].squeeze(0) for item in batch]  # Squeeze to remove the extra dimension
    labels = th.tensor([item['labels'] for item in batch])

    # Pre-pad sequences to the length of the longest sequence in the batch
    padded_ids = pad_sequence(ids, batch_first=True, padding_value=0)
    # Create attention mask (1 where there's real token, 0 where there's padding)
    attention_masks = (padded_ids == 0).bool()

    return {'ids': padded_ids, 'labels': labels, 'masks': attention_masks}
    
