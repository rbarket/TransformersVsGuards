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
from Preprocessing import device
from collections import Counter
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from Transformerclasses import PositionalEncoding, TransformerBinaryClassifier, train_model, IntegralDataset, collate_fn   
import Preprocessing

# Parameters
df=Github_preprocessing.df
max_size=Github_preprocessing.max_size
tokenizer=Github_preprocessing.tokenizer
train_data=Github_preprocessing.train_data
test_data=Github_preprocessing.test_data

vocab_size = tokenizer.num_words
max_size = max([len(data[1]) for data in train_data + test_data]) + 1 # add 1 for CLS token

MODEL_DIM = 128
HEADS = 4
LAYERS = 2
BATCH_SIZE = 128
LR = 0.0001
EPOCHS = 10

algo_dict = {
'default': 0,
'derivativedivides': 1,
'parts': 2,
'risch': 3,
'norman': 4,
'trager': 5,
'parallelrisch': 6,
'meijerg': 7,
'elliptic': 8, 
'pseudoelliptic':9,
# lookup is 10 but no need to train an ML model for lookup table
'gosper': 11,
'orering':12
}

for algo in algo_dict.keys():
    
    print(f'STARTING TRAINING FOR {algo}')
    
    model = TransformerBinaryClassifier(vocab_size=vocab_size, model_dim=MODEL_DIM, num_heads=HEADS, num_layers=LAYERS)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=3, verbose=True) # to update LR
    
    # Create DataLoaders
    df['algo_label'] = df['label'].apply(lambda x: x[algo_dict[algo]]).values    
    train_df, val_df = train_test_split(df, test_size=0.1, shuffle=True, stratify=df['algo_label'])
    
    train_df = train_df.sort_values(by='prefix', key=lambda x: x.apply(len))
    val_df = val_df.sort_values(by='prefix', key=lambda x: x.apply(len))
    
    train_dataset = IntegralDataset(train_df, tokenizer, algo)
    val_dataset = IntegralDataset(val_df, tokenizer, algo)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=2,  # Parallel data loading
                              pin_memory=True,  # Faster data transfer to GPU
                              prefetch_factor=2, # Prefetch batches for faster training)
                              drop_last=True
                              ) 
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=BATCH_SIZE, 
                            collate_fn=collate_fn,
                            num_workers=2,  # Parallel data loading
                            pin_memory=True,  # Faster data transfer to GPU
                            prefetch_factor=2, # Prefetch batches for faster training)
                            drop_last=True
                            )     
    
    class_model = train_model(model, algo, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=EPOCHS, device=device)
    
    # Save model
    # th.save(class_model.state_dict(), '/home/path_to_folder/' + algo + '.pth')
print('done')
