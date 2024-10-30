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

from collections import Counter
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix
import Github_preprocessing
from Github_transformerclasses import TransformerBinaryClassifier
from Github_transformerclasses import IntegralDataset
from Github_transformerclasses import collate_fn
from Github_preprocessing import replace_int_with_C
from Github_preprocessing import DataLoader
from Github_preprocessing import change_to_binary

vocab_size=Github_preprocessing.vocab_size
tokenizer=Github_preprocessing.tokenizer
max_size=Github_preprocessing.max_size
test_data=Github_preprocessing.test_data

MODEL_DIM = 128
HEADS = 4
LAYERS = 2



def test_model(algo, test_data):

        # load data 
    df_test = pd.DataFrame(test_data, columns=['integrand', 'prefix', 'integral', 'label'])
    
        # pre-process data
    df_test['prefix'] = df_test['prefix'].apply(replace_int_with_C)
    df_test["prefix"] = df_test["prefix"].transform(lambda k: tuple(k))  # transforming to tuple is much faster operation
    df_test.drop_duplicates(subset='prefix', inplace=True)
    df_test['prefix'] = df_test['prefix'].apply(list)
    df_test['prefix'] = df_test['prefix'].apply(lambda x: ['[CLS]'] + x)
        
    df_test['label'] = df_test['label'].apply(change_to_binary)
        
        # put in dataloader
    data_test = IntegralDataset(df_test, tokenizer, algo)
    test_loader = DataLoader(data_test, 
                                batch_size=256,
                                # shuffle=True,
                                collate_fn=collate_fn,
                                # num_workers=4,  # Parallel data loading
                                # pin_memory=True,  # Faster data transfer to GPU
                                # prefetch_factor=2 # Prefetch batches for faster training)
                                drop_last=True
                                )
        
        
        # If using DataParallel, wrap the model again
    model_test = TransformerBinaryClassifier(vocab_size=vocab_size, model_dim=MODEL_DIM, num_heads=HEADS, num_layers=LAYERS)
    model_test = th.nn.DataParallel(model_test)

        # Load model from memory
    model_path = '/path to your models folder /Models/' + algo + '.pth'
    model_test.load_state_dict(th.load(model_path))
        
        
        # Ensure the model is in evaluation mode
    model_test.eval()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model_test.to(device)
        
    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)

    with th.no_grad():
            # 6. Iterate over the DataLoader
        for batch in test_loader:
                
                inputs, masks, labels = batch['ids'], batch['masks'], batch['labels']
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                
                outputs = model_test(inputs, masks)
                outputs = outputs.to(th.float).squeeze()
                    
                #on gpu, need to unsqueeze              
                outputs = outputs.view(-1)
                    
                # Apply sigmoid to outputs before computing metrics
                preds = th.sigmoid(outputs)
                    
                    # Update metrics
                accuracy.update(preds.squeeze(), labels.int())
                precision.update(preds.squeeze(), labels.int())
                recall.update(preds.squeeze(), labels.int())            

    final_accuracy = accuracy.compute()
    final_precision = precision.compute()
    final_recall = recall.compute()

        # Print metrics
    print(f'Sub-Algo: {algo}, Accuracy: {final_accuracy:.4f}, Precision: {final_precision:.4f}, Recall: {final_recall:.4f}')
    return model_test, test_loader
algo='risch' # change the name to the model that needs you want to evaluate.
model_test, test_loader = test_model(algo, test_data)

  
