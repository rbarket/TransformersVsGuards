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
from collections import defaultdict
from collections import Counter
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from captum.attr import LayerIntegratedGradients
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Custom function to replace integers with "CONST"
def replace_int_with_C(L):
    # L is a list of strings (prefix notation for expression)    
    keep_list = list(range(-2,3)) # dont replace these integers with CONST
    new_L = L.copy()
        
    for i in range(len(L)):
        if L[i].isdigit():
            
            if int(L[i]) not in keep_list:
                if len(L[i])==1: # 1 digit integers
                    new_L[i] = 'CONST1'
                elif len(L[i])==2: # 2 digit integers
                    new_L[i] = 'CONST2'
                else: # all other cases
                    new_L[i] = 'CONST3'
    return new_L

def change_to_binary(L):
    return [1 if x > 0 else 0 for x in L]

# If running on a GPU, Check if GPU is availabale 
if th.cuda.is_available():
    print("Available GPUs:")
    for i in range(th.cuda.device_count()):
        print(f"GPU {i}: {th.cuda.get_device_name(i)}")
else:
    print("No GPUs available.")


# Set the random seed for reproducibility
seed = 1998
th.manual_seed(seed)

# Create a th.Generator with the specified seed for random sampler
generator = th.Generator()
generator.manual_seed(seed)

# If using a GPU, set the seed for GPU as well
if th.cuda.is_available():
    th.cuda.manual_seed(seed)


TRIG = ['sin',
'cos',
'tan',
'cot',
'sec',
'scs',
'cosh',
'sinh',
'tanh',
'acos',
'asin',
'atan',
'acot',
'asec',
'ascs',
'acosh',
'asinh',
'atanh']

BIN_OP = ['add', 'sub', 'mul', 'div', 'pow']


with open('Dataset/train/BWD_train.json') as f:
    bwd_train = json.load(f)
with open('Dataset/test/BWD_test.json') as f:
    bwd_test = json.load(f)
    
with open('Dataset/train/FWD_train.json') as f:
    fwd_train = json.load(f)
with open('Dataset/test/FWD_test.json') as f:
    fwd_test = json.load(f)

with open('Dataset/train/IBP_train.json') as f:
    ibp_train = json.load(f)
with open('Dataset/test/IBP_test.json') as f:
    ibp_test = json.load(f)
    
with open('Dataset/train/SUB_train.json') as f:
    sub_train = json.load(f)
with open('Dataset/test/SUB_test.json') as f:
    sub_test = json.load(f)
    
with open('Dataset/train/RISCH_train.json') as f:
    risch_train = json.load(f)
with open('Dataset/test/RISCH_test.json') as f:
    risch_test = json.load(f)

bwd_train = [sublist[:3] + sublist[4:] for sublist in bwd_train]
fwd_train = [sublist[:3] + sublist[4:] for sublist in fwd_train]
ibp_train = [sublist[:3] + sublist[4:] for sublist in ibp_train]

bwd_test = [sublist[:3] + sublist[4:] for sublist in bwd_test]
fwd_test = [sublist[:3] + sublist[4:] for sublist in fwd_test]
ibp_test = [sublist[:3] + sublist[4:] for sublist in ibp_test]

train_data = bwd_train + fwd_train + ibp_train + sub_train +  risch_train
test_data = bwd_test + fwd_test + ibp_test +  sub_test + risch_test

df = pd.DataFrame(train_data, columns=['integrand', 'prefix', 'integral', 'label'])
df['prefix'] = df['prefix'].apply(replace_int_with_C)


df['prefix'] = df['prefix'].apply(replace_int_with_C)
df["prefix"] = df["prefix"].transform(lambda k: tuple(k))  # transforming to tuple is much faster operation
df.drop_duplicates(subset='prefix', inplace=True)
df['prefix'] = df['prefix'].apply(list)


df['label'] = df['label'].apply(change_to_binary)
df['prefix'] = df['prefix'].apply(lambda x: ['[CLS]'] + x)


# Tokenize Expressions
vocab_size = set([word for sublist in df['prefix'] for word in sublist])
max_size = max([len(data[1]) for data in train_data + test_data])+1 # longest expression (one added because of the CLS token)
 
tokenizer = Tokenizer(num_words = len(vocab_size), oov_token='OOV', lower=False)
tokenizer.fit_on_texts(df['prefix'])
tokenizer.word_index
 
print("done")
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
test_data = bwd_test + fwd_test + ibp_test +  sub_test
vocab_size = tokenizer.num_words
max_size = max([len(data[1]) for data in train_data + test_data]) + 1