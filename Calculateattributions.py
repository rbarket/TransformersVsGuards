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
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from Transformerclasses import TransformerBinaryClassifier, IntegralDataset, collate_fn
from Preprocessing import replace_int_with_C, DataLoader, tokenizer 
from Testing import model_test, test_loader
from IPython.display import display, HTML
import matplotlib as mpl
import matplotlib.colors as mcolors

# Assuming 'model' is wrapped in DataParallel

if isinstance(model_test, nn.DataParallel):
    # Unwrap the model
    model_test = model_test.module

# Now move the model to the desired device (e.g., single GPU or CPU)
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model_test.to(device)

# Function to invert the word index (from index back to word)
def invert_word_index(word_index):
    return {index: word for word, index in word_index.items()}



def tokens_to_strings(token_ids, index_to_word):
    """
    Convert token IDs to their original string values using an inverted word_index.
    
    Parameters:
    - token_ids (torch.Tensor): Tensor of token IDs.
    - index_to_word (dict): Dictionary mapping token indices to token strings.
    
    Returns:
    - List of token strings, preserving specific tokens like 'int+' and 'int-'.
    """
    tokens = []
    for token_id in token_ids:
        token_id = token_id.item()  # Convert tensor item to integer
        
        if token_id in index_to_word:
            token_str = index_to_word[token_id]
        else:
            token_str = '<OOV>'
        
        tokens.append(token_str)

    return tokens


def plot_aggregated_attributions(mean_token_attributions):
    """
    Plots the mean attributions as a bar plot.
    
    Parameters:
    - mean_token_attributions (dict): Dictionary mapping tokens to their mean attribution values.
    """
    tokens_to_exclude = {'[CLS]', 'INT+', 'INT-'}

    # Filter out excluded tokens
    filtered_attributions = {token: attr for token, attr in mean_token_attributions.items() if token not in tokens_to_exclude}

    tokens = list(filtered_attributions.keys())
    attributions = list(filtered_attributions.values())
    colors = ['navy' if val >= 0 else 'red' for val in attributions]
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(tokens, attributions,color=colors)
  
    # Customize the plot
    plt.xlabel('Tokens', fontsize=14)
    plt.ylabel('Attribution Value', fontsize=14)
    plt.title('Feature Attributions', fontsize=18, weight='bold')
    plt.xlabel('Tokens')
    plt.xticks(rotation=45 ,ha='right', fontsize=12)  # Rotate tokens for better visibility
    plt.tight_layout()
    plt.savefig('attributions_aggregarted_final.png', format='png', bbox_inches='tight', dpi=600)
 


def plot_attribution_scores(attribution_scores):
    tokens = attribution_scores['tokens']
    scores = attribution_scores['scores']
    
    # Define custom colormaps for more vibrant red and navy
    red_cmap = mcolors.LinearSegmentedColormap.from_list("custom_red", ["#ffcccc", "#ff0000"])  # Light to dark red
    navy_cmap = mcolors.LinearSegmentedColormap.from_list("custom_navy", ["#ccccff", "#000080"])  # Light to dark navy

    # Separate positive and negative scores
    pos_scores = [score for score in scores if score >= 0]
    neg_scores = [score for score in scores if score < 0]

    # Normalize positive and negative values separately
    if pos_scores:
        pos_norm = plt.Normalize(0, max(pos_scores))  # Darker navy for higher positive values
    else:
        pos_norm = plt.Normalize(0, 1)  # Fallback normalization

    if neg_scores:
        neg_norm = plt.Normalize(min(neg_scores), 0)  # Reverse normalization for negative values
    else:
        neg_norm = plt.Normalize(-1, 0)  # Fallback normalization

    # Function to map scores to colors
    def get_color(score):
        if score >= 0:
            # Navy for positive values, darker as score increases
            return navy_cmap(pos_norm(score))
        else:
            # Red for negative values, darker as score decreases (more negative)
            return red_cmap(1 - neg_norm(score))  # Invert the normalization for red

    # Apply color mapping to all scores
    colors = [get_color(score) for score in scores]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(tokens, scores,color=colors)
   
    # Add labels and title
    plt.xlabel('Tokens',fontsize=14,weight='bold')
    plt.ylabel('Attribution Scores',fontsize=14,weight='bold')
    plt.title('Token Attribution Scores',fontsize=14, weight='bold')

    plt.xticks(rotation=45 ,ha='right', fontsize=12)  # Rotate x-axis labels for better readability

    #save the plot
    plt.tight_layout()
    plt.savefig('individual_attributions_colnew.png')

def create_red_navy_cmap():
    """
    Create a custom colormap that transitions from red to navy.
    """
    colors = ["red", "white", "navy"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("red_navy", colors)
    return cmap

def colorize(attrs, cmap):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses the provided colormap and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)
    
    # Compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors



def display_colored_text(tokens, scores):
    """
    Displays tokens with background colors based on their scores in a Jupyter Notebook.
    """
    # Create the custom red-navy colormap
    red_navy_cmap = create_red_navy_cmap()
    
    # Generate colors using the colorize function with the new colormap
    colors = colorize(scores, red_navy_cmap)
    
    # Create HTML content with colored tokens
    html_content = '<div style="font-family: Arial, sans-serif; font-size: 15px;">'
    
    for token, color in zip(tokens, colors):
        text_color = "#000000"
        
        html_content += f'<span style="background-color: {color}; color: {text_color}; padding: 5px; margin-right: 5px; border-radius: 3px; display: inline-block;">{token}</span>'
    html_content += '</div>'
    
    # Display the HTML
    with open("Individual_example.html", "w") as file:
        file.write(html_content)
    

def map_attribution_scores(input_seq, unique_tokens, attribution_scores, ignore_token='<OOV>'):
    # Create a dictionary to map unique tokens to their attribution scores
    token_score_map = dict(zip(unique_tokens, attribution_scores))
    
    # Create a list to store the mapped scores
    output_scores = []

    # Iterate over the input sequence
    for token in input_seq:
        # If the token is not the ignore token, map it to its score
        if token != ignore_token:
            # Append the corresponding score from the token_score_map
            if token in token_score_map:
                output_scores.append(token_score_map[token])
            else:
                # Append None or 0 if token is not found in the unique tokens list (optional)
                output_scores.append(None)
    
    return output_scores

word_index = {'OOV': 1,
 'INT+': 2,
 'mul': 3,
 'x': 4,
 'add': 5,
 'INT-': 6,
 'pow': 7,
 'CONST1': 8,
 '1': 9,
 '2': 10,
 'CONST2': 11,
 '[CLS]': 12,
 'div': 13,
 'exp': 14,
 'ln': 15,
 'CONST3': 16,
 'cos': 17,
 'sin': 18,
 'tan': 19,
 'cosh': 20,
 'sinh': 21,
 'tanh': 22,
 'atan': 23,
 'atanh': 24,
 'asinh': 25,
 'acos': 26,
 'asin': 27,
 'acosh': 28,
 'Pi': 29,
 'abs': 30,
 'Complex': 31,
 'I': 32,
 'cot': 33,
 'acot': 34,
 'coth': 35,
 'Re': 36}

# change names from e.g. acos -> arccos
index_to_word = invert_word_index(word_index)
index_to_word[28] = 'arcosh'
index_to_word[23] = 'arctan'
index_to_word[24] = 'arctanh'
index_to_word[27] = 'arcsin'
index_to_word[25] = 'arcsinh'
index_to_word[26] = 'arccos'
index_to_word[34] = 'arccot'
print(index_to_word)

# Calculate Leyered Integrated Gradients#
token_attributions = defaultdict(list)
token_counts = defaultdict(int)

# Initialize LayerIntegratedGradients with the correct layer
layer_ig = LayerIntegratedGradients(model_test, model_test.embedding)

# Variable to keep track of batches
batch_index = 0
all_attr = []
targetnew = []


for batch in test_loader:
    print(f"Processing batch {batch_index}")
    inputs, masks, labels = batch['ids'], batch['masks'], batch['labels']
    inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
    
    # Forward pass, passing mask as positional argument
    outputs = model_test(inputs, masks)  # Ensure mask is passed
    outputs = outputs.to(th.float).squeeze()

    labels = labels.to(th.long)

    # Compute attributions for each sample
    for i in range(inputs.size(0)):
        input_tensor = inputs[i].unsqueeze(0).to(device)  # Ensure correct shape [1, seq_len] and device
        mask_tensor = masks[i].unsqueeze(0).to(device)    # Get the corresponding mask
        target_class = labels[i].item()

        targetnew.append(target_class)

        # Compute attributions using LayerIntegratedGradients
        # For binary classification, no need to pass target index
        attributions = layer_ig.attribute(
            input_tensor, 
            n_steps=50, 
            additional_forward_args=(mask_tensor,)
        )
        
        attributions = attributions.squeeze().cpu().detach().numpy()
        all_attr.append(attributions)
        targetnew.append(target_class)

        # Detokenize the input token IDs to their string representation
        input_token_ids = inputs[i].cpu().numpy()  # Convert to numpy array for easier processing
        input_tokens = tokens_to_strings(input_token_ids, index_to_word)

        # Aggregate attributions for detokenized tokens based on target class
        for token, attribution in zip(input_tokens, attributions):
            if token != "<OOV>":  # Skip tokens that are not in the word index
                token_attributions[token].append(attribution)
                token_counts[token] += 1

    batch_index += 1

mean_token_attributions = {
    token: np.mean(token_attributions[token])  # Compute the mean of the list
    for token in token_attributions.keys()
}

plot_aggregated_attributions(mean_token_attributions)

###############################################
### for finding individual attributions #######
###############################################


# Extract inputs in the test data with abs token, along with their labels, masks and indices
abs_inputs = []  # List to store corresponding inputs where 'abs' appears
abs_labels = []  # List to store corresponding labels where 'abs' appears
abs_masks =[] # List to store corresponding masks where 'abs' appears
abs_indices=[] #List to store corresponding indices where 'abs' appears
batch_index=0
new_test_data=[]
i=0
# Loop through the batches in the test_loader
for batch_index, batch in enumerate(test_loader):
    print(f"Processing batch {batch_index}")
    inputs, masks, labels = batch['ids'], batch['masks'], batch['labels']
    print(inputs.size())
    #inputs, labels = inputs.to(device), labels.to(device)
    
    # Process each input in the batch
    for input_index in range(inputs.size(0)):
        # Get input tensor for the current input
        input_token_ids = inputs[input_index].cpu().numpy()
        input_tokens = tokens_to_strings(input_token_ids, index_to_word)
        
        # Check if 'abs' is in the input tokens
        if 'abs' in input_tokens:
            
            abs_indices.append((batch_index, input_index))  # Store the batch index and input index
            abs_inputs.append(inputs[input_index].cpu()) 
            abs_masks.append(masks[input_index].cpu())    # Store the corresponding input tensor
            abs_labels.append(labels[input_index].cpu())      # Store the corresponding label tensor
            #new_test_data.append(test_data[batch_index])
            i+=1

    print(f"Finished processing batch {batch_index}")
batch_index+=1
# Print results
print("Indices where 'abs' appears:", abs_indices)
print("Corresponding inputs where 'abs' appears:", abs_inputs)
print("Corresponding labels where 'abs' appears:", abs_labels)            

# Initialize dictionaries to store attributions and counts for individual attribution case 
token_attributions_ind = defaultdict(list)
token_counts_ind = defaultdict(int)


b=15 #index that you want to check attributions of (15 is inserted as this was reported in the paper)

input_tensor=abs_inputs[b].unsqueeze(0).to(device)
target_class=int(abs_labels[b])
mask_tensor=abs_masks[b].unsqueeze(0).to(device)

try:
        # Compute attributions
        attributions = layer_ig.attribute(
            input_tensor, 
            n_steps=50, 
            additional_forward_args=(mask_tensor,)
        )
        attributions = attributions.squeeze().cpu().detach().numpy()
        
        # Detokenize the input token IDs to their string representation
        input_token_ids =abs_inputs[b].cpu().numpy()  # Convert to numpy array for easier processing
        input_tokens = tokens_to_strings(input_token_ids, index_to_word)

        # Aggregate attributions for detokenized tokens
        for token, attribution in zip(input_tokens, attributions):
            if token != "<OOV>":  # Skip tokens that are not in the word index
                token_attributions_ind[token].append(attribution)
                token_counts_ind[token] += 1

        # Store data in variables
        input_sequence = input_tokens
        attribution_scores = {
            'tokens': list(token_attributions_ind.keys()),
            'scores': [np.mean(token_attributions_ind[token]) for token in token_attributions_ind],
            'counts': [token_counts_ind[token] for token in token_attributions_ind]
        }


except Exception as e:
        print(f"Error computing attributions for sample {input_index}: {e}")

plot_attribution_scores(attribution_scores)

a=input_sequence
tokens = attribution_scores['tokens']
scores = attribution_scores['scores']

mapped=map_attribution_scores(a,tokens,scores,ignore_token='<OOV>')
print((mapped))
print(a)
cleaned_sequence = [token for token in a if token != '<OOV>']
display_colored_text(cleaned_sequence, mapped)
