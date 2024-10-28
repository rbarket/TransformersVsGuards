## Transformers vs Gaurds
This is the main code respoitory for Maple and Python Implementation of "Transformers to Predict the Applicability of Symbolic Integration Routines"
## Requirements and Installation
The code can be implemented after installing the dependencies. The dependecies can be installed by creating a [conda](https://www.anaconda.com/products/individual) environment from the included YAML file with

  conda env create -f TREELSTM_DGL.yml
  
  conda activate TreeLSTM_DGL

  (add information regarding setting up the kernel)
  
## Training
  After setting up the environment the file Github_training.py can be directly implemented which uses a training dataset provided at [zenodo](https://zenodo.org/records/13992762) to train a model which can be saved and later used to carry out the testing and explainability part of the code. Files are named accordingly, ones with names ending with ...._train.json are training files whereas ...._TEST.json are testing files.
  
  Pre-trained models, which were trained and used to obtain the results reported in the [paper](provide link to paper) have also been provided at [zenodo](link for models).

## Testing  
  The testing can be implemented using Github_testing.py, which utilizes the trained model saved in the respective directory or the pretrained model which will utilize the testing dataset provided at [zenodo](https://zenodo.org/records/13992762) 

## Explainability
  The file Github_calculateattributions.py can be run to calculate the attribution scores. The file utilizes the testing dataset and implements Layer Integrated Gradients to calculate the attribution scores corresponding to each input token. The explainability part of the paper has been divided into two different parts. 

#### 1. Calculating Collective attribution scores:
  The first part of the file Github_calculateattributions.py calculates the averaged attribution scores corresponding to each input token as shown in figure 4 of the [paper](link to paper).

#### 2. Calculating attribution score for individual example:
  The second part of the file Github_calculateattributions.py is responsible for calculating the attribution scores corresponding to just one individual example as shown in figure 3 of the [paper](link to paper).
