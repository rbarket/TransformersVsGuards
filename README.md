## Transformers vs Gaurds
This is the main code respoitory for Maple and Python Implementation of "Transformers to Predict the Applicability of Symbolic Integration Routines"
## Requirements and Installation
The code can be implemented after installing the dependencies. The dependecies can be installed by creating a [conda](https://www.anaconda.com/products/individual) environment from the included YAML file with

  conda env create -f TreeLSTM_DGL.yml
  
  conda activate TreeLSTM_DGL


## Dataset
The training and testing dataset is available on training dataset provided at [zenodo](https://zenodo.org/records/14013787). The data is in JSON format and can be read with the `parse` function in Maple or the `sympify` function in the Python library SymPy or just simply kept in string form. Each data point is in the format `[integrand, integrand_prefix, integral, DAG sizes]`. The integrand_prefix entry is the prefix notation of the integrand (useful for constructing the tree form of the data as well as use in the LSTM) and DAG sizes is a list of each DAG size from all successful algorithms (-1 means the algorithm failed). An example of one entry from the dataset: `[
    "1/5/(2+x)", 
    [
      "mul",
      "div",
      "INT+",
      "1",
      "INT+",
      "5",
      "pow",
      "add",
      "INT+",
      "2",
      "x",
      "INT-",
      "1"
    ],
    "1/5*ln(1+1/2*x)",
    [
      60,
      -1,
      -1,
      60,
      60,
      -1,
      60,
      63,
      -1,
      -1,
      -1,
      -1,
      74
    ]`.

The DAG sizes correspond to the following order: "default", "derivativedivides", "parts", "risch", "norman", "trager", "parallelrisch", "meijerg", "elliptic", "pseudoelliptic", "lookup", "gosper", "orering. To see these sub-methods that `int` calls in Maple, see the [help page](https://www.maplesoft.com/support/help/maple/view.aspx?path=int%2fmethods). The DAG sizes were acquired by taking the integrand, running the integrand through Maple's `int` command with each available method, and then recording the DAG size of the output. This is then converted to a binary label where it is 1 if the DAG size is positive and 0 otherwise. 

  
## Training
  After setting up the environment, the file Github_training.py can be directly implemented which uses a training dataset provided at [zenodo](https://zenodo.org/records/14013787) to train a model which can be saved and later used to carry out the testing and explainability part of the code. Files are named accordingly, ones with names ending with ...._train.json are training files whereas ...._TEST.json are testing files.
  
  Pre-trained models, which were trained and used to obtain the results reported in the [paper](https://openreview.net/forum?id=b2Ni828As7) have also been provided at [zenodo](https://zenodo.org/records/14013787).

## Testing  
  The testing can be implemented using Github_testing.py, which utilizes the trained model saved in the respective directory or the pretrained model which will utilize the testing dataset provided at [zenodo](https://zenodo.org/records/14013787). In the test file, do not forget to change the path to where you have downloaded the models. 

## Explainability
  The file Github_calculateattributions.py can be run to calculate the attribution scores. The file utilizes the testing dataset and implements Layer Integrated Gradients to calculate the attribution scores corresponding to each input token. The explainability part of the paper has been divided into two different parts. 

#### 1. Calculating Collective attribution scores:
  The first part of the file Github_calculateattributions.py calculates the averaged attribution scores corresponding to each input token as shown in figure 4 of the [paper](https://openreview.net/forum?id=b2Ni828As7).

#### 2. Calculating attribution score for individual example:
  The second part of the file Github_calculateattributions.py is responsible for calculating the attribution scores corresponding to just one individual example as shown in figure 3 of the [paper](https://openreview.net/forum?id=b2Ni828As7).

## Reference
[Transformers to Predict the Applicability of Symbolic Integration Routines](https://openreview.net/forum?id=b2Ni828As7)
```
@inproceedings{
barket2024transformers,
title={Transformers to Predict the Applicability of Symbolic Integration Routines},
author={Rashid Barket and Uzma Shafiq and Matthew England and Juergen Gerhard},
booktitle={The 4th Workshop on Mathematical Reasoning and AI at NeurIPS'24},
year={2024},
url={https://openreview.net/forum?id=b2Ni828As7}
}
```

