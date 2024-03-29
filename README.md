# Demo video on Azure ML notebook
![Alt text](https://img.youtube.com/vi/7EgA9zOhAP4/0.jpg)](https://youtu.be/7EgA9zOhAP4)
# RNN-LSTMS, stack-augmented RNN and Reinforcement learning drug design for COVID-19 using Microsoft Azure 
* Started the project from https://github.com/mattroconnor/deep_learning_coronavirus_cure , the approach is great and everyone should check this repo.
* Combining Generative Recurrent Networks for De Novo Drug Design(https://onlinelibrary.wiley.com/doi/full/10.1002/minf.201700111) & Deep reinforcement learning for de novo drug design(https://advances.sciencemag.org/content/4/7/eaap7885)papers,
* GitHub rep links: https://github.com/mattroconnor/deep_learning_coronavirus_cure , https://github.com/isayev/ReLeaSE
* using deep learning techniques for small molecule generation and PyRx to evaluate binding affinities.
* Binding scores of leading existing drugs are around -10 to -11(more negative score the better) and around -13 for the drug Remdesivir which recently entered clinical testing. 
* Most of the design generate molecules without considering their property. Higher binding affinity will not ensure a practical drug. We need a policy based algorithm to obtain practical micro molecules.
* We changed the hyperaprameters, model architecture and combined these two novel methods in to a single approach to solve the most complex drug designing problem.
* We uses RNN-LSTM to generate novel micro molecules and saves the model. It has taken 10 hours to train in Microsoft azure ML notebook after we changed the hyperparam.
* the model gets fine tuned using transfer learning and genetic algorithm by injecting Remdesivir and HIV inhibitors.
* The final generation is saved and used to generate 10000 new molecules and save it as "generation_0.smi".
* "generation_0.smi" will then passed to a Stack-augmented RNN and generate 10000 more molecules
* The network architecture has to be modified inorder to work better on the given cleaned dataset. Hyperparameter optimization done prior to the training
* These molecules then fine tuned using logP optimization Reinforcement learning and generate molecules with optimized logP value. As the generator as an agent and the predictor as a critic.
* The file is saved as final_data.smi and convert to a .sdf file for PyRx analysis.
## We were able to create several small molecule candidates which achieved binding scores up to -18.
* Afer getting the ligand we will test the binding affinty with covid-19 protein using PyRx.
* We are studying the genomics sequence which may helps as a policy function to generate better molecules
* We worked on the practical side effects of the drugs, which will helps to fine tune our model. 
## Training the model on Microsoft Azure ML notebook
![alt text](https://github.com/kishorkuttan/Covid-19-drug-discovery-using-RNN-LSTMS-stack-augmented-RNN-and-RL-powered-by-Microsoft-Azure/blob/master/Training_on_azure_overview.png?raw=true)

## Generated new molecules after training
![alt text](https://github.com/kishorkuttan/Covid-19-drug-discovery-using-RNN-LSTMS-stack-augmented-RNN-and-RL-powered-by-Microsoft-Azure/blob/master/generating_molecule.png?raw=true)



# Requirements:

In order to get started you will need:
* Modern NVIDIA GPU, [compute capability 3.5](https://developer.nvidia.com/cuda-gpus) of newer.
* [CUDA 9.0](https://developer.nvidia.com/cuda-downloads)
* [Pytorch 0.4.1](https://pytorch.org)
* [Tensorflow 1.8.0](https://www.tensorflow.org/install/) with GPU support
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [Scikit-learn](http://scikit-learn.org/)
* [Numpy](http://www.numpy.org/)
* [tqdm](https://github.com/tqdm/tqdm)
* [Mordred](https://github.com/mordred-descriptor/mordred)

# Installation with Anaconda
* conda create - n deeplearning pip python=3.7
* conda activate deeplearning
* conda install -c rdkit rdkit
* pip install mordred
* pip install tensorflow
* pip install tqdm
* pip install xgboost
* pip install jupyter notebook

# Training using Microsoft Azure
* Use the ml notebook instance powered by GPU
* clone this repo
* and use demo.ipynb notebook to run and test

# Training using local system

* open terminal/cmd 
* type jupyter notebook
* open the demo.ipynb



