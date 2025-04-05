# An Efficient Low-Shot Class-Agnostic Counting Framework with Hybrid Encoder and Iterative Exemplar Feature learning

The official PyTorch implementation of the paper 'An Efficient Low-Shot Class-Agnostic Counting Framework with Hybrid Encoder and Iterative Exemplar Feature learning'.

### Download the dataset

•	The FSC147 dataset can be downloaded as instructed in FamNet: https://github.com/cvlab-stonybrook/LearningToCountEverything.
•	The CARPK dataset is available at: https://lafi.github.io/LPN/.
•	The ShanghaiTech Dataset is available at: https://github.com/desenzhou/ShanghaiTechDataset.

Download the FSC147 dataset as instructed in its [official repository](https://github.com/cvlab-stonybrook/LearningToCountEverything). Make sure to also download the `annotation_FSC147_384.json` and `Train_Test_Val_FSC_147.json` and place them alongside the image directory (`images_384_VarV2`) in the directory of your choice.


### Installation

The installation packages that required for the environment are located in the file enviroment.yml.

### Training & Testing

The scripts for training and testing are train.by and evaluate.py, respectively.