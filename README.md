# Non-Invasive Infarct Prediction Using Multi-Fidelity Machine Learning Model
This repository contains the codes and datasets used in our study on non-invasive myocardial infarction (MI) diagnosis through composite in-silico-human data learning. Our work combines simulations and limited human cardiac magnetic resonance (CMR) strain data to train machine learning (ML) models for predicting infarct regions in the left ventricle. Below are the highlights of the repository content:
## Strain Transformation and Preprocessing
* **Strain_transformation/ES_strains.ipynb:** Python code for extracting cardiac strains at the end-systole timepoint from simulation datasets provided in the Strain_transformation/reference_files folder.
* **Strain_transformation/strains_AHA.m:** MATLAB script for transforming strains into American Heart Association (AHA) format for machine learning inputs.
* **Reference Files:** Includes .dat, .sta, fiber orientations, elements, nodes, and short-axis slice files for strain computation. [link](https://drive.google.com/drive/folders/1SV6Owg5v7aOsDtsXI9JdyM3tQ-Do-l4J?usp=drive_link)

## Estimation of Cardiac Strains from CMR
* **Super-Resolution Reconstruction in Cardiac Magnetic Resonance (SRR in CMR):** This in-house built tool is used for estimating the four-dimensional motion of the heart using super-resolution reconstruction (SRR) of CMR images. This code is publicly available on [GitHub](https://github.com/Tanmay24Mukh/SRR_in_CMR.git).

## Datasets
* In-silico simulated cardiac strain datasets generated from finite element models. [link](https://drive.google.com/drive/folders/1SV6Owg5v7aOsDtsXI9JdyM3tQ-Do-l4J?usp=drive_link)
* Human CMR-derived strain data and corresponding infarct labels, organized for model training and evaluation.

## Machine Learning Models (`Multifidelity_codes`)
*	Single-fidelity UNet models trained exclusively on simulation data.
*	Multi-fidelity codes integrating in-silico (low-fidelity) and human (high-fidelity) strain data for robust infarct region prediction.


## Performance Evaluation (`Evaluation_metrics`)
*	Codes for evaluating the ML model performance in predicting infarct regions using Dice Similarity Coefficient (DSC), Intersection over Union (IoU), and other metrics..
*	Single-fidelity and multi-fidelity trained model weights for random initialization and leave one out cross-validation approaches. [link](https://drive.google.com/drive/folders/1SV6Owg5v7aOsDtsXI9JdyM3tQ-Do-l4J?usp=drive_link)

## Note: 
Due to the large size of the datasets, they are hosted on Google Drive and can be accessed using this [link](https://drive.google.com/drive/folders/1SV6Owg5v7aOsDtsXI9JdyM3tQ-Do-l4J?usp=drive_link).
