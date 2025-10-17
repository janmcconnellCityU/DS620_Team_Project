# DS620_Team_Project
Applying Deep Learning Techniques for Audio Pattern Recognitions with AudioMNIST

## Team Members

- Svetlana Grabar – grabarsvetlana@cityuniversity.edu
- Jared Graham – grahamjared@cityuniversity.edu
- Zsolt Kiss – kisszsolt@cityuniversity.edu
- Jan McConnell – janmcconnell@cityuniversity.edu

## Abstract

This project explores how deep learning can be used to recognize spoken digits from the AudioMNIST dataset, which contains thousands of short recordings of numbers spoken by different people. The goal is to find out how well two types of neural networks, convolutional and recurrent, can learn and classify these audio patterns. Spectrograms and visual examples will be used to explain how the networks interpret the sound data. The project will also discuss what factors affect performance, including preprocessing choices, network structure, and overfitting. By the end, the study aims to demonstrate the strengths and limitations of deep learning models for simple speech recognition tasks.

## Keywords

Deep learning, speech recognition, audio classification, convolutional neural networks, recurrent neural networks, AudioMNIST, spectrograms

## Repository Overview

This repository supports the DS620 Machine Learning and Deep Learning Team Project at City University of Seattle. It includes all scripts, notebooks, and documentation used for model training, evaluation, and analysis.

### Downloading the Dataset
To download the dataset directly from Kaggle:
1. Install the Kaggle API:  
   `pip install kaggle`
2. Place your Kaggle credentials file (`kaggle.json`) in `~/.kaggle/`
3. Run the script:  
   `python src/data/download_kaggle.py`
