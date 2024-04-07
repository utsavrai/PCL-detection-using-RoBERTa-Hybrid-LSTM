# Detecting Patronizing and Condescending Language (PCL) in NLP

This repository contains a comprehensive project that focuses on detecting Patronizing and Condescending Language (PCL) in text, a nuanced form of communication that implies superiority over the recipient without the use of openly unpleasant words. The project explores the use of advanced NLP models and techniques to better understand and identify PCL, aiming to contribute to the development of AI systems capable of respectful and empathetic communication.

## Project Overview

The project is structured around an in-depth analysis and modeling effort to detect PCL in natural language. Key components of the project include:

- **Data Analysis and Visualization:** Examination of class imbalance in the dataset and insights into the distribution of text lengths and their association with PCL.
- **Modeling Efforts:**
  - **Baseline Model (RoBERTa):** Utilization of a cased RoBERTa model to retain letter cases crucial for understanding context and sentiment.
  - **Experimental Model (Hybrid LSTM):** A novel approach combining CNNs and LSTM units to capture nuanced language features.
  - **Improvements and Data Processing:** Exploration of data augmentation techniques and processing strategies to enhance model performance.

## Key Findings

- **Subtlety of PCL:** PCL detection poses significant challenges due to its subtle nature and the requirement for contextual understanding.
- **Model Performance:** Both the RoBERTa and Hybrid LSTM models show promising results, particularly when unprocessed text is used. This indicates that preprocessing techniques that simplify text might remove essential information for accurate PCL detection.
- **Data Augmentation Impact:** Strategic data augmentation, including back translation and synonym replacement, significantly improves model accuracy by introducing linguistic diversity.

## Usage

The repository provides scripts and notebooks detailing the data analysis, model training, and evaluation processes. Users interested in replicating the study or exploring the detection of PCL further can follow the outlined steps in the notebooks.

## Code

The code is present in `code/`.

All the experiments and evaluation run for RoBERTa base model are documented in `RoBERTa.ipynb`.

## Data Analysis

For Data Preprocessing and Analysis, please see `code/data_preprocessing_and_analysis.ipynb`.

`images/` contains the images generated from data analysis.

## Data Augmentation

For data augmentation, which includes Back Translation and Synonym Augmentation, please see `code/data_augmentation.ipynb`.

## Models

All the models are either available in this repository or on [HuggingFace](https://huggingface.co/ImperialIndians23).


## Contribution

Contributions to this project are welcome. Whether it's improving the models, experimenting with new data processing techniques, or exploring alternative approaches to detecting PCL, your input can help advance this area of research.

## Evaluation and Model Comparison

The project includes a thorough comparison of model performances, offering insights into the effectiveness of different architectures and data handling strategies in identifying PCL. Tables and figures in the report detail these comparisons and highlight the impact of various factors on model accuracy.

## Conclusion

This work underscores the complexity of detecting PCL in text and demonstrates the potential of deep learning models to tackle this challenge. Future research directions could include exploring layer-wise learning rate decay, weighted random sampling to address class imbalance, and ensemble methods to leverage the strengths of different models.

