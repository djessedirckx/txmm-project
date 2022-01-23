# Extractive summarisation of scientific publications using Wordnet

Djesse Dirckx (s1046968)

## About
This repository contains the source code that was used for the research project of the master course "Text and Multimedia Mining" at Radboud University, Nijmegen. In this research project, it was evaluated whether the general semantic lexicon Wordnet could be used to improve the extractive summarisation quality of scientific publications.

## Data
The data set used for this research was originally published in [1]. In order to download the data set, the script provided by the authors can be used. This script can be found [here](https://github.com/EdCo95/scientific-paper-summarisation/tree/master/DataDownloader). Note that an Elsevier API key is required. This key, and more information about the API, can be found on the [Elsevier developer website](https://dev.elsevier.com/).

## Requirements
All source code in this repository was developed using Python (version 3.10.0). Below, requirements that are required for running the scripts are listed. Other versions might work as well, but versions below were used during development.
```python
notebook==6.4.6
numpy==1.21.5
pandas==1.3.5
rouge-score==0.0.4
tqdm==4.62.3
```

## Files
- [tfisf_runner.py](tfisf_runner.py): this file can be used to summarise papers using a TF-ISF approach. This file can be used, using the following parameters:
```
tfisf_runner.py [-h] [--paper_path PAPER_PATH] [--summary_path SUMMARY_PATH] [--sum_length SUM_LENGTH]

Summarise papers

options:
  -h, --help            show this help message and exit
  --paper_path PAPER_PATH
                        Directory where parsed papers are stored
  --summary_path SUMMARY_PATH
                        Directory where generated summaries are stored
  --sum_length SUM_LENGTH
                        Percentage of paper to determine summary length
```
- [tfisf_wordnet_runner.py](tfisf_wordnet_runner.py): this file can be used to summarise papers using a TF-ISF approach that also considers Wordnet synsets. This file can be used using the following parameters:
```
tfisf_wordnet_runner.py [-h] [--paper_path PAPER_PATH] [--summary_path SUMMARY_PATH] [--sum_length SUM_LENGTH]

Summarise papers

options:
  -h, --help            show this help message and exit
  --paper_path PAPER_PATH
                        Directory where parsed papers are stored
  --summary_path SUMMARY_PATH
                        Directory where generated summaries are stored
  --sum_length SUM_LENGTH
                        Percentage of paper to determine summary length
```
-[lesk_runner.py](lesk_runner.py): this file can be used to summarise papers using the Lesk algorithm incorporating Wordnet synset definitions. This file can be used using the following parameters:
```
lesk_runner.py [-h] [--paper_path PAPER_PATH] [--summary_path SUMMARY_PATH] [--sum_length SUM_LENGTH]

Summarise papers

options:
  -h, --help            show this help message and exit
  --paper_path PAPER_PATH
                        Directory where parsed papers are stored
  --summary_path SUMMARY_PATH
                        Directory where generated summaries are stored
  --sum_length SUM_LENGTH
                        Number of sentences for a paper
```
- [Rouge evaluation.ipynb](Rouge%20evaluation.ipynb): This notebook can be used to score generated summaries using the Rouge-1, Rouge-2 and Rouge-3 metrics for summarisation.

[1] Ed Collins, Isabelle Augenstein, and Sebastian Riedel. 2017. A Super-vised Approach to Extractive Summarisation of Scientific Papers. InProceedings of the 21st Conference on Computational Natural LanguageLearning (CoNLL 2017). Association for Computational Linguistics, Van-couver, Canada, 195–205.  https://doi.org/10.18653/v1/K17-1021