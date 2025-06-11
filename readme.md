# Neural decoding from stereotactic EEG: accounting for electrode variability across subjects

In this work, we introduce a training framework and architecture that can be used for multi-subject neural decoding
based on stereotactic electroencephalography (sEEG). We use our framework  to decode the trial-wise response time
of subjects performing a behavioral task solely from their neural data. For more information, please refer to our 
[project page](https://gmentz.github.io/seegnificant).

To protect the privacy of the people that kindly shared their sEEG data with us (and to compy with HIPAA regulations),
we cannot make our dataset public. To help you structure your data in a way that can be processed by our framework,
we provide synthetic (fake) sEEG data for 3 "subjects".  You can find the synthetic data 
[here](https://drive.google.com/drive/folders/1UFSRT3wGNYZAXdpndDyRHr-CmjRQPlbM?usp=sharing).

## Installation

In a clean virtual environment with python 3.8.10, run the following to clone the repository and download the synthetic data:

```
git clone https://github.com/gmentz/seegnificant.git
cd seegnificant
pip install -e .
gdown --folder https://drive.google.com/drive/folders/1UFSRT3wGNYZAXdpndDyRHr-CmjRQPlbM -O data
```

## Getting started

For an easy-to-follow introduction to our framework, please follow along the `example.ipynb`. 

Alternatively, you can run individual components of the framework by running:
- `python3 -m Signal_Processing.harmonize`
- `python3 -m Signal_Processing.FDR_correction`
- `python3 -m Model_and_Training.sEEGDataset`
- `python3 -m Model_and_Training.SingleSubjectTrain`
- `python3 -m Model_and_Training.MultiSubjectTrain`
- `python3 -m Model_and_Training.TransferPreTrained`

## Pretrained weights
The pretrained weights of seegnificant, trained on the combined data from 21 subjects across 5 different data splits, are available in the ```models``` folder. For completeness, each set of saved weights includes the subject-specific ```TaskHeads``` used during pretraining (refer to Section 4.3 of the paper). To load the model weights for any number of subjects, use the following code:
```
model = seegnificant(2, 1, XX)  # XX is the number of subjects in your study
model.load_state_dict(torch.load('./models/pretrained_seegnificant_21_subjets_seed_YY.pt'), strict=False) # YY is the seed from {0, 1, 2, 3, 4} 
```
Caution: Unless you are replicating the results reported in Section 4.3 of our paper, the weights of the ```taskHeads``` layer (the subject-specific MLPs) should be finetuned to your own dataset.

## Citation:

We hope that you will find this code useful. If you do, please consider citing our work as:

```bibtex
@inproceedings{mentzelopoulos2024neural,
 author = {Mentzelopoulos, Georgios and Chatzipantazis, Evangelos and Ramayya, Ashwin and Hedlund, Michelle and Buch, Vivek and Daniilidis, Kostas and Kording, Konrad and Vitale, Flavia},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {108600--108624},
 publisher = {Curran Associates, Inc.},
 title = {Neural decoding from stereotactic EEG: accounting for electrode variability across subjects},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/c473b9c8897f50203fa23570687c6b30-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```

## Acknowledgments
We would like to thank the authors of the following repositories for making their code publicly available.
- [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)
- [POYO](https://github.com/neuro-galaxy/poyo)
