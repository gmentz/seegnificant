# Neural decoding from stereotactic EEG: accounting for electrode variability across subjects

In this work, we introduce a training framework and architecture that can be used for multi-subject neural decoding
based on stereotactic electroencephalography (sEEG). We use our framework  to decode the trial-wise response time
of subjects performing a behavioral task solely from their neural data. For more information, please refer to our 
[project page](https://gmentz.github.io/seegnificant).

To protect the privacy of the people that kindly shared their sEEG data with us (and to compy with HIPAA regulations),
we cannot make our dataset public. To help you structure your data in a way that can be processed by our framework,
we provide synthetic (fake) sEEG data for 3 "subjects".  You can find the synthetic data 
[here](https://drive.google.com/drive/folders/1UFSRT3wGNYZAXdpndDyRHr-CmjRQPlbM?usp=sharing).

## Installation:

In a clean virtual environment with python 3.8.10, run the following to clone the repository and download the synthetic data:

```
git clone https://github.com/gmentz/seegnificant.git
cd seegnificant
pip install -e .
gdown --folder https://drive.google.com/drive/folders/1UFSRT3wGNYZAXdpndDyRHr-CmjRQPlbM -O data
```

## Getting started:

For an easy-to-follow introduction to our framework, please follow along the `example.ipynb`. 

Alternatively, you can run individual components of the framework by running:
- `python3 -m Signal_Processing.harmonize`
- `python3 -m Signal_Processing.FDR_correction`
- `python3 -m Model_and_Training.sEEGDataset`
- `python3 -m Model_and_Training.SingleSubjectTrain`
- `python3 -m Model_and_Training.MultiSubjectTrain`
- `python3 -m Model_and_Training.TransferPreTrained`

## Citation (Coming soon)

We hope that you will find this code useful. If you do, please consider citing our work.

## Acknowledgments
We would like to thank the authors of the following repositories for making their code publicly available.
- [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)
- [POYO](https://github.com/neuro-galaxy/poyo)
