from .utils import participant
import pickle
from statsmodels.stats.multitest import fdrcorrection as fdr
import os
import numpy as np

# Specify the subjects:
participants = ['Synth_1', 'Synth_2', 'Synth_3']

probs = dict()  # The probabilities

# Extract the probabilities for each participant
for part in participants:
    # print(f'Extracting p-values from participant: {part}.')

    # Import the subject
    with open(os.getcwd() + f'/participants/{part}', 'rb') as handle:
        subject = pickle.load(handle)

    # Extract the p-values
    probs[part] = subject.task_related_locations_prob

# Get all the p-values in a list
prob_values = list(probs.values())
flat_prob_values = [x for xs in prob_values for x in xs]

# Do FDR with significance level of 0.05
rejected, pvalues_corrected = fdr(flat_prob_values, alpha=0.05, is_sorted=False)

# Put the probabiltiies for each participant back in the participant object
for part in participants:
    # print(f'Importing corrected p-values to subject: {part}.')

    with open(os.getcwd() + f'/participants/{part}', 'rb') as handle:
        subject = pickle.load(handle)

    # Save the p-values that are relevant for each subject
    number_of_electrodes = subject.anatDfThisSubj.shape[0]
    subject.task_related_locations_prob_fdr = pvalues_corrected[:number_of_electrodes]
    subject.task_related_locations_prob_fdr_significant = rejected[:number_of_electrodes]
    print(f'Number of electrodes whose activity is modulated by the color-change for {part}: '
          f'{np.sum(rejected[:number_of_electrodes])} / {number_of_electrodes}')

    # Remove the values from the rest of the list
    pvalues_corrected = pvalues_corrected[number_of_electrodes:]
    rejected = rejected[number_of_electrodes:]

    #  Save the outputs in the new participant_fdr directory
    if not os.path.exists(os.getcwd() + f'/participants_fdr'):
        os.mkdir(os.getcwd() + f'/participants_fdr/')

    # Save the outputs in the participant_fdr directory
    with open(os.getcwd() + f'/participants_fdr/{part}', 'wb') as handle:
        pickle.dump(subject, handle, protocol=pickle.HIGHEST_PROTOCOL)

assert pvalues_corrected.size == 0
assert rejected.size == 0
print('Done')