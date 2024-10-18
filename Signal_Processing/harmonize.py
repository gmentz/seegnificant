import numpy as np
import scipy
import pickle
import multiprocessing
import os
from .utils import participant

def randomize_and_calculate_SNR(combined_matrix, seed):
    """
    This function randomizes the rows of a 3D matrix, splits the data into baseline and task groups,
    computes the median for each group, and calculates the Signal-to-Noise Ratio (SNR)
    by comparing the variance of the task data to the baseline data.

    Parameters:
    combined_matrix (numpy.ndarray): A 3D array where rows represent trials, and columns/depth represent time points/electrodes.
                                      Shape: (num_trials, num_timepoints, num_electrodes)
    seed (int): The random seed to ensure reproducibility of the row shuffling.

    Returns:
    numpy.ndarray: An array representing the SNR for each electrode, calculated as
                   the variance of the task data divided by the variance of the baseline data.
    """
    # Set the random seed
    np.random.seed(seed)
    # Indexes
    order = np.arange(combined_matrix.shape[0])
    # Shuffle the indexes
    np.random.shuffle(order)
    # Keep half rows as baseline and half as task
    median_baseline = np.median(combined_matrix[order[:len(order)//2], :, :], axis=0)
    median_task = np.median(combined_matrix[order[len(order)//2:], :, :], axis=0)

    return np.nanvar(median_task, axis=0) / np.nanvar(median_baseline, axis=0)


# Specify the participant:
# participants = ['HUP' + x for x in ['133', '136', '139', '140', '142', '143', '145', '146', '150', '152',
#                                     '153', '154', '157', '160', '165', '168', '171', '178', '179', '181',
#                                     '182', '187', '191']]

if __name__ == '__main__':
    participants = ['Synth_1', 'Synth_2', 'Synth_3']

    fbands = [(4, 8), (8, 12), (15, 30), (30, 70), (70, 150)]  # Canonical frequency bands.
    time_baseline = (-0.5, 0)
    time_task = (0, 1.5)
    nperms = 100
    fprime = 400

    for part in participants:
        print(f'Processing participant: {part}.')

        # Import the subject
        subject = participant(os.getcwd() + '/data/' + part)

        # Filter in the broadband gamma range and extract the envelope
        subject.calculate_phase_and_envelope_in_fbands(fbands)

        # Downsample to 400 Hz
        fs = subject.eeg_metadata['samplerate']
        s = fs / fprime
        timestamps_sec = subject.timestamps_sec

        eeg_data, timestamps_eeg = scipy.signal.resample(subject.eeg_data, int(((timestamps_sec[-1] - timestamps_sec[0]) * fs) / s), t=timestamps_sec, axis=1)
        eeg_data_filtered, timestamps_eeg_filtered = scipy.signal.resample(subject.eeg_filtered, int(((timestamps_sec[-1] - timestamps_sec[0]) * fs) / s), t=timestamps_sec, axis=1)
        phase, timestamps_phase = scipy.signal.resample(subject.phase, int(((timestamps_sec[-1] - timestamps_sec[0]) * fs) / s), t=timestamps_sec, axis=1)
        envelope, timestamps_envelope = scipy.signal.resample(subject.envelope, int(((timestamps_sec[-1] - timestamps_sec[0]) * fs) / s), t=timestamps_sec, axis=1)

        # Check that the downsampling was the same for all variables
        assert np.all(timestamps_eeg == timestamps_eeg_filtered)
        assert np.all(timestamps_eeg == timestamps_phase)
        assert np.all(timestamps_eeg == timestamps_envelope)

        # Store the variables in their correct place in the participant object
        subject.eeg_data = eeg_data
        subject.eeg_filtered = eeg_data_filtered
        subject.envelope = envelope
        subject.phase = phase
        subject.eeg_metadata['timestamps'] = np.arange(subject.eeg_metadata['timestamps'][0] / fs * fprime,
                                                       subject.eeg_metadata['timestamps'][-1] / fs * fprime)
        subject.eeg_metadata['samplerate'] = fprime
        subject.timestamps_sec = subject.eeg_metadata['timestamps'] / subject.eeg_metadata['samplerate']

        assert len(subject.eeg_metadata['timestamps']) == len(timestamps_envelope)

        # Calculate the responsive electrodes based on SNR
        # Identify the samples that are associated with the task and those that are associated with the baseline
        idx_task = np.flatnonzero(np.logical_and(subject.timestamps_sec > min(time_task), subject.timestamps_sec < max(time_task)))
        idx_baseline = np.flatnonzero(np.logical_and(subject.timestamps_sec > min(time_baseline), subject.timestamps_sec < max(time_baseline)))

        # Save the indexes associated with the baseline and those associated with the task
        subject.idx_task = idx_task
        subject.idx_baseline = idx_baseline

        # Normalize eeg, filtered_eeg, and envelope to zero mean and unit variance
        subject.normalize()

        # Electrode selection / Calculate the SNR of neo-gamma between the task state and the baseline state
        gamma = subject.envelope[:, :, :, 4]
        norm_gamma_baseline = gamma[:, idx_baseline, :]
        norm_gamma_task = gamma[:, idx_task, :]
        median_norm_gamma_baseline = np.median(norm_gamma_baseline, axis=0)
        median_norm_gamma_task = np.median(norm_gamma_task, axis=0)
        SNR = np.var(median_norm_gamma_task, axis=0) / np.var(median_norm_gamma_baseline, axis=0)

        # Create the combined matrix by selecting columns from both matrices - used in permutation test
        if len(idx_baseline) > len(idx_task):
            pad_nans = np.full((norm_gamma_task.shape[0], len(idx_baseline)-len(idx_task), norm_gamma_task.shape[2]), np.nan)
            norm_gamma_task = np.hstack((norm_gamma_task, pad_nans))
        elif len(idx_baseline) < len(idx_task):
            pad_nans = np.full((norm_gamma_task.shape[0], len(idx_task)-len(idx_baseline), norm_gamma_task.shape[2]), np.nan)
            norm_gamma_baseline = np.hstack((norm_gamma_baseline, pad_nans))

        combined_matrix = np.vstack((norm_gamma_baseline, norm_gamma_task))

        # Permutation test for statistical significance
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            perm_SNRs = pool.starmap(randomize_and_calculate_SNR, [(combined_matrix, i) for i in range(nperms)])

        # Calculate the probability that an electrode has statistically significant SNR
        xx = np.vstack((perm_SNRs, SNR))
        ranks = scipy.stats.rankdata(xx, method='max', axis=0)
        subject.task_related_locations_prob = 1 - ranks[-1, :] / (nperms + 1)

        # Normalize response times to zero mean and unit variance
        # subject.normalize_RT()

        # Save the outputs in the participant directory
        if not os.path.exists(os.getcwd() + f'/participants'):
            os.mkdir(os.getcwd() + f'/participants/')
        else:
            pass
        with open(os.getcwd() + f'/participants/{part}', 'wb') as handle:
            pickle.dump(subject, handle, protocol=pickle.HIGHEST_PROTOCOL)

