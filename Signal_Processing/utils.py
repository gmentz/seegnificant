import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import pickle

class participant:
    def __init__(self, fpath: str):
        '''
        :param fpath: Absolute path to pickle file containing the data for the participant
        '''

        if os.path.exists(fpath):
            thisSubjData = pd.read_pickle(fpath)
        else:
            sys.exit('Filepath does not exist')

        self.eeg_data = thisSubjData[0]
        self.eeg_metadata = thisSubjData[1]
        self.beh_df = thisSubjData[2]
        self.anatDfThisSubj = thisSubjData[3]
        self.anatDfThisSubj_yeo = thisSubjData[4]

        # Variables concerning phase estimation
        self.eeg_filtered = None
        self.phase = None
        self.envelope = None
        self.fbands = list()

        # Store the time vector associated with each epoch
        self.timestamps_sec = self.eeg_metadata['timestamps'] / self.eeg_metadata['samplerate']

        # Store the responsive electrodes, if identified
        self.task_related_locations_prob = None
        self.idx_baseline = None
        self.idx_task = None

        # Store the mean and std of the response times of the subject
        self.RT_mu = None
        self.RT_std = None
        self.RT_targ_mu = None
        self.RT_targ_std = None

        # Remove invalid trials
        self.remove_invalid_trials()

    def plot_epochs(self, trials_to_plot: list, channels_to_plot: list, plot_mean=True, plot_individual_epochs=True):

        number_of_channels_to_plot = len(channels_to_plot)
        timestamps_sec = self.eeg_metadata['timestamps'] / self.eeg_metadata['samplerate']

        # Case where you plot one channel only
        if number_of_channels_to_plot == 1:
            fig = plt.figure()
            # Plot each trial
            if plot_individual_epochs:
                for trial in trials_to_plot:
                    plt.plot(timestamps_sec, self.eeg_data[trial, :, channels_to_plot[0]])
                # Plot trial mean
            if plot_mean:
                plt.plot(timestamps_sec, np.mean(self.eeg_data[trials_to_plot, :, channels_to_plot[0]], 0), color='k', linewidth=3)

            # Plot axlines to denote the colorchange
            plt.axvline(0.5, color='k')
            plt.axvline(1, color='k')
            plt.axvline(1.5, color='k')
            # Set the labels
            plt.xlabel('Time (sec)')
            plt.ylabel('Voltage (?)')
            plt.title(f'Channel {channels_to_plot[0]}')

        # Case where you plot multiple channels.
        else:
            fig, ax = plt.subplots(number_of_channels_to_plot)
            subfig_idx = 0
            # Plot individual channels
            for channel in channels_to_plot:
                if plot_individual_epochs:
                    for trial in trials_to_plot:
                        ax[subfig_idx].plot(timestamps_sec, self.eeg_data[trial, :, channel])

                # Plot the mean of each channel
                if plot_mean:
                    ax[subfig_idx].plot(timestamps_sec, np.mean(self.eeg_data[trials_to_plot, :, channel], 0), color='k', linewidth=3)

                # Plot axvlineas to denote the colorchange
                ax[subfig_idx].axvline(x=0.5, color='k')
                ax[subfig_idx].axvline(x=1, color='k')
                ax[subfig_idx].axvline(x=1.5, color='k')
                # Set the labels
                ax[subfig_idx].set(xlabel='Time (sec)', ylabel='Voltage (?)')
                ax[subfig_idx].set_title(f'Channel: {channel}')

                # Update the subplot idx for each channel.
                subfig_idx += 1
        return fig

    def filter_eeg_data(self, low_cuttoff, high_cuttoff, filter_order=4, plot_frequency_response=False):
        '''
        :param low_cuttoff: Lower cuttoff Frequency (Hz)
        :param high_cuttoff: Upper cuttoff frequency (Hz)
        :param filter_order: Order of the filter (must be even)
        :param plot_frequency_response: If true, a bode plot of the filter's frequency response is plotted
        :return: Replaces the eeg_data with the filtered eeg_data
        '''

        # Create a butterworth filter:
        sos = signal.butter(filter_order, [low_cuttoff, high_cuttoff], btype='bandpass', output='sos', fs=self.eeg_metadata['samplerate'])

        # Plot the frequency response
        if plot_frequency_response:
            w, h = signal.sosfreqz(sos, fs=self.eeg_metadata['samplerate'])
            plt.figure()
            plt.subplot(2, 1, 1)
            db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
            plt.plot(w, db)
            plt.grid(True)
            plt.ylabel('Gain (dB)')
            plt.xlabel('Frequency (Hz)')
            plt.title('Frequency Response')
            plt.subplot(2, 1, 2)
            plt.plot(w, np.angle(h))
            plt.grid(True)
            plt.ylabel('Phase (rad)')

        # Filter the data forwards and backwards to minimize phase distortion
        return signal.sosfiltfilt(sos, self.eeg_data, axis=1)

    def filter_and_get_phase_and_envelope(self, freq_band: tuple):
        '''
        Calculated the phase of each trial of each channel of the eeg_data using the Hilbert Transform
        ** Consider changing so that it automatically calculates the phase in multiple different frequency bands **
        :return:
        '''
        filtered_eeg = self.filter_eeg_data(freq_band[0], freq_band[1])
        analytic_signal = signal.hilbert(filtered_eeg, axis=1)
        return filtered_eeg, np.angle(analytic_signal), np.abs(analytic_signal)

    def calculate_phase_and_envelope_in_fbands(self, fbands: list):
        r, c, z = self.eeg_data.shape
        d = len(fbands)
        self.phase = np.empty((r, c, z, d))  # Array that stores phase of the filtered sEEG
        self.envelope = np.empty((r, c, z, d))  # Array that stores phase of the filtered sEEG
        self.eeg_filtered = np.empty((r, c, z, d))  # Array that stored the filtered sEEG
        self.fbands = fbands  # Store the frequency bands for which phase has been calculated.

        for counter, band in zip(range(len(fbands)), fbands):
            self.eeg_filtered[:, :, :, counter], self.phase[:, :, :, counter], self.envelope[:, :, :, counter] = self.filter_and_get_phase_and_envelope(band)

    def plot_phase_and_envelope(self, trials: list, channels: list, fband: tuple):
        idx = -1
        for i in range(len(self.fbands)):
            if fband == self.fbands[i]:
                idx = i
                break
        if idx == -1:
            print('The specified frequency band does not exist.')
            return

        timestamps_sec = self.eeg_metadata['timestamps'] / self.eeg_metadata['samplerate']
        for ch in channels:
            for tr in trials:
                plt.figure()
                plt.plot(timestamps_sec, self.eeg_filtered[tr, :, ch, idx], label='Signal')
                plt.plot(timestamps_sec, self.envelope[tr, :, ch, idx], label='Envelope')
                plt.plot(timestamps_sec, self.phase[tr, :, ch, idx], label='Phase')
                plt.legend()
                plt.xlabel('Time (sec)')
                plt.ylabel('Amplitude or phase (rad)')
                plt.title(f'Chanel: {ch} | Trial: {tr} | Frequency: [{fband[0]}, {fband[1]}]')

    def plot_response_times_as_histogram(self, n_bins=50):
        RT = self.beh_df.reset_index().pop('RT')
        ax = RT.plot.hist(bins=n_bins)
        ax.set_xlabel('Response Time (ms)')
        ax.set_ylabel('Count (N)')

    def plot_response_times_by_trial(self):
        RT = self.beh_df.reset_index().pop('RT')
        plt.figure()
        plt.plot(RT)
        plt.xlabel('Trial')
        plt.ylabel('Response time (ms)')

    def remove_bad_trials_and_electrodes(self, bad_trials_and_electrodes):
        # Check which channels and trials are bad.
        self.bad_trials_and_electrodes = bad_trials_and_electrodes
        bad_trials = np.mean(bad_trials_and_electrodes, axis=1) > self.bad_trial_thresh
        bad_channels = np.mean(bad_trials_and_electrodes, axis=0) > self.bad_chan_thresh
        # Remove the bad channels/trials
        self.eeg_data = self.eeg_data[bad_trials == False, :, :]
        self.eeg_data = self.eeg_data[:, :, bad_channels == False]
        self.beh_df = self.beh_df.iloc[bad_trials == False, :]
        self.anatDfThisSubj = self.anatDfThisSubj.iloc[bad_channels == False, :]
        self.anatDfThisSubj_yeo = self.anatDfThisSubj_yeo.iloc[bad_channels == False, :]

    def keep_electrodes_that_are_true(self, is_responsive):
        # Keep only responsive electrodes:
        self.eeg_data = self.eeg_data[:, :, is_responsive]
        self.anatDfThisSubj = self.anatDfThisSubj.iloc[is_responsive, :]
        self.anatDfThisSubj_yeo = self.anatDfThisSubj_yeo.iloc[is_responsive, :]

        if self.fbands:
            self.eeg_filtered = self.eeg_filtered[:, :, is_responsive, :]
            self.phase = self.phase[:, :, is_responsive, :]
            self.envelope = self.envelope[:, :, is_responsive, :]

        if self.task_related_locations_prob is None:
            pass
        else:
            self.task_related_locations_prob = self.task_related_locations_prob[is_responsive]

    def remove_invalid_trials(self):
        valid_trials = (self.beh_df.error == False).to_numpy()
        self.eeg_data = self.eeg_data[valid_trials, :, :]
        self.beh_df = self.beh_df.iloc[valid_trials, :]
        if self.fbands:
            self.eeg_filtered = self.eeg_filtered[valid_trials, :, :, :]
            self.phase = self.phase[valid_trials, :, :, :]
            self.envelope = self.envelope[valid_trials, :, :, :]

    def normalize(self, idx_baseline=None):
        if idx_baseline is None:
            # z-score across the signal
            self.eeg_data = (self.eeg_data - np.mean(self.eeg_data, axis=(0, 1))) / np.std(self.eeg_data, axis=(0, 1))
            for j in range(self.eeg_filtered.shape[-1]):
                self.eeg_filtered[:, :, :, j] = (self.eeg_filtered[:, :, :, j] - np.mean(self.eeg_filtered[:, :, :, j], axis=(0, 1))) / np.std(self.eeg_filtered[:, :, :, j], axis=(0, 1))
                self.envelope[:, :, :, j] = (self.envelope[:, :, :, j] - np.mean(self.envelope[:, :, :, j], axis=(0, 1))) / np.std(self.envelope[:, :, :, j], axis=(0, 1))
        else:
            # z-score based on mu and sigma of baseline
            self.eeg_data = (self.eeg_data - np.mean(self.eeg_data[:, idx_baseline, :], axis=(0, 1))) / np.std(self.eeg_data[:, idx_baseline, :], axis=(0, 1))
            for j in range(self.eeg_filtered.shape[-1]):
                baseline = self.eeg_filtered[:, idx_baseline, :, :]
                self.eeg_filtered[:, :, :, j] = (self.eeg_filtered[:, :, :, j] - np.mean(baseline[:, :, :, j], axis=(0, 1))) / np.std(baseline[:, :, :, j], axis=(0, 1))
                baseline = self.envelope[:, idx_baseline, :, :]
                self.envelope[:, :, :, j] = (self.envelope[:, :, :, j] - np.mean(baseline[:, :, :, j], axis=(0, 1))) / np.std(baseline[:, :, :, j], axis=(0, 1))

