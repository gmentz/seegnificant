import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class ParticipantDataset(Dataset):
    def __init__(self, participants: list, path_to_data):
        self.participants = list()  # List containing the names of participants
        self.X = None  # Contains sEEG data: trial x electrodes x timepoints
        self.y = None  # Contains the response time for each trial
        self.biny = None  # Response time binarized to fast and slow based on the median response time of each subject
        self.id = None  # Participant ID: [0, ..., 21]
        self.mni_coords = None  # MNI Coordinates of each electrode: trial x electrodes x (MNIx, MNIy, MNIz)

        all_participants = ['Synth_' + x for x in ['1', '2', '3']]

        id_codes = dict((name, idx) for idx, name in enumerate(all_participants))

        for part in participants:

            id = id_codes[part]

            # Import the subject
            with open(path_to_data + '/' + part, 'rb') as handle:
                subject = pickle.load(handle)

            # Re epoch around the color - change
            # subject.re_epoch_data_around_color_change()
            # Keep only the task related electrodes
            subject.keep_electrodes_that_are_true(subject.task_related_locations_prob_fdr_significant)
            # # Invert the RT normalization
            # subject.inverse_normalize_RT()

            num_trials, samples, num_elec, numfbands = subject.envelope.shape
            self.participants.append(part)  # Name of participant

            # Save the envelope amplitude in multiple frequency bands
            if self.X is None:
                self.X = np.pad(subject.eeg_data, ((0, 0), (0, 0), (0, 28 - num_elec)), mode='constant')
                self.y = subject.beh_df['RT'].to_numpy() / 1000
                self.biny = self.y < np.median(self.y)  # Fast = 1 / slow = 0
                self.id = id * np.ones((subject.envelope.shape[0]))
                # Store the information about the position of each electrode (MNI coords)
                mni_coords = subject.anatDfThisSubj[['mni_x', 'mni_y', 'mni_z']].to_numpy()
                mni_coords = np.pad(mni_coords, ((0, 28 - num_elec), (0, 0), ), mode='constant')
                # Make shape equal to that of the X
                mni_coords = np.repeat(mni_coords[np.newaxis, :, :], num_trials, axis=0)
                self.mni_coords = mni_coords
            else:
                self.X = np.vstack([self.X, np.pad(subject.eeg_data, ((0, 0), (0, 0), (0, 28 - num_elec)), mode='constant')])
                self.y = np.hstack([self.y, subject.beh_df['RT'].to_numpy() / 1000])
                self.biny = np.hstack([self.biny, subject.beh_df['RT'].to_numpy() < np.median(subject.beh_df['RT'].to_numpy())])
                self.id = np.hstack([self.id, id * np.ones((subject.envelope.shape[0]))])
                # Store the information about the position of each electrode
                mni_coords = subject.anatDfThisSubj[['mni_x', 'mni_y', 'mni_z']].to_numpy()
                mni_coords = np.pad(mni_coords, ((0, 28 - num_elec), (0, 0), ), mode='constant')
                # Make shape equal to that of the X
                mni_coords = np.repeat(mni_coords[np.newaxis, :, :], num_trials, axis=0)
                self.mni_coords = np.vstack([self.mni_coords, mni_coords])

        self.X = np.swapaxes(self.X, 1, 2)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx, :, :]
        y = self.y[idx]
        biny = self.biny[idx]
        mni_pos = self.mni_coords[idx, :, :]
        x = x  # x[:, 400:1000]  # Keep time from 0 up to 1 sec after the color-change

        return torch.Tensor([self.id[idx]]).squeeze(), torch.Tensor(x).unsqueeze(0), torch.Tensor(mni_pos), torch.Tensor([y]), torch.Tensor([biny])


if __name__ == "__main__":
    # Make the dataset
    dat = ParticipantDataset(['Synth_' + x for x in ['1', '2', '3']], os.getcwd() + '/participants_fdr/')
    # Save the dataset
    torch.save(dat, os.getcwd() + '/data/dataset.pt')
    # Ensure that you can load the dataset.
    dat = torch.load(os.getcwd() + '/data/dataset.pt')

    # print('Hello')
