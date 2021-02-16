# @author semese

import csv
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable

from src.utils.hmm_utils import load_model

ONE_MONTH = 30


# ------------------------------------ Data set generation for Static models ----------------------------------------- #
def df_to_list(df, columns):
    """
    Function to transform a dataframe into a list of observation sequences grouped by patient.

    :param pd.DataFrame df: dataframe containing the patient data
    :param list columns: list of columns to keep for training
    :return: list of observation sequences grouped by patient
    """
    res = []
    # for each patient
    for unq in df.id.unique():
        # extract observations
        res.append(df[df.id == unq][columns].to_numpy())
    return res


def extract_labelled_samples(df, columns, dt, mix_mod):
    """
    Split up each patient sequence such that it contains a labelled observation as last element.
    :param pd.DataFrame df: data frame containing patient observation sequences
    :param list columns: list of column labels to extract
    :param int dt: number of days before label to consider
    :param bool mix_mod: indicator whether a MM or a HMM will be used for imputation; if a MM is used,
        then only the dt amount of days are extracted, otherwise the entire sequence before
    :return:
    """
    """
        Split up each patient sequence such that it contains a labelled
        observation as last element.
    """
    X = []
    meta = []
    for pid in df.id.unique():
        df_par = df[df.id == pid][columns]
        label_inds = [
            idx[0] for idx in np.argwhere(df_par.iloc[:, -1].notnull().values).tolist()
        ]
        for idx in label_inds:
            if (idx + 1) - dt >= 0:
                if not mix_mod:
                    X.append(df_par.iloc[0: idx + 1, :].values)
                else:
                    X.append(df_par.iloc[idx + 1 - dt: idx + 1, :].values)
                meta.append(df[df.id == pid][['id', 'date', columns[-1]]].iloc[idx, :].to_list())
    return X, meta


def load_and_impute(df, columns, hmm_path, dt=1, n_processes=4, mix_mod=False):
    """
    Load patient data and perform imputation and feature extraction using a pretrained MM or HMM.
    :param pd.DataFrame df: data frame containing patient observation sequences
    :param list columns: list of column labels to extract
    :param str hmm_path:
    :param int dt:
    :param int n_processes:
    :param bool mix_mod: indicator whether a MM or a HMM will be used for imputation;
    :return:
    """
    # df to array
    data_list, meta = extract_labelled_samples(df, columns, dt, mix_mod)

    # load pre-trained MM/HMM
    hmm = load_model(hmm_path)

    # split up observation sequences between the processes
    n_splits = int(np.ceil(len(data_list) / n_processes))
    split_list = [
        sl
        for sl in list(
            (
                data_list[i * n_splits: i * n_splits + n_splits]
                for i in range(n_processes)
            )
        )
        if sl
    ]

    # create pool of processes and perform the imputation
    p = Pool(processes=n_processes)
    imp_list = p.map(
        partial(impute_missing, hmm=hmm, dt=dt),
        [split_ij for split_i in split_list for split_ij in split_i],
    )
    p.close()

    if dt == 1:
        state_columns = define_new_feature_columns(dt, hmm.n_states, mix_mod)
        df_imp = pd.DataFrame(
            np.vstack(imp_list),
            columns=columns[:-1] + state_columns,
        )
        df_imp = df_imp.assign(
            **{"id": [m[0] for m in meta], "date": [m[1] for m in meta], columns[-1]: [m[2] for m in meta]})
        df_imp = df_imp[["id", "date"] + columns[:-1] + state_columns + [columns[-1]]]
    else:
        data_columns = define_new_feature_columns(dt, hmm.n_states, mix_mod, columns[:-1])

        df_imp = pd.DataFrame(
            np.vstack([x.reshape(1, -1) for x in imp_list]),
            columns=data_columns,
        )
        df_imp = df_imp.assign(
            **{"id": [m[0] for m in meta], "date": [m[1] for m in meta], columns[-1]: [m[2] for m in meta]})
        df_imp = df_imp[["id", "date"] + data_columns + [columns[-1]]]

    return df_imp


def impute_missing(obs_seq, hmm, dt):
    """
    Function to impute the patient sequences using the pre-trained HMM.
    It also attaches the state posterior probabilities to the features.

    :param np.ndarray obs_seq: an observation sequence
    :param hmm: a trained MM or HMM
    :param int dt: number of previous observations to take into account
    :return: the imputed sequences with the attached posterior probabilities
    """
    imputed_obs = obs_seq[-1, :-1].reshape(1, -1) if dt == 1 else obs_seq[-dt:, :-1]

    # set last emotion to missing
    obs_seq[-1, -1] = np.nan

    # compute most probable state sequence
    state_seq = hmm.decode([obs_seq], algorithm="viterbi")[1][0]

    for t in range(dt, 0, -1):
        # create mask for NaN values in the observation
        nan_mask = np.isnan(obs_seq[-t, :-1])

        if np.any(nan_mask):
            # generate a sample from the state and make sure they are all positive
            sample = hmm.generate_sample_from_state(state_seq[-t])

            # fill in missing values from the generated sample
            imputed_obs[-t, nan_mask] = sample[:-1][nan_mask]

    # compute the posteriors for the sequence
    posteriors = hmm.score_samples([obs_seq], fwd_only=True)[0]
    posteriors = posteriors[-1, :].reshape(1, -1) if dt == 1 else posteriors[-dt:, :]

    # append the posteriors to the features
    feature_seq = np.hstack((imputed_obs, posteriors)).astype(np.float32)

    return feature_seq


def define_new_feature_columns(dt, ns, mix_mod, data_columns=None):
    """
    Define new feature column names for the imputed dataframe with posterior probabilities.

    :param int dt: number of days before label to consider
    :param int ns: number of states in the generative model, which corresponds to the number of posterior probabilities
    :param mix_mod: indicator whether a MM or a HMM will be used for imputation;
    :param list data_columns: list of data column label names; only necessary if dt > 1
    :return: the redefined column labels
    """
    if dt == 1:
        return ["p(s_" + str(int(i)) + "|x_t)" for i in range(ns)] if mix_mod else [
            "p(s_" + str(int(i)) + "|x_{0:t})" for i in range(ns)]
    else:
        if data_columns is None:
            raise ValueError("data_columns must be provided if dt > 1")

        feature_columns = []
        for t in range(dt - 1, -1, -1):
            feature_columns += [col + "_t-" + str(int(t)) if t > 0 else col + "_t" for col in data_columns]
            feature_columns += [
                "p(s_" + str(int(i)) + "|x_{t-" + str(dt - 1) + ":t-" + str(int(t)) + "})" if t > 0 else "p(s_" + str(
                    int(i)) + "|x_{t-" + str(dt - 1) + ":t})" for i in range(ns)] if mix_mod else [
                "p(s_" + str(int(i)) + "|x_{0:t-" + str(int(t)) + "})" if t > 0 else "p(s_" + str(
                    int(i)) + "|x_{0:t})" for i in range(ns)]

        return feature_columns


# --------------------------------------- Data set generation for RNNs ----------------------------------------------- #
def get_patient_dfs(
        data_path,
        x_columns,
        y_columns,
        seq_len=365,
        miss_len=91,
        pad_token=None,
        pre_pad=True,
):
    """
    Load patient sequences for the training and test sets, and pre-process them according
    to the RNN
    :param data_path:
    :param x_columns:
    :param y_columns:
    :param seq_len:
    :param miss_len:
    :param pad_token:
    :param pre_pad:
    :return:
    """
    # read data from csv files
    df_train = pd.read_csv(data_path + "df_train.csv")
    df_test = pd.read_csv(data_path + "df_test.csv")

    # make sure of the ordering
    df_train["date"] = pd.to_datetime(df_train["date"], format="%Y-%m-%d")
    df_train.sort_values(by=["id", "date"], ascending=[True, True], inplace=True)
    df_test["date"] = pd.to_datetime(df_test["date"], format="%Y-%m-%d")
    df_test.sort_values(by=["id", "date"], ascending=[True, True], inplace=True)

    # transform dataframe columns to list of observation sequences
    X_train = df_to_list(df_train, x_columns)
    y_train = df_to_list(df_train, y_columns)

    X_test = df_to_list(df_test, x_columns)
    y_test = df_to_list(df_test, y_columns)

    df_train = process_sequences(
        X_train, y_train, seq_len, miss_len, pad_token, pre_pad
    )

    df_test = process_sequences(X_test, y_test, seq_len, miss_len, pad_token, pre_pad)

    return df_train, df_test


def process_sequences(obs_sequences, labels, seq_len, miss_len, pad_token, pre_pad):
    """
    Process observation sequences by first removing consecutively missing chunks, then limiting maximum sequence length,
    finally padding the short sequences.

    :param list obs_sequences: list of data sequences
    :param list labels: list of target sequences
    :param int seq_len: maximum sequence length to which the sequences have to be cut
    :param int miss_len: threshold for consecutive missing rows
    :param int pad_token: token to be used as padding in the sequences
    :param bool pre_pad: if True, the sequences are pre-padded, otherwise post-padded
    :return:
    """

    # if a sequence contains more than miss_len consecutive missing observations, cut that part out and divide it
    obs_sequences, labels = deal_with_consecutive_missing(
        obs_sequences, labels, seq_len, miss_len, pad_token
    )

    # if a sequence is longer than seq_len, chop it up
    obs_sequences, labels = deal_with_too_long(
        obs_sequences, labels, seq_len, pad_token
    )

    # pad the features to have equal size mini-batches
    obs_sequences, labels, lengths = pad_obs_sequences(
        obs_sequences, labels, seq_len=seq_len, pad_token=pad_token, pre_pad=pre_pad
    )

    return (
        torch.from_numpy(np.asarray(obs_sequences)),
        torch.from_numpy(np.asarray(labels)),
        torch.from_numpy(np.asarray(lengths)),
    )


def deal_with_consecutive_missing(obs_sequences, labels, seq_len, miss_len, pad_token):
    """
    Remove missing chunks from observation sequences that are longer than a given threshold.

    :param list obs_sequences: list of data sequences
    :param list labels: list of target sequences
    :param int seq_len: maximum sequence length to which the sequences have to be cut
    :param int miss_len: threshold for consecutive missing rows
    :param int pad_token: token to be used as padding in the sequences
    :return: observation and label sequences with removed missing chunks
    """
    new_obs_sequences = []
    new_labels = []

    for obs_seq, lab in zip(obs_sequences, labels):
        missing_row_idx = get_consecutive_nan_row_indices(obs_seq, miss_len)
        if not missing_row_idx:
            new_obs_sequences.append(obs_seq)
            new_labels.append(lab)
        else:
            new_seq, new_lab = chop_up_sequence(
                obs_seq, lab, seq_len, missing_row_idx, pad_token
            )
            for ns, nl in zip(new_seq, new_lab):
                new_obs_sequences.append(ns)
                new_labels.append(nl)

    return new_obs_sequences, new_labels


def chop_up_sequence(obs_seq, labels, seq_len, missing_row_idx, pad_token):
    """
    Segment observation sequences, cutting out long missing chunks.

    :param np.ndarray obs_seq: observation sequence
    :param np.ndarray labels: label sequence
    :param int seq_len: maximum sequence length to which the sequences have to be cut
    :param list missing_row_idx: list of indices indicating missing chunks
    :param int pad_token: token to be used as padding in the sequences
    :return: segmented observation sequence and corresponding label sequence
    """
    new_seq = []
    new_lab = []
    for i in missing_row_idx[0::2]:
        ns = obs_seq[i[0]: i[1], :]
        nl = labels[i[0]: i[1]]
        if pad_token is None:
            if len(ns) >= seq_len and not np.isnan(nl).all():
                new_seq.append(ns)
                new_lab.append(nl)
        else:
            if not np.isnan(nl).all() and len(nl) >= ONE_MONTH:
                new_seq.append(ns)
                new_lab.append(nl)

    return new_seq, new_lab


def get_consecutive_nan_row_indices(obs_seq, miss_len):
    """
    Get list of consecutively missing row indices in an observation sequence.

    :param np.ndarray obs_seq: observation sequence
    :param int miss_len: threshold for consecutive missing rows
    :return: list of indices
    """
    nan_row_found = False
    missing_row_idx = [0]
    row_count = 1
    first_idx, last_idx = 0, 0
    # for each row
    for i, obs in enumerate(obs_seq):
        # if there are only NaNs
        if np.isnan(obs).all():
            if nan_row_found:
                row_count += 1
                last_idx = i
            else:
                first_idx = i
                nan_row_found = True
        else:
            if nan_row_found:
                if row_count > miss_len:
                    missing_row_idx += [first_idx, last_idx]
                nan_row_found = False
                row_count = 0

    return [missing_row_idx + [len(obs_seq) - 1]]


def deal_with_too_long(obs_sequences, labels, seq_len, pad_token):
    """
    Segment too long observation sequences

    :param list obs_sequences: list of data sequences
    :param list labels: list of target sequences
    :param int seq_len: aximum sequence length to which the sequences have to be cut
    :param int pad_token: token to be used as padding in the sequences
    :return: segmented observation sequences and corresponding label sequences
    """
    new_obs_sequences = []
    new_labels = []

    for obs_seq, lab in zip(obs_sequences, labels):
        if pad_token is None:
            if len(obs_seq) == seq_len:
                new_obs_sequences.append(obs_seq)
                new_labels.append(lab)
            else:
                n_seq = int(len(obs_seq) / seq_len)
                new_seq = np.array_split(obs_seq, n_seq)
                new_lab = np.array_split(lab, n_seq)
                for ns, nl in zip(new_seq, new_lab):
                    if len(ns) == seq_len and not np.isnan(nl).all():
                        new_obs_sequences.append(ns)
                        new_labels.append(nl)
                    elif len(ns) > seq_len:
                        ns = ns[0:seq_len]
                        nl = nl[0:seq_len]
                        if not np.isnan(nl).all() and len(nl) >= ONE_MONTH:
                            new_obs_sequences.append(ns)
                            new_labels.append(nl)
        else:
            if seq_len >= len(obs_seq) >= ONE_MONTH:
                new_obs_sequences.append(obs_seq)
                new_labels.append(lab)
            else:
                n_seq = int(len(obs_seq) / seq_len)
                seq_lens = [seq_len * (i + 1) for i in range(n_seq)]
                new_seq = np.split(obs_seq, seq_lens)
                new_lab = np.split(lab, seq_lens)
                for ns, nl in zip(new_seq, new_lab):
                    if not np.isnan(nl).all() and len(nl) >= ONE_MONTH:
                        new_obs_sequences.append(ns)
                        new_labels.append(nl)

    return new_obs_sequences, new_labels


def pad_obs_sequences(
        obs_sequences, labels, seq_len=None, pad_token=-999, pre_pad=True
):
    """
    Function to pad each sequence with pad_token or truncate to the desired seq_length.

    :param list obs_sequences: list of data sequences
    :param list labels: list of target sequences
    :param int seq_len: maximum sequence length to which the sequences have to be cut or padded
    :param int pad_token: token to be used as padding in the sequences
    :param bool pre_pad: if True, the sequences are pre-padded, otherwise post-padded
    :return: the padded data and target sequences, and the lengths of each sequence in a list
    """
    # get the number of sequences
    n_seq = len(obs_sequences)

    # get the length of each sentence
    lengths = [len(obs) for obs in obs_sequences]
    if seq_len is None:
        seq_len = max(lengths)
    lengths = [ln if ln <= seq_len else seq_len for ln in lengths]

    # getting the correct rows x cols shape
    padded_obs_sequences = pad_token * np.ones(
        (n_seq, seq_len, obs_sequences[0].shape[1])
    )
    padded_labels = pad_token * np.ones((n_seq, seq_len, labels[0].shape[1]))

    # for each sequence
    for i, (obs, lab) in enumerate(zip(obs_sequences, labels)):
        if pre_pad:
            padded_obs_sequences[i, -len(obs):, :] = obs[:seq_len]
            padded_labels[i, -len(obs):, :] = lab[:seq_len]
        else:
            padded_obs_sequences[i, 0: len(obs), :] = obs[:seq_len]
            padded_labels[i, 0: len(obs), :] = lab[:seq_len]

    return padded_obs_sequences, padded_labels, lengths


# ----------------------------------------------- Collator for RNNs -------------------------------------------------- #
class VariableLengthCollator(object):
    def __init__(
            self, hmm_path, pad_token, add_posteriors=False, only_posteriors=False
    ):
        """
        Class initializer.

        :param str hmm_path: path to trained HMM models
        :param int pad_token: token used as padding in the sequences
        :param bool add_posteriors: indicator whether to add the posterior probabilities to the features or not
        :param bool only_posteriors: indicator whether to use only the posterior probabilities as features
        """
        self.hmm = load_model(hmm_path)
        self.pad_token = pad_token
        self.add_posteriors = add_posteriors
        self.only_posteriors = only_posteriors

    def __call__(self, batch):
        orig_features, labels, lengths = zip(*batch)

        # list to store the imputed features
        features = []

        # perform imputation on each sequence
        for i in range(len(batch)):
            obs_seq = np.hstack(
                (orig_features[i].numpy(), labels[i].numpy().reshape(-1, 1))
            )
            imputed = self.impute_missing_data_from_hmm(obs_seq)
            features.append(torch.from_numpy(imputed))

        return torch.stack(features, dim=0), torch.stack(labels), torch.stack(lengths)

    def impute_missing_data_from_hmm(self, obs_seq):
        """
        Function to perform data imputation on the mini-batches. For this a
        pre-trained HMM models is used. We decode the observation sequences
        with the HMM, then for each time step, we draw samples from the
        corresponding state and fill in the missing values with that. We
        don't fill in the labels, only the input data.
        As an extra feature we add the posterior probabilities of the imputed
        observation sequence.

        :param np.ndarray obs_seq: a sequence of n-dimensional time series
        :return: imputed sequence with or without posteriors.
        """
        # for each observation in the sequence we will impute the missing
        # values by a sample generated from an HMM
        imputed_obs_seq = obs_seq[:, :-1]
        data_mask = np.all(
            imputed_obs_seq != self.pad_token, axis=1
        )  # all values in each row aren't the padding
        masked_seq = obs_seq[data_mask]
        # set all emotions to missing
        masked_seq[:, -1] = np.nan

        state_seq = self.hmm.decode([masked_seq], algorithm="viterbi")[1][0]

        for t, obs in enumerate(masked_seq):
            # create mask for NaN values in the observation
            nan_mask = np.isnan(obs[:-1])
            if np.any(nan_mask):
                # generate a sample from the state and make sure they are all positive
                sample = self.hmm.generate_sample_from_state(state_seq[t])
                # fill in missing values from the generated sample
                masked_seq[t, :-1][nan_mask] = sample[:-1][nan_mask]
        imputed_obs_seq[data_mask] = masked_seq[:, :-1]

        if self.add_posteriors:
            # initialise an array to store the posteriors
            posteriors = self.pad_token * np.ones(
                (len(imputed_obs_seq), self.hmm.n_states)
            )
            # compute the posteriors for the sequence
            posteriors[data_mask] = self.hmm.score_samples([masked_seq])[0]
            if not self.only_posteriors:
                # append the posteriors to the features
                feature_seq = np.hstack((imputed_obs_seq, posteriors)).astype(
                    np.float32
                )
            else:
                feature_seq = posteriors.astype(np.float32)
        else:
            feature_seq = imputed_obs_seq.astype(np.float32)

        return feature_seq


# --------------------------------------- Train-Test split for the Hierarchical Model -------------------------------- #
def train_test_split(data, x_columns, y_column, p=0.2):
    """
    Function to split the patient sequences into training and test parts.
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    pidx_train, pidx_test = [], []
    patient_lookup = {}
    pidx = 0
    for pid in data.id.unique():
        assert len(X_test) == len(X_train)

        df_patient = data[data.id == pid]
        split_idx = int(df_patient.shape[0] * (1 - p))

        X_train.append(df_patient[x_columns].values[:split_idx, :])
        y_train.append(df_patient[y_column].values[:split_idx])
        pidx_train += [pidx] * X_train[-1].shape[0]

        X_test.append(df_patient[x_columns].values[split_idx:, :])
        y_test.append(df_patient[y_column].values[split_idx:])
        pidx_test += [pidx] * X_test[-1].shape[0]

        patient_lookup[pid] = pidx

        pidx += 1

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=None)

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=None)

    return X_train, y_train, np.asarray(pidx_train), X_test, y_test, np.asarray(pidx_test), patient_lookup


# ---------------------------------------------- Data Analysis Utils ------------------------------------------------- #

def entry_count(df, columns, print_it=True):
    """
    Count and print the unique values in the DataFrame.

    :param df: the pandas dataframe to work with
    :param str/list columns: label of the columns in the DataFrame
    :param bool print_it: indicator print the statistics or return the value count as it is
    :return: a Series containing counts of unique rows in the DataFrame
    """
    """
        Function to create entry statistics in a given column in a pandas DataFrame.
        Args:
            df (DataFrame) - the pandas dataframe to work with
            columns (str or list of str) - label of the columns in the DataFrame
            print_it (boolean) - indicator print the statistics or return the
                the value count as it is
    """
    # --- count entries per patient
    entry_count = df[columns].value_counts()

    if print_it:
        # --- print statistics of the entry count
        print(entry_count.describe())

    return entry_count


def compute_missing_data_stats(df, pid, data_columns):
    """
    Compute missing data statistics on the given data frame.
    :param pd.DataFrame df: data frame containing time series observations
        of patients
    :param str pid: column name of patient ids
    :param list data_columns: data columns of interest
    :return: list of missing data info
    """
    # counting number of patients
    n_patients = df[pid].nunique(dropna=True)

    # length range for observation sequences
    patient_count = entry_count(df, pid, print_it=False)
    min_obs_len = patient_count.min()
    max_obs_len = patient_count.max()

    # counting partially missing observations
    data = df[data_columns].to_numpy()
    partially_missing = sum(
        [True for row in data if any(np.isnan(row)) and not all(np.isnan(row))]
    )
    p_partially_missing = float(
        "{0:.2f}".format((partially_missing * 100.0) / data.shape[0])
    )

    # counting completely missing observations
    completely_missing = sum([True for row in data if all(np.isnan(row))])
    p_completely_missing = float(
        "{0:.2f}".format((completely_missing * 100.0) / data.shape[0])
    )

    # full observations
    full = data.shape[0] - (partially_missing + completely_missing)
    p_full = float("{0:.2f}".format((full * 100.0) / data.shape[0]))

    return [
        n_patients,
        min_obs_len,
        max_obs_len,
        data.shape[0],
        full,
        p_full,
        partially_missing,
        p_partially_missing,
        completely_missing,
        p_completely_missing,
    ]


# --------------------------------------------------- IO Utils ------------------------------------------------------- #
def print_table(rows, header):
    """
    Pretty print a table.
    :param list rows: 2D list of table content
    :param list header: list of table headings
    :return:
    """
    t = PrettyTable(header)
    for row in rows:
        t.add_row(row)
    print(t)


def write_data_to_csv(filename, header, data):
    """

    :param str filename: full path or just file name where the csv is saved
    :param list header: list of table headings
    :param list data: 2D list of table content
    :return:
    """
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
