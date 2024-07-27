

import os, sys, re, time, joblib, json
import torch, mne, h5py
import numpy as np
import scipy.signal as signal
from datetime import datetime
from pyentrp import entropy as ent
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve

import multiprocessing
from multiprocessing import Manager, Pool


#%% Model - LOAD and SAVE 
def save_model(model, model_name):
    """
    Saves the model to the disk with a specified naming convention.
    """
    model_path = os.path.join('saved_models', model_name)
    joblib.dump(model, model_path)
    return model_path

def load_latest_model(model_name_pattern):
    """
    Loads the latest model for a given data tag, feature tag, and fold index.
    """
    model_files = [f for f in os.listdir('saved_models') if f.startswith(model_name_pattern)]
    if model_files:
        latest_model_file = max(model_files, key=os.path.getctime)
        return joblib.load(os.path.join('saved_models', latest_model_file))
    else:
        return None
    
#%% Metrics - Segment level & Subject level

def calculate_mean_variance(metrics_list, metric_name):
    values = [metrics[metric_name] for metrics in metrics_list]
    mean_value = np.mean(values)
    variance_value = np.var(values)
    return mean_value, variance_value

def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Dictionary of metrics (accuracy, precision, recall, f1-score).
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    return metrics

def binary_threshold(predictions, threshold=0.5):
    """ Apply a binary threshold to continuous predictions. """
    return [1 if pred >= threshold else 0 for pred in predictions]

def optimal_threshold(true, predictions):
    """
    Youden's J statistic to obtain the optimal probability threshold and this method gives equal weights to both false positives and false negatives
    """
    fpr, tpr, thresholds = roc_curve(true, predictions)
    optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), thresholds)), key=lambda i: i[0], reverse=True)[0][1]
    print(f"Opt Threshould: {optimal_proba_cutoff}")
    binary_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in predictions]
    return binary_predictions


def aggregate_subject_level_predictions(subject_indices, predictions):
    subject_level_results = {}
    for idx, pred in zip(subject_indices, predictions):
        if idx not in subject_level_results:
            subject_level_results[idx] = []
        subject_level_results[idx].append(pred)
    # Calculate average prediction for each subject and feedback the value corresponding to the positive label 
    subject_level_avg_predictions = {idx: np.mean(np.array(preds), axis=0)[-1] for idx, preds in subject_level_results.items()}
    return subject_level_avg_predictions

def convert_labels_to_binary(labels, positive_label='MCI'):
    """
    Convert string labels to binary (0, 1).
    :param labels: Array of string labels ('HC' or 'MCI') / （'HC' or 'PD').
    :param positive_label: The label to be considered as 1 (positive class).
    :return: Array of binary labels.
    """
    binary_labels = np.array([1 if label == positive_label else 0 for label in labels])
    return binary_labels


#%% Cross-validation Data Checker
class InconsistentDatasetException(Exception):
    """Exception raised for inconsistencies in dataset across different folds."""
    def __init__(self, message="Inconsistent dataset across folds"):
        self.message = message
        super().__init__(self.message)

def check_folds_data_consistency(fold_dataname):
    """
    Checks whether all folds in cross-validation use the same dataset.
    Merges train and validation data for each fold and compares datasets across folds.

    Parameters:
    fold_dataname (dict): Dictionary containing fold data with train and test data names for each fold.

    Returns:
    list: List of all data names if all folds use the same dataset.
    
    Raises:
    InconsistentDatasetException: If not all folds use the same dataset.
    """
    all_fold_datasets = []

    for fold in fold_dataname:
        # Merge train and test data names for each fold
        current_fold_dataset = sorted(fold_dataname[fold]['train'] + fold_dataname[fold]['test'])
        all_fold_datasets.append(current_fold_dataset)

    # Check if all folds have the same dataset
    first_fold_dataset = all_fold_datasets[0]
    if all(current_fold_dataset == first_fold_dataset for current_fold_dataset in all_fold_datasets):
        return list(first_fold_dataset)
    else:
        raise InconsistentDatasetException()


#%% DATA LOADER - Shared functions

# Function to retrieve data from data_combined based on dataname in each fold 
def retrieve_data_by_name(train_data_list, eeg_data_combined):
    retrieved_data = []
    for name in train_data_list:
        for data in eeg_data_combined:
            if data[0] == name:  # data[0] is the data_name
                retrieved_data.append((data[1], data[2]))  # Adding label and eeg_data to the list
                break  # Assuming each data_name is unique, break after finding the match
    return retrieved_data


def split_fold_data(fold_idx, fold_dataname, eeg_data_combined):

    train_data_name = fold_dataname[list(fold_dataname.keys())[fold_idx]]["train"]
    val_data_name = fold_dataname[list(fold_dataname.keys())[fold_idx]]["test"]

    train_data = retrieve_data_by_name(train_data_name, eeg_data_combined)
    val_data = retrieve_data_by_name(val_data_name, eeg_data_combined)

    return train_data, val_data


def split_eeg_data(data_list, fs, segment_len):
    segmented_data = []
    segment_samples = int(fs * segment_len)  # Number of samples per segment

    for subject_idx, (label, eeg_data) in enumerate(data_list):
        num_segments = eeg_data.shape[1] // segment_samples

        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            eeg_segment = eeg_data[:, start:end]
            segmented_data.append((subject_idx, label, eeg_segment))

    return segmented_data

def align_data_to_shortest(data_list, fs, min_length):
    """
    Align all EEG data segments to the length of the shortest segment in the dataset.
    :param data_list: List of data tuples (label, eeg_data).
    :param fs: Sampling frequency.
    :return: List of aligned data tuples (subject_idx, label, eeg_data).
    """
    aligned_data = []
    for subject_idx, (label, eeg_data) in enumerate(data_list):

        start_idx = (eeg_data.shape[1] - min_length) // 2
        end_idx = start_idx + min_length
        aligned_segment = eeg_data[:, start_idx:end_idx]
        aligned_data.append((subject_idx, label, aligned_segment))

    return aligned_data

def segment_or_align_data(train_data, val_data, sfreq, segment_len, align_len):
    """
    Segments or aligns the given EEG data based on the specified segment length.

    If a segment length is provided, the function segments the EEG data into smaller, equal-sized segments.
    If the segment length is not provided (False), the function aligns the EEG data to the length of the shortest data in the dataset.

    Parameters:
    train_data (list): List of tuples containing training data.
    val_data (list): List of tuples containing validation data.
    sfreq (float): Sampling frequency of the EEG data.
    segment_len (int or False): Length of each segment in seconds, or False for alignment.

    Returns:
    tuple: Two lists containing segmented or aligned training and validation data.
    """
    if segment_len:
        # Segment data
        segmented_train_data = split_eeg_data(train_data, sfreq, segment_len)
        segmented_val_data = split_eeg_data(val_data, sfreq, segment_len)
    else:
        # Align data
        shorest_len = min(min(eeg_data.shape[1] for _, eeg_data in train_data), min(eeg_data.shape[1] for _, eeg_data in val_data)) / sfreq
        if align_len is False or align_len > shorest_len:
            align_len = shorest_len
        align_length = int(align_len * sfreq) 
        segmented_train_data = align_data_to_shortest(train_data, sfreq, align_length)
        segmented_val_data = align_data_to_shortest(val_data, sfreq, align_length)

    return segmented_train_data, segmented_val_data, align_len

    
#%% DATA LOAD functions for AD based on 'fold.txt'

def parse_fold_file(file_path):
    """
    Parses a file to extract data names for each fold in a cross-validation setup.

    Args:
    file_path (str): Path to the file containing fold data.

    Returns:
    dict: A dictionary where each key is a fold identifier (e.g., 'fold1'), and each value is another dictionary with 'train' and 'test' keys, containing lists of data names.

    This function reads a file where each line contains information about a fold, specifying the data names to be used for training and testing. The function constructs a dictionary that organizes this information for easy access and use in cross-validation.
    """
    folds = {}
    current_fold = None
    current_section = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("fold"):
                current_fold = line
                folds[current_fold] = {"train": [], "test": []}
            elif line in ["train", "test"]:
                current_section = line
            elif line.startswith("[") and line.endswith("]"):
                data_names = line[1:-1].split(", ")
                folds[current_fold][current_section].extend(data_names)
    
    return folds


def find_and_read_set_files(dataname_list, directories):
    """
    Searches for and reads .set files based on provided data names in specified directories, extracting EEG data.

    Args:
    dataname_list (list): A list of data names to be searched for.
    directories (list): A list of directories to search for the .set files.

    Returns:
    combined_data (list): A list of tuples, each containing the data name, label, and EEG data.
    sfreq (float): Sampling frequency of the EEG data.
    channel_names (list): Names of the EEG channels.

    This function iterates through the provided data names and searches for corresponding .set files within the specified directories. Upon finding a file, it reads the EEG data and appends it to a list along with the data name and label. This function is essential for preparing the data for analysis.
    """
    combined_data = []  # List to store EEG data objects

    for data_name in dataname_list:
        # Remove 'EC_' prefix and trim whitespace and single quotes, then append '.vhdr' extension
        modified_data_name = data_name.replace("EC_", "").strip().strip("'") + ".vhdr"

        for directory in directories:
            # Construct the expected folder path with the modified data name
            folder_path = os.path.join(directory, modified_data_name)
            if os.path.isdir(folder_path):
                # Search for .set file in the folder
                for file in os.listdir(folder_path):
                    if file.startswith('EC') and file.endswith('.set'):
                        file_path = os.path.join(folder_path, file)
                        try:
                            eeg_raw = mne.io.read_raw_eeglab(file_path, verbose='ERROR')
                            eeg_data = eeg_raw.get_data()
                            label = os.path.basename(directory)
                            combined_data.append((data_name, label, eeg_data))
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
    
    sfreq = eeg_raw.info['sfreq']
    channel_names = eeg_raw.info['ch_names']

    # Return the combined list of EEG data objects and EEG information
    return combined_data, sfreq, channel_names  

#%% DATA LOAD functions for PD based on '.json'

def construct_fold_dict(data_path, pattern):
    """
    Read and parse the JSON files to construct the fold_dataname dictionary

    Args:
    data_path (str): The path to the directory containing the files.
    pattern (str): Regular expression pattern to match file names.
    Returns:
    dict: A dictionary with fold data.
    """
    fold_dict = {}
    fold_dict_with_label = {}
    regex = re.compile(pattern)

    for filename in os.listdir(data_path):
        if regex.match(filename):
            with open(os.path.join(data_path, filename), 'r') as file:
                data = json.load(file)

                # Extract fold number and train/test from filename
                fold_number = re.search(r'fold(\d+)', filename).group(1)
                train_or_test = 'train' if 'Train' in filename else 'test'

                # Initialize fold in dictionary if not present
                fold_key = 'fold' + fold_number
                if fold_key not in fold_dict:
                    fold_dict[fold_key] = {'train': [], 'test': []}
                if fold_key not in fold_dict_with_label:
                    fold_dict_with_label[fold_key] = {'train': [], 'test': []}

                # Add data names to the appropriate list in the dictionary
                fold_dict[fold_key][train_or_test].extend(list(data.keys()))
                combined_list = [f"{name}-label{label}" for name, label in data.items()]
                fold_dict_with_label[fold_key][train_or_test].extend(combined_list)

    return fold_dict, fold_dict_with_label


def find_and_read_mat_files(dataname_list, data_directory):
    """
    Reads and processes .mat files based on a list of data names and labels, and retrieves EEG data from these files.

    Args:
    data_directory (str): The path to the directory containing the .mat files.
    dataname_list (list): A list of data names in the format "{data_name}.mat-label{label}".

    Returns:
    combined_data (list): A list of tuples, each containing the data name, label, and EEG data in the format (data_name, label, eeg_data).
    sfreq (int): The sampling frequency of the EEG data. Set to a default value as per the original paper, or updated if available in the .mat file.
    channel_names (list or None): The names of the EEG channels. This will be None if the channel names are not available in the .mat file.

    Raises:
    KeyError: If a specified .mat file is not found in the specified directory.
    ValueError: If the data in the .mat file is not 2-dimensional after squeezing single-dimensional entries.

    Each .mat file is expected to contain an array of EEG data. The function will check the existence of each .mat file, read its contents, and process the data to ensure it's in the format of channels x time points. If the number of channels is less than the number of time points, the data is transposed for consistency.
    """

    combined_data = []
    channel_names = None
    sfreq = 500 # According to the original paper

    for item in dataname_list:
        # Split the item into data_name and label
        data_name, label = item.split('-label')
        mat_file_name = f"{data_name}"

        # Construct the path to the .mat file
        mat_file_path = os.path.join(data_directory, mat_file_name)

        # Check if the file exists
        if os.path.exists(mat_file_path):

            # Load the .mat file
            with h5py.File(mat_file_path, 'r') as file:
                # Load the dataset and convert to numpy array
                eeg_data = np.array(file['data'])
                # Remove single-dimensional entries from the shape of the array
                eeg_data = np.squeeze(eeg_data)
                # Remove single-dimensional entries from the shape of the array
                if eeg_data.ndim != 2:
                    raise ValueError(f"Squeezed data is not 2-dimensional, shape is {eeg_data.shape}")
                # Transpose if channels are less than time points
                if eeg_data.shape[0] > eeg_data.shape[1]:
                    eeg_data = eeg_data.T

            # Rename Label (consistent with AD data type)
            label = 'PD' if int(label) == 1 else 'HC'

            # Append to the combined data list
            combined_data.append((data_name, label, eeg_data))
        else:
            raise KeyError(f"Dataset '{mat_file_name}' not found in file {data_directory}")
        
    return combined_data, sfreq, channel_names

#%% DATA LOADER  
 
def load_data(project_path, data_tag):
    
    if data_tag == 'AD':

        data_path = os.path.join(project_path, 'AD_CITY1')
        data_division = os.path.join(data_path, 'fold.txt')
        fold_dataname = parse_fold_file(data_division)

        dataname_list = check_folds_data_consistency(fold_dataname)

        hc_directory = os.path.join(data_path, 'HC')
        mci_directory = os.path.join(data_path, 'MCI')
        eeg_data_combined, sfreq, channel_names = find_and_read_set_files(dataname_list, [hc_directory, mci_directory])

    elif data_tag == 'PD':

        data_path = os.path.join(project_path, 'PD_CITY1')
        pattern = r'(Train|Test)fold\d+\.json'
        fold_dataname, fold_dataname_with_label = construct_fold_dict(data_path, pattern)

        dataname_list = check_folds_data_consistency(fold_dataname_with_label)

        data_directory = os.path.join(data_path, 'UNMDataset_OFF','lw_data_v1')
        eeg_data_combined, sfreq, channel_names = find_and_read_mat_files(dataname_list, data_directory)

    return eeg_data_combined, fold_dataname, sfreq, channel_names

#%% Feature Engineering Functions

## Normalization 
def flatten_data(data_list):
    """ Flatten each data segment and combine into a single array. """
    flattened_data = np.array([segment.flatten() for _, _, segment in data_list])
    return flattened_data

def normalize_data(data_list, mean, std):
    """ Apply z-score normalization to each data segment. """
    normalized_data = []
    for subject_idx, label, segment in data_list:
        normalized_segment = (segment.flatten() - mean) / std
        normalized_data.append((subject_idx, label, normalized_segment.reshape(segment.shape)))
    return normalized_data

def clean_data(data):
    data = np.array(data)
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    return data

## Method 1: Frequency-Domain Feature Engineering as Employed in the Original Study of the Public PD Dataset

"""
James F. Cavanagh, Praveen Kumar, Andrea A. Mueller, Sarah Pirio Richardson, Abdullah Mueen,
Diminished EEG habituation to novel events effectively classifies Parkinson’s patients,
Clinical Neurophysiology,
Volume 129, Issue 2,
2018,
Pages 409-418,
ISSN 1388-2457,
https://doi.org/10.1016/j.clinph.2017.11.023
"""

# def 
#     psd.welch(fmin=0, fmax=40)


def extract_fft_coefficients(windowed_data, sfreq, num_coefficients=50):
    """
    Apply FFT to EEG data and extract first 'num_coefficients' coefficients.
    :param eeg_data: EEG data (2D numpy array; channels x time points).
    :param sfreq: Sampling frequency of the EEG data.
    :param num_coefficients: Number of FFT coefficients to extract.
    :return: Extracted FFT coefficients.
    """
    fft_result = np.fft.fft(windowed_data, axis=1)

    # Extract the first 'num_coefficients' coefficients
    fft_coefficients = fft_result[:, :num_coefficients]
    
    return np.abs(fft_coefficients)

def linearize_fft_coefficients(fft_coeffs):
    """
    Linearize the FFT coefficients from all channels into a single vector.
    :param fft_coeffs: FFT coefficients (2D numpy array; channels x coefficients).
    :return: Linearized vector of FFT coefficients.
    """
    return fft_coeffs.flatten()

## Method 2: Complexity-Domain Feature Engineering as Employed in the Lastest Study using the Public PD Dataset

"""
I. Suuronen, A. Airola, T. Pahikkala, M. Murtojärvi, V. Kaasinen and H. Railo, 
Budget-Based Classification of Parkinson's Disease From Resting State EEG,
IEEE Journal of Biomedical and Health Informatics,
vol. 27, no. 8, pp. 3740-3747, 
Aug. 2023
doi: 10.1109/JBHI.2023.3235040.
"""


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Apply a Butterworth band-pass filter to EEG data.
    :param data: EEG data (channels x samples).
    :param lowcut: Low cutoff frequency.
    :param highcut: High cutoff frequency.
    :param fs: Sampling frequency of the EEG data.
    :param order: Order of the Butterworth filter.
    :return: Band-pass filtered data.
    """
    data = clean_data(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    return filtered_data

def apply_band_filters_to_eeg(eeg_data, fs):
    """
    Apply band filters to EEG data to extract frequency components.
    :param eeg_data: EEG data (channels x samples).
    :param fs: Sampling frequency of the EEG data.
    :return: Dictionary of frequency components.
    """
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'low_alpha': (8.0, 10.0),
        'high_alpha': (10.0, 13.0),
        'beta': (13.0, 30.0)
    }

    filtered_bands = {}
    for band, (low, high) in bands.items():
        filtered_bands[band] = butter_bandpass_filter(eeg_data, low, high, fs)
    
    return filtered_bands


def chebyshev_distance(x, y):
    """ Calculate the Chebyshev distance between two vectors. """
    return np.max(np.abs(x - y))

def sample_entropy_chebyshev(signal, m, r):
    """
    Calculate sample entropy using Chebyshev distance.
    :param signal: The input signal (time series).
    :param m: Embedding dimension (length of sequences to compare).
    :param r: Tolerance for similarity.
    :return: Sample Entropy value.
    """
    N = len(signal)
    B = 0.0
    A = 0.0

    # Create subsequences of length m
    for i in range(N - m):
        for j in range(i + 1, N - m):
            if chebyshev_distance(signal[i:i + m], signal[j:j + m]) < r:
                B += 1
                if chebyshev_distance(signal[i:i + m + 1], signal[j:j + m + 1]) < r:
                    A += 1

    # Calculate SampEn using Chebyshev distance
    return -np.log(A / B) if B != 0 else np.inf

def sample_entropy_euclidean(signal, m, r):
    """
    Calculate sample entropy using pyentrp.
    :param signal: The input signal (time series).
    :param m: Embedding dimension (length of sequences to compare).
    :param r: Tolerance for similarity.
    :return: Sample Entropy value.
    """
    return ent.sample_entropy(signal, m, r)

def calculate_sampen_for_all_bands(filtered_bands, m=2, r_factor=0.2):
    """
    Calculate sample entropy for all frequency bands using Chebyshev distance.
    :param filtered_bands: Dictionary containing band-pass filtered data for each band.
    :param m: Embedding dimension.
    :param r_factor: Factor for tolerance calculation.
    :return: Dictionary containing SampEn for each band.
    """
    # start = time.time()
    sampen_bands = {}
    for band in filtered_bands:
        sampen_values = []
        for channel in filtered_bands[band]:
            channel = clean_data(channel)
            std_dev = np.std(channel)
            r = r_factor * std_dev  # Tolerance based on standard deviation
            # sampen = sample_entropy_chebyshev(channel, m, r)
            sampen = sample_entropy_euclidean(channel, m, r)
            sampen_values.append(sampen[0] if sampen.size > 1 else sampen)
        sampen_bands[band] = sampen_values

    # print(f"{time.time()}-calculate_sampen_for_all_bands in {time.time() - start:.2f}seconds.")

    return sampen_bands

def concatenate_sampen_features(sampen_bands):
    """
    Concatenate SampEn features from all frequency bands into a single vector.
    :param sampen_bands: Dictionary with SampEn values for each band and channel.
    :return: Concatenated feature vector.
    """
    feature_vector = []
    # Iterate through each band and extend the feature vector with SampEn values
    for band in sampen_bands:
        feature_vector.extend(sampen_bands[band])
    return np.array(feature_vector)

## Method 3: Network-Domain Feature Engineering 
# -- Related to the existing paper（CORR）
# -- Related to the Manifold Mapping described in our article（SPD）

def extract_upper_triangle_to_vector(matrix):
    """
    Extract the upper triangular part of a matrix and convert it into a vector.

    Args:
    matrix (numpy.ndarray): Correlation matrix.

    Returns:
    numpy.ndarray: Vectorized upper triangular part of the matrix.
    """
    upper_triangle_indices = np.triu_indices_from(matrix, k=1)
    return matrix[upper_triangle_indices]

def compute_corr_matrix(x):
    """
    Compute the correlation matrix of the EEG signal data.

    Args:
    x (numpy.ndarray): Input EEG signal data, shape (channel_dim, time_points).

    Returns:
    numpy.ndarray: Correlation matrix of the EEG signal.
    """
    # NumPy corrcoef returns the correlation matrix
    correlation_matrix = np.corrcoef(x)
    return correlation_matrix


def compute_spd_matrix(x):
    """
    Compute the Symmetric Positive Definite (SPD) matrix from EEG signal data using PyTorch for acceleration.

    Args:
    x (numpy.ndarray): Input EEG signal data, shape (channel_dim, time_points).

    Returns:
    torch.Tensor: SPD matrix of the EEG signal.
    """
    # Convert to PyTorch tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # Subtract the mean
    mean = x_tensor.mean(dim=1, keepdim=True)
    x_centered = x_tensor - mean

    # Compute the covariance matrix
    cov_matrix = x_centered @ x_centered.t() / (x_tensor.shape[1] - 1)

    # Regularization to ensure the matrix is positive definite
    regularization = 1e-5 * torch.eye(cov_matrix.size(0))
    spd_matrix = cov_matrix + regularization

    return spd_matrix.numpy()


def compute_coupling_matrix(eeg,mode='valid', normalize=True):
    'SAME AS CORR RESULTS'
    N_chs, _ = eeg.shape
    graph = np.ones((N_chs,N_chs))
    for ch_i in range(N_chs-1):
        for ch_j in range(ch_i+1,N_chs):
                x = eeg[ch_i]
                y = eeg[ch_j]
                xcorr = np.correlate(x, y, mode='valid')
                if normalize:
                    # the below normalization code refers to matlab xcorr function
                    cxx0 = np.sum(np.absolute(x)**2)
                    cyy0 = np.sum(np.absolute(y)**2)
                    if (cxx0 != 0) and (cyy0 != 0):
                        scale = (cxx0 * cyy0) ** 0.5
                        xcorr /= scale
                graph[ch_i, ch_j] = xcorr
    i_lower = np.tril_indices(N_chs, -1)
    graph[i_lower] = graph.T[i_lower]
    return graph
#%% Data Process

def process_data(data, feature_tag, sfreq):    
    """
    Process a single data.

    Args:
    data (numpy.ndarray): Input EEG signal data.
    feature_tag (str): Tag indicating the type of feature to be extracted ('frequency', 'complexity', or 'network').
    sfreq (float): Sampling frequency of the EEG data.

    Returns:
    numpy.ndarray: Extracted features from the input data.
    """
    # print('multiprocessing!')
    if feature_tag == 'frequency':
        # Feature 1: frequency-domain
        features = linearize_fft_coefficients(extract_fft_coefficients(data, sfreq))
    elif feature_tag == 'complexity':
        # Feature 2: complexity-domain
        features = concatenate_sampen_features(calculate_sampen_for_all_bands(apply_band_filters_to_eeg(data, sfreq)))
    elif feature_tag == 'network_corr':
        # Feature 3: network-domain_correlation matrix
        features = extract_upper_triangle_to_vector(compute_corr_matrix(data))
    elif feature_tag == 'network_spd':
        # Feature 3: network-domain_spd matrix
        features = extract_upper_triangle_to_vector(compute_spd_matrix(data)) 
    elif feature_tag == 'network_coupling':
        # Feature 4: network-domain_coupling_matrix matrix
        features = extract_upper_triangle_to_vector(compute_coupling_matrix(data)) 
    
    return features

   
def parallel_feature_extraction(data_list, feature_tag, sfreq, pool, manager):
    """
    Parallel feature extraction using Manager for shared memory.
    """
    with manager.Pool() as pool:
        results = []
        for data in data_list:
            result = pool.apply_async(process_data, args=(data, feature_tag, sfreq))
            results.append(result)
        
        return [result.get() for result in results]
    
#%% Results Print
def print_results(segment_level_metrics, subject_level_metrics):
    print('-'*10)
    print("Results per Fold:\n")

    for fold_idx in range(len(segment_level_metrics)):
        segment_metrics = segment_level_metrics[fold_idx]
        subject_metrics = subject_level_metrics[fold_idx]

        print(f"-------------\nFold {fold_idx + 1}")
        print("Segment Level - Accuracy, Precision, Recall, F1-score")
        print(f"{segment_metrics['accuracy']*100:.2f}%, {segment_metrics['precision']*100:.2f}%, "
            f"{segment_metrics['recall']*100:.2f}%, {segment_metrics['f1_score']*100:.2f}%")
        
        print("Subject Level - Accuracy, Precision, Recall, F1-score")
        print(f"{subject_metrics['accuracy']*100:.2f}%, {subject_metrics['precision']*100:.2f}%, "
            f"{subject_metrics['recall']*100:.2f}%, {subject_metrics['f1_score']*100:.2f}%")

    # Calculate and print mean and variance for segment and subject level metrics
    print("\nMean and Variance of Metrics Across All Folds:")
    print("\Segment Level")
    for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
        segment_mean, segment_variance = calculate_mean_variance(segment_level_metrics, metric_name)
        print(f"{metric_name.capitalize()} - Segment Level: Mean = {segment_mean*100:.2f}%, Variance = {segment_variance*100:.2f}%")
    print("\Subject Level")
    for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
        subject_mean, subject_variance = calculate_mean_variance(subject_level_metrics, metric_name)
        print(f"{metric_name.capitalize()} - Subject Level: Mean = {subject_mean*100:.2f}%, Variance = {subject_variance*100:.2f}%")


#%% Main Function
def main():

    ## USER OPTIONS
    project_path= '../EEG_MACAC_ON_AD_PD/'
    phase = 'train'
    data_tag = 'AD' # 'PD' \ 'AD'
    feature_tag = 'network_spd' # 'frequency' \ 'complexity' \ 'network_corr' \ 'network_spd' \ 'network_coupling'
    norm_tag = 1  # 0: False ,1: train_mean/std, 2: train_mean/std + val_mean/std
    ensemble_tag = 'prob' # 'prob' \ 'label'
    thred_tag = 'fix' # 'opt' \ 'fix'

    # Check User Option Correctness before Starting the Program
    allowed_data_tags = ['AD', 'PD']
    allowed_feature_tags = ['frequency', 'complexity', 'network_corr', 'network_spd', 'network_coupling']
    allowed_ensemble_tags = ['prob', 'label']
    allowed_thred_tags = ['fix', 'opt']
    allowed_norm_tags = [0,1,2]
    if data_tag not in allowed_data_tags:
        raise ValueError(f"Invalid data_tag: '{data_tag}'. Allowed values are {allowed_data_tags}.")
    if norm_tag not in allowed_norm_tags:
        raise ValueError(f"Invalid norm_tag: '{norm_tag}'. Allowed values are {allowed_norm_tags}.")
    if feature_tag not in allowed_feature_tags:
        raise ValueError(f"Invalid feature_tag: '{feature_tag}'. Allowed values are {allowed_feature_tags}.")
    if ensemble_tag not in allowed_ensemble_tags:
        raise ValueError(f"Invalid ensemble_tag: '{ensemble_tag}'. Allowed values are {allowed_ensemble_tags}.")
    if thred_tag not in allowed_thred_tags:
        raise ValueError(f"Invalid thred_tag: '{thred_tag}'. Allowed values are {allowed_thred_tags}.")
    
    if data_tag == 'AD':
    
        fold_num = 4
        segment_len = 8 # False / 8
        align_len = 8
        positive_label = 'MCI'
        
    elif data_tag == 'PD':
        
        fold_num = 3
        segment_len = 2 # False / 2
        align_len = False
        positive_label = 'PD'
    
    align_len = False if segment_len else align_len

    ## START

    # Create a directory for saved models if it doesn't exist
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    # Create a directory for saved results if it doesn't exist
    if not os.path.exists('saved_results'):
        os.makedirs('saved_results')

    
    # Generate log file name with parameters and current datetime
    start_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file_name = f"log_{data_tag}_{feature_tag}_seg{segment_len}_align{align_len}_\
        norm{norm_tag}_{ensemble_tag}_{thred_tag}_{start_timestamp}.txt"
    log_file_path = os.path.join('saved_results', log_file_name)

    
    # Save the current standard output
    original_stdout = sys.stdout

    # Open the log file and redirect standard output to it
    with open(log_file_path, 'w') as log_file:

        sys.stdout = log_file

        start_time = time.time()

        # LOAD DATA
        eeg_data_combined, fold_data, sfreq, channel_names = load_data(project_path, data_tag)
        print(f"Data Tag: {data_tag}")
        print(f'Sampling Frequency: {sfreq} Hz')
        if channel_names:
            channel_num = len(channel_names)
        else:
            channel_num, _ = eeg_data_combined[0][2].shape # eeg_data_combined : A list of (data_name, label, eeg_data)
        print(f'Channels: {channel_num} - {channel_names}')
        print('Total Data Subjects:', len(eeg_data_combined))
        
        end_time_load = time.time()
        print(f"Data loaded in {end_time_load - start_time:.2f}seconds.")

        segment_level_metrics = []
        subject_level_metrics = []

        for fold_idx in range(fold_num):
            print('-'*10)
            print("{}-domian, Fold{}".format(feature_tag, fold_idx))

            start_time_process = time.time()

            # SPLIT N-Fold DATA
            train_data, val_data = split_fold_data(fold_idx, fold_data, eeg_data_combined)

            # SEGMENT DATA
            segmented_train_data, segmented_val_data, align_len = segment_or_align_data(train_data, val_data, sfreq, segment_len, align_len)
            print(f"Segment Length: {segment_len} seconds")
            print(f"Aligned Length: {align_len} seconds")
            
            # NORMALIZATION
            # Compute mean and standard deviation for z-score normalization
            if norm_tag == 0:
                normalized_train_data = segmented_train_data
                normalized_val_data = segmented_val_data
            else:
                flattened_train_data = flatten_data(segmented_train_data) # Flatten the training data
                train_mean = np.mean(flattened_train_data, axis=0)
                train_std = np.std(flattened_train_data, axis=0)
                normalized_train_data = normalize_data(segmented_train_data, train_mean, train_std)
                if norm_tag == 1:
                    normalized_val_data = normalize_data(segmented_val_data, train_mean, train_std)
                elif norm_tag == 2:
                    flattened_val_data = flatten_data(segmented_val_data) # Flatten the validation data
                    val_mean = np.mean(flattened_val_data, axis=0)
                    val_std = np.std(flattened_val_data, axis=0)
                    normalized_val_data = normalize_data(segmented_val_data, val_mean, val_std)

            end_time_preprocess = time.time()
            print(f"Fold {fold_idx} pre-processed in {end_time_preprocess - start_time_process} seconds.")

            # LABEL
            train_labels = convert_labels_to_binary(np.array([label for _, label, _ in normalized_train_data]), 
                                                    positive_label=positive_label)
            val_labels = convert_labels_to_binary(np.array([label for _, label, _ in normalized_val_data]), 
                                                  positive_label=positive_label)

            # FEATURE EXTRACTION

            # Create data list for parallel processing
            train_data_list = [data for _, _, data in normalized_train_data]
            val_data_list = [data for _, _, data in normalized_val_data]
            print('Train data samples: {} \nTest data samples: {}'.format(len(train_data_list),len(val_data_list)))

            # Parallel feature extraction for M EEG Segments
            manager = Manager()
            pool = Pool(processes=multiprocessing.cpu_count()-2)
            train_features = parallel_feature_extraction(train_data_list, feature_tag, sfreq, pool, manager)
            val_features = parallel_feature_extraction(val_data_list, feature_tag, sfreq, pool, manager)
            pool.close()
            pool.join()
            print(f'Feature Dim: {train_features[0].shape}')
            print(f"Fold {fold_idx} feature extracted in {time.time() - end_time_preprocess} seconds.")

            # CLASSIFIER
            svm_classifier = SVC(kernel='linear', random_state=42, probability=True)

            if phase == 'train':
                svm_classifier.fit(train_features, train_labels)
                # Save the trained model
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                model_name_base = log_file_name.replace("log_", "")
                model_name_parts = model_name_base.rsplit('_', 1)[0]
                model_name = f"{model_name_parts}_fold{fold_idx}_{timestamp}.pkl"
                save_model(svm_classifier, model_name)
                
            elif phase == 'test':
                model_name_parttern = f"{data_tag}_{feature_tag}_seg{segment_len}_align{align_len}_\
                    norm{norm_tag}_{ensemble_tag}_fold{fold_idx}_*.pkl"
                svm_classifier = load_latest_model(model_name_parttern)
                if svm_classifier is None:
                    raise Exception("No trained model found for testing.")

            # PREDICTION & PERFORMANCE
                
            # Segment-level predictions on the validation set
            if ensemble_tag == 'prob':
                val_segment_proba = svm_classifier.predict_proba(val_features) # step1: probability
                val_segment_predictions = np.argmax(val_segment_proba, axis=1) # step2: label
            elif ensemble_tag == 'label':
                val_segment_predictions = svm_classifier.predict(val_features) # label

            # Segment-Level Metrics
            segment_metrics = calculate_metrics(val_labels, val_segment_predictions)

            # Subject-level predictions on the validation set
            val_subject_idx = np.array([subject_idx for subject_idx, _, _ in normalized_val_data])
            val_labels_one_hot = np.eye(2)[val_labels]
            val_subject_labels = aggregate_subject_level_predictions(val_subject_idx, val_labels_one_hot)
            val_subject_predictions = aggregate_subject_level_predictions(val_subject_idx, val_segment_proba)
            print(f"Pred {val_subject_predictions}")
            print(f"True {val_subject_labels}")
            if thred_tag == 'fixp':
                val_subject_binary_predictions = binary_threshold(list(val_subject_predictions.values()))
            elif thred_tag == 'opt':
                val_subject_binary_predictions = optimal_threshold(list(val_subject_labels.values()), 
                                                                   list(val_subject_predictions.values()))
            
            # Subject-Level Metrics
            subject_metrics = calculate_metrics(list(val_subject_labels.values()), val_subject_binary_predictions)
            
            # Store N-fold Metrics
            segment_level_metrics.append(segment_metrics)
            subject_level_metrics.append(subject_metrics)

        print('')
        print(f"All folds processed in {time.time() - end_time_load:.2f} seconds.")

        # Show Results
        print_results(segment_level_metrics, subject_level_metrics)

    # Restore standard output to its original stat
    sys.stdout = original_stdout

    # Rename log file
    end_timestamp = datetime.now().strftime("%H%M%S")
    org_log_file_name, extension = log_file_name.rsplit('.',1)
    new_log_file_name = f"{org_log_file_name}-{end_timestamp}.{extension}"
    new_log_file_path = os.path.join('saved_results', new_log_file_name)

    # Attempt to rename the log file
    try:
        # Ensure the original log file is closed if it's open in your program
        os.rename(log_file_path, new_log_file_path)
        print(f"Log file successfully renamed to {new_log_file_name}")
    except OSError as e:
        print(f"Error renaming log file: {e}")
        os.rename(log_file_path, new_log_file_path)

#%% Runing

if __name__ == '__main__':
    main()