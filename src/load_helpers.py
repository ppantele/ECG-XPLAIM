import pandas as pd
import wfdb


# Load labels and metadata, and save as .csv.
def metadata_extract_save(ds_name, data_input_dir, metadata_output_dir, label_kwds):
    '''
    Extracts metadata and diagnostic labels from a dataset (MIMIC-IV or PTB-XL) and saves the processed results as a CSV file.

    Args:
        ds_name (str): Name of the dataset ('mimic-iv' or 'ptb-xl').
        data_input_dir (str): Directory containing the dataset's raw metadata files.
        metadata_output_dir (str): Directory to save the processed metadata CSV.
        label_kwds (dict): Dictionary mapping label names to lists of associated keywords.

    Returns:
        None: Writes a new CSV file to disk with label annotations.
    '''
    
    # Load the .csv files into DataFrames
    if ds_name == 'mimic-iv':
        record_list = pd.read_csv(data_input_dir + 'record_list.csv', sep=',')
        machine_measurements = pd.read_csv(data_input_dir + 'machine_measurements.csv', sep=',')
        record_df = pd.merge(record_list, machine_measurements, on=['study_id', 'subject_id'])
    elif ds_name == 'ptb-xl':
        record_df = pd.read_csv(data_input_dir + 'ptbxl_database.csv', sep=',')

    # Function to check for multiple diagnostic labels
    def detect_labels(row):
        if ds_name == 'mimic-iv':
            # Combine all report columns into one text block
            report_text = ' '.join(str(row[f'report_{i}']) for i in range(18)).lower()  # 18: number of cols with reports in .csv file
        elif ds_name == 'ptb-xl':
            report_text = row['scp_codes']
        report_text = f" {report_text} "  # Add spaces to the beginning and end of the text
        # Initialize a dictionary to store label detections
        detected_labels = {label: 0 for label in label_kwds.keys()}
        # Check for each label's keywords in the report text
        for label, keywords in label_kwds.items():
            if any(keyword in report_text for keyword in keywords):
                detected_labels[label] = 1
        return detected_labels

    # Apply the function to detect all labels and expand into separate columns
    label_detections = record_df.apply(detect_labels, axis=1)
    label_df = pd.DataFrame(label_detections.tolist())

    # Add the detected label columns to the merged DataFrame
    record_df = pd.concat([record_df, label_df], axis=1)

    # Select only relevant columns and label columns
    if ds_name == 'mimic-iv':
        selected_columns = ['subject_id', 'study_id', 'path'] + list(label_kwds.keys())
    elif ds_name == 'ptb-xl':
        selected_columns = ['patient_id', 'ecg_id', 'filename_hr'] + list(label_kwds.keys())
    output_df = record_df[selected_columns]

    # Save the combined and filtered data to a new .csv file
    output_file = metadata_output_dir + ds_name + '_labels_metadata.csv'
    output_df.to_csv(output_file, index=False)

    print(f'New CSV file created: {output_file}')



def id_to_path(ecg_id, ds_name, metadata_dir):
    '''
    Maps an ECG ID (study_id or ecg_id) to the corresponding file path using the dataset's metadata CSV.

    Args:
        ecg_id (int): Unique identifier of the ECG (study_id or ecg_id).
        ds_name (str): Dataset name ('mimic-iv' or 'ptb-xl').
        metadata_dir (str): Directory containing the processed metadata CSV.

    Returns:
        str: Relative file path to the ECG recording.
    '''
    
    metadata = pd.read_csv(metadata_dir + ds_name + '_labels_metadata.csv')
    if ds_name == 'mimic-iv':
        id_to_relevant_path_map = dict(zip(metadata['study_id'], metadata['path']))
    elif ds_name == 'ptb-xl':
        id_to_relevant_path_map = dict(zip(metadata['ecg_id'], metadata['filename_hr']))
    return id_to_relevant_path_map[ecg_id]



def load_and_preprocess_signal(file_id_or_path, ds_name, data_input_dir, metadata_dir=None, preprocess_funcs={}):
    '''
    Loads a single ECG signal from file and applies optional preprocessing steps.

    Args:
        file_id_or_path (int or str): ECG ID (int) or relative file path (str).
        ds_name (str): Dataset name ('mimic-iv' or 'ptb-xl').
        data_input_dir (str): Directory containing raw WFDB files.
        metadata_dir (str, optional): Path to the metadata CSV (used if ECG ID is given).
        preprocess_funcs (dict): Dictionary of preprocessing functions and their arguments, e.g. {func1: [arg1], func2: [arg2, arg3]}.

    Returns:
        tuple:
            np.ndarray: ECG signal with shape (5000, 12).
            float: Sampling frequency of the signal.
    '''

    if type(file_id_or_path) == int:
        relevant_file_path = id_to_path(file_id_or_path, ds_name=ds_name, metadata_dir=metadata_dir)
    else:
        relevant_file_path = file_id_or_path
    
    # Read the signal using wfdb
    record = wfdb.rdrecord(data_input_dir + relevant_file_path)
    signal = record.p_signal  # Signal dims: [5000, 12], ie. [length, leads]
    fs = record.fs

    # If dataset is 'ptb-xl', flip leads (columns) 4 and 5 (aVF, aVL)
    if ds_name == 'ptb-xl' and signal.shape[1] > 5:  # Ensure there are at least 6 leads
        signal[:, [4, 5]] = signal[:, [5, 4]]  # Swap columns 4 and 5

    # Pre-process the signal
    for func in preprocess_funcs.items():
        signal = func[0](signal, *func[1])

    return signal, fs
