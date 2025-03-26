import tensorflow as tf
import numpy as np
import pandas as pd
from . import load_helpers as lh
from . import signal_preprocess as sig_prep


def shuf_bat_rep_pref(dataset, shuffle_buffer=None, batch_size=None, n_batches=None, repeat=False, prefetch=True):
    '''
    Applies common TensorFlow dataset transformations: shuffle, batch, take, repeat, and prefetch.

    Args:
        dataset (tf.data.Dataset): The input dataset.
        shuffle_buffer (int, optional): Size of the shuffle buffer.
        batch_size (int, optional): Batch size to apply.
        n_batches (int, optional): Number of batches to take.
        repeat (bool): Whether to repeat the dataset.
        prefetch (bool): Whether to enable data prefetching.

    Returns:
        tf.data.Dataset: Transformed dataset.
    '''
    
    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    if n_batches is not None:
        dataset = dataset.take(n_batches)
    if repeat:
        dataset = dataset.repeat()
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



def tf_bal_dataset(
    ds_name,
    data_input_dir,
    metadata_dir,
    batch_size,
    n_samples_per_label,
    shuffle_buffer=None, n_batches=None, repeat=False,
    preprocess_funcs={sig_prep.replace_nan: [0]}):
    '''
    Creates a balanced TensorFlow dataset by sampling an equal number of samples per label.

    Args:
        ds_name (str): Dataset name ('ptb-xl' or 'mimic-iv').
        data_input_dir (str): Path to input signal files.
        metadata_dir (str): Path to metadata CSV.
        batch_size (int): Batch size. batch_size*(train + val + test batches) must be <= total number of samples.
        n_samples_per_label (dict): Number of samples to draw per label, e.g., {'lqt': 1000, 'neg': 1000}.
        shuffle_buffer (int, optional): Shuffle buffer size.
        n_batches (int, optional): Number of batches to take.
        repeat (bool): Whether to repeat dataset.
        preprocess_funcs (dict): Dictionary of preprocessing functions to apply.

    Returns:
        tf.data.Dataset: A balanced, batched, preprocessed dataset.
    '''

    # Load metadata
    metadata_file = metadata_dir + ds_name + '_labels_metadata.csv'
    labels_metadata = pd.read_csv(metadata_file)

    # Prepare label list and file paths
    label_list = list(n_samples_per_label.keys())
    if 'neg' in label_list:
        label_list.remove('neg')  # Exclude 'neg' from label-specific processing

    if ds_name == 'mimic-iv':
        all_file_paths = np.array(labels_metadata['path'])
    elif ds_name == 'ptb-xl':
        all_file_paths = np.array(labels_metadata['filename_hr'])

    # Convert labels to a NumPy array for fast processing
    label_data = labels_metadata[label_list].values

    # Separate indices for each label
    label_indices = {label: np.where(label_data[:, i] == 1)[0] for i, label in enumerate(label_list)}
    neg_indices = np.where(label_data.sum(axis=1) == 0)[0]
    label_indices['neg'] = neg_indices

    # Aggregate indices for all labels, SHUFFLE, and remove duplicates
    aggregated_indices = []
    for label, n_samples in n_samples_per_label.items():
        selected_indices = np.random.choice(label_indices[label], n_samples, replace=False)
        aggregated_indices.extend(selected_indices)
    
    # Remove duplicates and Shuffle
    aggregated_indices = list(set(aggregated_indices))  # Remove duplicates
    np.random.shuffle(aggregated_indices)               # Shuffle indices

    # Signal and label shapes
    signal_shape = lh.load_and_preprocess_signal(all_file_paths[0], ds_name=ds_name, data_input_dir=data_input_dir, preprocess_funcs=preprocess_funcs)[0].shape
    label_shape = (len(label_list), )

    def signal_generator():
        for idx in aggregated_indices:
            file_path = all_file_paths[idx]
            signal = lh.load_and_preprocess_signal(file_path, ds_name=ds_name, data_input_dir=data_input_dir, preprocess_funcs=preprocess_funcs)[0]
            label_values = np.zeros(label_shape, dtype=np.int32)
            
            # Assign label
            row_labels = label_data[idx]
            for i, value in enumerate(row_labels):
                if value == 1:
                    label_values[i] = 1  # Set label value for corresponding index
            
            yield signal, label_values

    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        signal_generator,
        output_signature=(
            tf.TensorSpec(shape=signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=label_shape, dtype=tf.int32)
        )
    )

    # Shuffle, batch, repeat, and prefetch
    dataset = shuf_bat_rep_pref(
        dataset,
        shuffle_buffer=shuffle_buffer,
        batch_size=batch_size,
        n_batches=n_batches,
        repeat=repeat
    )

    return dataset



def tf_input_shape(dataset):
    '''
    Infers the input and label shapes from a TensorFlow dataset.

    Args:
        dataset (tf.data.Dataset): A batched dataset.

    Returns:
        tuple: (input_shape, label_shape), both excluding batch dimension.
    '''

    for signal, label in dataset.take(1):
        input_shape = signal.shape[1:]  # Exclude batch dimension
        label_shape = label.shape[1:]
        return input_shape, label_shape



def tf_subset(dataset, n_batches, repeat=True):
    '''
    Creates a subset of a dataset with a limited number of batches.

    Args:
        dataset (tf.data.Dataset): The original dataset.
        n_batches (int): Number of batches to take.
        repeat (bool): Whether to repeat the subset.

    Returns:
        tf.data.Dataset: Subset of the dataset.
    '''
    
    dataset = shuf_bat_rep_pref(dataset, n_batches=n_batches, repeat=repeat)
    return dataset



def tf_train_val_test_sets(
    dataset, 
    n_train, n_val, n_test,
    train_repeat=True, 
    val_repeat=False,
    test_repeat=False):
    '''
    Splits a dataset into training, validation, and test sets (by batch count).

    Args:
        dataset (tf.data.Dataset): The dataset to split.
        n_train (int): Number of training batches.
        n_val (int): Number of validation batches.
        n_test (int): Number of test batches.
        train_repeat (bool): Whether to repeat training set.
        val_repeat (bool): Whether to repeat validation set.
        test_repeat (bool): Whether to repeat test set.

    Returns:
        tuple: (train_set, val_set, test_set) as tf.data.Dataset objects.
    '''
    
    val_set = dataset.take(n_val)  # Start with val_set, so that the validation in each epoch doesn't have to wait for n_train to be skipped.
    train_set = dataset.skip(n_val).take(n_train)
    test_set = dataset.skip(n_train + n_val).take(n_test)  # However, test set in the end, and will have to wait for n_train + n_val to be obtained - but no prob, this will happen once.

    train_set = shuf_bat_rep_pref(train_set, repeat=train_repeat)
    val_set = shuf_bat_rep_pref(val_set, repeat=val_repeat)
    test_set = shuf_bat_rep_pref(test_set, repeat=test_repeat)

    return train_set, val_set, test_set



def tf_dataset_to_numpy(dataset, data_switch=True, labels_switch=True):
    '''
    Converts a TensorFlow dataset to NumPy arrays for samples and labels.

    Args:
        dataset (tf.data.Dataset): The input dataset.
        data_switch (bool): Whether to return sample data.
        labels_switch (bool): Whether to return labels.

    Returns:
        tuple or np.ndarray:
            - If both switches are True: (samples_array, labels_array)
            - If one is True: only the respective array
    '''
    
    data_list = []
    labels_list = []

    batch_counter = 0
    for data, labels in dataset:
        if data_switch:
            data_list.append(data.numpy())
        if labels_switch:
            labels_list.append(labels.numpy())
        batch_counter += 1
        if batch_counter % 10 == 0:
            print(f'{batch_counter} batches completed.')

    # Combine all batches into single NumPy arrays
    if (data_switch and labels_switch):
        data_np = np.concatenate(data_list, axis=0)
        labels_np = np.concatenate(labels_list, axis=0)
        return (data_np, labels_np)
    elif data_switch:
        data_np = np.concatenate(data_list, axis=0)
        return data_np
    elif labels_switch:
        labels_np = np.concatenate(labels_list, axis=0)
        return labels_np
    else:
        return None