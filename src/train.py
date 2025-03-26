import numpy as np
import datetime
import argparse
from sys import getsizeof
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from . import manipulate_dataset as md
from . import models
from . import load_helpers as lh
from tensorflow.config import list_physical_devices
print('GPUs Available: ', list_physical_devices('GPU'))  # Verify GPU use


def tf_model_train(
    model_name, label_dict,
    train_ds_name, train_ds_dir, metadata_dir,
    train_batches, val_batches, test_batches,
    n_epochs, batch_size, models_output_dir,
    model_generator=None,
    tensorboard_update_freq=5):
    '''
    Train a deep learning model on a balanced ECG dataset using TensorFlow/Keras.
    This function prepares a balanced dataset, builds the model using a specified
    generator (default: ECG-XPLAIM), trains it with checkpoints and TensorBoard logging,
    and saves both the final, trained model and separate test set for evaluation.

    Args:
        model_name (str): Name for the model and output directory tag.
        label_dict (dict): Dictionary with label names and number of samples per label. Example: {'lqt': 1000, 'neg': 1000}
        train_ds_name (str): Dataset identifier (adjusted for 'mimic-iv' or 'ptb-xl').
        train_ds_dir (str): Path to the dataset directory.
        metadata_dir (str): Path to the directory of .csv files with metadata, including labels.
        train_batches (int): Number of training batches (per epoch).
        val_batches (int): Number of validation batches.
        test_batches (int): Number of test batches (excluded from training).
        n_epochs (int): Number of training epochs.
        batch_size (int): Batch size used during training. batch_size*(train + val + test batches) must be <= total number of subset samples. 
        models_output_dir (str): Directory where models and logs will be saved.
        model_generator (object, optional): Model generator class with '.create_model()' method. If None, it defaults to ECG_XPLAIM.
        tensorboard_update_freq (int): TensorBoard logging frequency in steps.

    Returns:
        None. (Saves the trained model and test set to disk.)
    '''

    model_specific_output_dir = f'{models_output_dir}{model_name}_package/'
    log_dir = model_specific_output_dir + 'logs/'
    checkpoint_dir = model_specific_output_dir + f'checkpoints/' + 'model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras'

    print(f'Model: {model_name} - Epochs: {n_epochs} - Batch_size: {batch_size}')
    print(f'Train/val/test: {train_batches}/{val_batches}/{test_batches}')

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_dir,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=10,
            write_graph=True,
            write_images=True,
            update_freq=tensorboard_update_freq
        )
    ]

    tf_dataset = md.tf_bal_dataset(
        ds_name=train_ds_name,
        data_input_dir=train_ds_dir,
        metadata_dir=metadata_dir,
        batch_size=batch_size,
        n_samples_per_label=label_dict)
    print('Balanced dataset loaded.')

    tf_train, tf_val, tf_test = md.tf_train_val_test_sets(
        tf_dataset, train_batches, val_batches, test_batches,
        train_repeat=True, val_repeat=False, test_repeat=False)
    
    input_shape, label_shape = md.tf_input_shape(tf_dataset)
    n_classes = label_shape[0]
    print('Dataset split. Shape:')
    print(f'Train ({train_batches}): {tf_train} \n Val ({val_batches}): {tf_val} \n Test ({test_batches}): {tf_test}')
    
    print("Creating new model...")
    if model_generator is None:
        model_generator = models.ECG_XPLAIM_model_generator(
            lr_initial=1e-2,
            lr_decay_steps=train_batches,
            lr_decay_rate=0.95
        )
    model = model_generator.create_model(input_shape, n_classes)

    print('Training started...')
    model.fit(tf_train, epochs=n_epochs, steps_per_epoch=train_batches,
              validation_data=tf_val, validation_steps=val_batches, callbacks=callbacks)
     
    model_file_path = model_specific_output_dir + f'model_{model_name}.keras'
    model.save(model_file_path)
    print(f"Model saved as {model_file_path}.")

    print('Test_set is being converted to np...')
    tf_to_np_test = md.tf_dataset_to_numpy(tf_test)
    print(f'tf_to_np_test set size: {round((getsizeof(tf_to_np_test[0]) + getsizeof(tf_to_np_test[1])) / (1024*1024), 2)} Mb.')
    testset_file_path = model_specific_output_dir + f'test_set_{model_name}.npz'
    np.savez(testset_file_path, samples=tf_to_np_test[0], labels=tf_to_np_test[1])
    print(f"Test set saved as {testset_file_path}.")


# CLI support
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ECG-XPLAIM model")

    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--label-counts', nargs='+', metavar=('LABEL', 'COUNT'), required=True,
                        help="Provide label/count pairs like --label-counts lqt 100 neg 1000")
    parser.add_argument('--train-ds-name', type=str, choices=['mimic-iv', 'ptb-xl'], required=True)
    parser.add_argument('--train-ds-dir', type=str, required=True)
    parser.add_argument('--metadata-dir', type=str, required=True)
    parser.add_argument('--train-batches', type=int, required=True)
    parser.add_argument('--val-batches', type=int, required=True)
    parser.add_argument('--test-batches', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--models-output-dir', type=str, required=True)
    parser.add_argument('--tensorboard-freq', type=int, default=5)
    # model_generator > ECG_XPLAIM by default

    args = parser.parse_args()

    # Convert label-counts list into dict
    label_dict = {args.label_counts[i]: int(args.label_counts[i + 1])
                  for i in range(0, len(args.label_counts), 2)}

    tf_model_train(
        model_name=args.model_name,
        label_dict=label_dict,
        train_ds_name=args.train_ds_name,
        train_ds_dir=args.train_ds_dir,
        metadata_dir=args.metadata_dir,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        test_batches=args.test_batches,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        models_output_dir=args.models_output_dir,
        tensorboard_update_freq=args.tensorboard_freq
    )
