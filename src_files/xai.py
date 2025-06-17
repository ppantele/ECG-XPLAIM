import tensorflow as tf
import numpy as np


def generate_gradcam(model, input_sample, target_layer_name, class_idx=None):
    '''
    Generate a Grad-CAM activation map for a 1D CNN model.

    This function calculates class-specific activation maps by computing the 
    gradient of the target class score with respect to the output of a specified 
    convolutional layer. It applies global average pooling and ReLU to produce 
    a 1D relevance map over the input signal.

    Args:
        model (tf.keras.Model): Trained Keras model.
        input_sample (np.ndarray): Input sample with shape (1, time_steps, channels), e.g. (1, 5000, 12) for ECG.
        target_layer_name (str): Name of the convolutional layer to target for Grad-CAM.
        class_idx (int, optional): Index of the class for which to compute the activation. If None, the class with the highest prediction score is used.

    Returns:
        np.ndarray: Grad-CAM activation map with shape (time_steps,).
    '''

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_sample)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])  # Default: highest predicted class
        target_class_score = predictions[:, class_idx]

    # Compute gradients
    grads = tape.gradient(target_class_score, conv_output)

    # Perform Global Average Pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=1)

    # Multiply pooled gradients with the convolutional outputs
    conv_output = conv_output[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

    # Apply ReLU and normalize
    cam = np.maximum(cam, 0)  # ReLU activation
    cam = cam / np.max(cam)  # Normalize

    return cam