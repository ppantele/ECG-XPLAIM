from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Recall, Precision, AUC
from tensorflow.keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay



class Simple_CNN_generator:
    '''
    Builds a basic 1D CNN model for multi-label ECG classification.

    Architecture:
        - 3 Conv1D + MaxPooling + BatchNorm blocks
        - Flatten + Dense + Dropout
        - Sigmoid output for multi-label classification

    Methods:
        create_model(input_shape, n_classes): Returns a compiled Keras model.
    '''

    def __init__(self):
        pass

    def create_model(self, input_shape, n_classes):
        model = models.Sequential([
            layers.Input(shape=input_shape),  # Explicit Input layer
            layers.Conv1D(filters=32, kernel_size=5, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            layers.Dense(n_classes, activation='sigmoid')
        ])

        model.compile(
           optimizer=Adam(),
           loss="binary_crossentropy",
           metrics=['accuracy', Recall(), Precision(), AUC()])    
        
        return model



class Simple_GRU_generator:
    '''
    Builds a simple GRU-based recurrent neural network for multi-label ECG classification.

    Architecture:
        - 2 stacked GRU layers with LayerNormalization and Dropout
        - Dense layers with BatchNorm and Dropout
        - Sigmoid output for multi-label classification

    Methods:
        create_model(input_shape, n_classes): Returns a compiled Keras model.
    '''

    def __init__(self):
        pass

    def create_model(self, input_shape, n_classes):
        model = models.Sequential([
          layers.GRU(128, activation='tanh', return_sequences=True, input_shape=input_shape),
          layers.LayerNormalization(),  # Normalize activations
          layers.Dropout(0.3),  # Dropout for regularization
          
          layers.GRU(64, activation='tanh', return_sequences=False),
          layers.LayerNormalization(),
          layers.Dropout(0.3),
          
          layers.Dense(64, activation='relu'),
          layers.BatchNormalization(),
          layers.Dropout(0.3),
          layers.Dense(32, activation='relu'),
          layers.BatchNormalization(),
          layers.Dropout(0.3),
          
          layers.Dense(n_classes, activation='sigmoid')  # Multi-label classification
        ])

        model.compile(
           optimizer=Adam(),
           loss="binary_crossentropy",
           metrics=['accuracy', Recall(), Precision(), AUC()])    
        
        return model



class ECG_XPLAIM_model_generator:
    '''
    Generator for the ECG-XPLAIM model: a multi-branch, residual CNN architecture 
    with hierarchical temporal features for ECG classification.

    Args:
        n_filters (list): Number of filters per block. Default: [16, 32, 128]
        kernel_size (list): Set of kernel sizes used in each multi-branch Conv1D layer. Default: [2, 10, 40]
        lr_initial (float): Initial learning rate. Default: 0.01
        lr_decay_steps (int): Steps for exponential LR decay. Default: 1,000
        lr_decay_rate (float): Learning rate decay rate. Default: 0.9

    Architecture:
        - 3 residual blocks with multiple parallel Conv1D layers
        - Global Average Pooling + Dense (sigmoid) output
        - Exponential learning rate decay scheduler

    Methods:
        create_model(input_shape, n_classes): Returns a compiled Keras model.
    '''
    
    def __init__(
        self,
        n_filters = [16, 32, 128],  # alternatively: [8, 16, 64] or [32, 64, 256]
        kernel_size = [2, 10, 40],  # alternatively: [2, 20, 80] or [2, 40, 150]      
        lr_initial=1e-2,
        lr_decay_steps=1000,
        lr_decay_rate=0.9):
        
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.lr_initial = lr_initial
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate

    
    def _process_branch(self, input, depth):
        kernels = self.kernel_size
        filters = self.n_filters[depth]
        x = input

        for i in range(2):
            conv_list = []
            for kernel in kernels:
                conv_list.append(layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel,
                    strides=1, 
                    padding='same', 
                    activation='relu', 
                    use_bias=False)(x))
            x = layers.Concatenate(axis=2)(conv_list)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation='relu')(x)
        
        return x

  
    def _retain_branch(self, input, output):
        shortcut_x = layers.Conv1D(
            filters=int(output.shape[-1]),
            kernel_size=1,
            padding='same',
            use_bias=False)(input)
        shortcut_x = layers.BatchNormalization()(shortcut_x)
    
        x = layers.Add()([shortcut_x, output])
        x = layers.Activation('relu')(x)
        x = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x)
    
        return x


    def create_model(self, input_shape, n_classes):
        
        input_layer = layers.Input(input_shape)
        x = input_layer
        input_retain = input_layer
           
        for block_index in range(3):  # 3 Res blocks
            x = self._process_branch(x, depth=block_index)
            x = self._retain_branch(input_retain, x)
            input_retain = x
        
        gap_layer = layers.GlobalAveragePooling1D()(x)
        output_layer = layers.Dense(n_classes, activation='sigmoid')(gap_layer)
    
        model = models.Model(inputs=input_layer, outputs=output_layer)
    
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.lr_initial,
            decay_steps=self.lr_decay_steps,
            decay_rate=self.lr_decay_rate,
            staircase=True)
        
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss="binary_crossentropy",
            metrics=['accuracy', Recall(), Precision(), AUC()])
        return model

