# ECG-XPLAIM: Deep learning tool for accurate and explainable arrhythmia detection from 12-lead electrocardiogram (ECG) signals

#### Panteleimon Pantelidis, Samuel Ruipérez-Campillo, Julia E Vogt, Alexios Antonopoulos, Ioannis Gialamas, George E Zakynthinos, Polychronis Dilaveris, Jose Millet, Theodore G Papaioannou, Evangelos Oikonomou, Gerasimos Siasos

This repository provides a reproducible pipeline for training and evaluating ECG-XPLAIM (eXPlainable Locally-adaptive Artificial Intelligence Model), which is a deep learning-based model designed for explainable ECG classification, optimized for multi-label classification of 12-lead electrocardiogram (ECG) signals. The model integrates a custom Inception-style one-dimensional convolutional neural network (CNN), specialized for time-series analysis, capturing both local waveform features (waves, intervals, QRS morphology) and global rhythm patterns (RR variability, conduction disturbances).

---

## Citation

> Citation information (journal link & bibtex entry) – **coming soon**

---

## Environment Setup

Create and activate the conda environment with required packages in Linux:

```bash
conda env create -f custom_dl.yaml
conda activate custom_dl
```

Python version: `3.10+`  
Key packages: `tensorflow`, `numpy`, `matplotlib`, `scikit-learn`, `wfdb`, `statsmodels`, `pandas`, `scipy`

---

## Datasets

This repository is configured to work with the following datasets, which are expected to be downloaded and placed in the corresponding directories:

- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
- [MIMIC-IV ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)

Download into the following folder structure from terminal:

```bash
# PTB-XL
mkdir -p data/ptb-xl
wget -r -N -c -np -nH --cut-dirs=3 -P data/ptb-xl https://physionet.org/files/ptb-xl/1.0.3/

# MIMIC-IV
mkdir -p data/mimic-iv
wget -r -N -c -np -nH --cut-dirs=3 -P data/mimic-iv https://physionet.org/files/mimic-iv-ecg/1.0/
```

---

## Metadata Preprocessing

Set up the required directory structure and extract annotation metadata for both datasets (PTB-XL and MIMIC-IV): 

```bash
python src_files/initial_setup.py --datasets='all' --data-dir='data/'
```

This will generate:

```
output/metadata/ptb-xl_labels_metadata.csv
output/metadata/mimic-iv_labels_metadata.csv
```

---

## Model Architecture

![Figure - GitHub](https://github.com/user-attachments/assets/64d27a71-633b-4591-af9d-c0ae793a14e7)

**Input**: `shape = (5000, 12)`  
**Output**: multi-label classification (sigmoid)

Adaptable to other settings with different sampling frequencies, signal lengths or number of channels.

---

## Training

You can train models either via a Python IDE (e.g., Jupyter Notebook) or through the command line (CLI). The following example uses a small number of input samples and is trained for few epochs for quick demonstration purposes (train on MIMIC-IV dataset, for identifying long QT):

### Python (script / notebook)

```python
from src_files.train import tf_model_train

tf_model_train(
    model_name='ecg_pace_test_IDE_v1', 
    label_dict={'lqt': 500, 'neg': 500},
    train_ds_name='mimic-iv',
    train_ds_dir='data/mimic-iv/',
    metadata_dir='output/metadata/',
    train_batches=10, val_batches=3, test_batches=2,
    n_epochs=3, batch_size=32,
    models_output_dir='output/models/',
    model_generator=None,  # Defaults to ECG-XPLAIM
    tensorboard_update_freq=5
)
```

### CLI

```bash
python -m src_files.train \
  --model-name ecg_pace_test_CLI_v1 \
  --label-counts lqt 500 neg 500 \
  --train-ds-name mimic-iv \
  --train-ds-dir data/mimic-iv/ \
  --metadata-dir output/metadata/ \
  --train-batches 10 \
  --val-batches 3 \
  --test-batches 2 \
  --epochs 3 \
  --batch-size 32 \
  --models-output-dir output/models/
```

---

## Saved Outputs

Each model training procedure creates a directory:

```
output/models/<model_name>_package/
├── checkpoints/          # Best checkpoints by val_loss
├── logs/                 # TensorBoard logs
├── model_<name>.keras    # Final model weights after training completion
├── test_set_<name>.npz   # Test set kept separately to be used for evaluation
```

---

## Evaluation & Explainability

Open and run the notebook `evaluate.ipynb` (documentation within the notebook).

It covers:

- Loading saved and pre-trained versions of ECG-XPLAIM
- Running predictions (inference) on separate subsets and saved test sets
- Evaluate performance metrics
- Train baseline models and compare against ECG-XPLAIM
- Apply explainability mechanisms with Grad-CAM visualizations

---

## Visualization Types

### ECG Signal Plots

- `quick_plot`: fast and minimal
- `fine_plot`: traditional grid-style ECGs with red boxes

### Grad-CAM Explainability

- `gradcam_plot`: 12-lead heatmap overlay
- `gradcam_plot_single`: single-lead focused view

Visualizations saved to:

```
output/imgs/
```

---

## Pretrained Models

Pre-trained, task-specific ECG-XPLAIM model versions are available at [Zenodo](https://zenodo.org/records/14968732).

---

## Project Structure

```
home/
├── data/                        # Raw ECG datasets (placed here manually)
│   ├── mimic-iv/                # MIMIC-IV waveform data
│   └── ptb-xl/                  # PTB-XL waveform data
│
├── output/                      # Output directory
│   ├── models/                  # Trained model checkpoints, final weights and separated test sets
│   ├── metadata/                # Label metadata (.csv files)
│   └── imgs/                    # Saved ECG plots and Grad-CAM visualizations
│
├── src_files/                   # Source code
│   ├── ecg_plot.py              # ECG and Grad-CAM visualization utilities
│   ├── initial_setup.py         # Metadata generation and folder setup
│   ├── load_helpers.py          # Signal loading utilities
│   ├── manipulate_dataset.py    # Dataset creation and batching logic
│   ├── models.py                # Model definitions (CNN, GRU, ECG-XPLAIM)
│   ├── signal_preprocess.py     # Preprocessing functions for signals
│   ├── test_metrics.py          # Evaluation and comparison metrics
│   ├── train.py                 # Training pipeline
│   └── xai.py                   # Grad-CAM generation
│
├── evaluate.ipynb               # Jupyter notebook for evaluation & explainability
├── ecg_xplaim_env.yaml          # Conda environment specification
├── LICENSE                      # Code license
└── README.md                    # Project documentation
```

---

## Contact

For questions, suggestions or collaborations:  
[Panteleimon Pantelidis](https://www.linkedin.com/in/ppantelidis/) – pan.g.pantelidis@gmail.com
