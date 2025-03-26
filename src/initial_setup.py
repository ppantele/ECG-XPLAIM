import os
import argparse
import load_helpers as lh


# Argument parser for selecting datasets
parser = argparse.ArgumentParser(description='Select which ECG dataset(s) to process.')
parser.add_argument('--datasets', choices=['mimic-iv', 'ptb-xl', 'all'], default='all',
                    help='Dataset to preprocess: mimic-iv, ptb-xl, or all (default: all)')
parser.add_argument('--data-dir', type=str, required=True,
                    help='Path to the base directory containing the ECG datasets')
args = parser.parse_args()

data_input_dir = args.data_dir


# Define and create output directories
parent_path = 'output/'
output_dirs = [parent_path + end_dir for end_dir in ['metadata/', 'models/', 'imgs/']]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)


# Define search keys in ECG reports for label annotation
labels_mimic_iv = {
    'afib': ['afib', 'atrial fib'],
    'aflu': ['flut'],
    'afifl': ['afib', 'atrial fib', 'flut'],
    'rbbb': ['rbbb', 'right b'],
    'lbbb': ['lbbb', 'left b'],
    'lafb': ['lafb', 'left ant'],
    'lpfb': ['lpfb', 'left post'],
    'pvc': ['pvc', 'premature vent', 'ventricular premat'],
    'pac': ['pac', 'premature atr', 'atrial premat', 'supraventricular extra'],
    'ami': ['anterior st e', 'anteroseptal st e', 'anterolateral st e', 'ant/septal st e', 'anterior inf', 'infarct anter', 'anteroseptal inf', 'anterolateral inf'],
    'imi': ['inferior st e', 'inferior and lateral st e', 'inferior and septal st e', 'inferior inf', 'infarct infer', 'inferolateral inf'],
    'smi': ['septal st e', 'septal inf', 'infarct septal'],
    'lmi': ['lateral st e', 'lateral inf', 'infarct later'],
    'avb1': ['st deg', 'longed PR'],
    'avb2i': ['mobitz i ', 'wenck'],
    'avb2ii': ['mobitz ii'],
    'avb2': ['nd deg'],
    'avb3': ['rd deg', 'complete a-', 'complete av'],
    'bradsr': ['sinus brad'],
    'svt': ['supraventricular tach'],
    'vt': [' ventricular tach'],
    'tachsr': ['sinus tach'],
    'lqt': ['long qt', 'longed qt'],
    'wpw': ['wpw', 'wolf'],
    'pace': ['paci', 'pace']
}

labels_ptb_xl = {
    'afib': ['AFIB'],
    'aflu': ['AFLT'],
    'afifl': ['AFIB', 'AFLT'],
    'rbbb': ['CRBBB'],
    'lbbb': ['CLBBB'],
    'lafb': ['LAFB'],
    'lpfb': ['LPFB'],
    'pvc': ['PVC'],
    'pac': ['PAC'],
    'ami': ['AMI', 'ASMI', 'ALMI'],
    'imi': ['IMI', 'ILMI', 'IPMI'],
    'smi': ['ASMI', 'SMI'],
    'lmi': ['LMI', 'ILMI', 'ALMI'],
    'avb1': ['1AVB', 'LPR'],
    'avb2i': [],
    'avb2ii': [],
    'avb2': ['2AVB'],
    'avb3': ['3AVB'],
    'bradsr': ['SBRAD'],
    'svt': ['SVTAC', 'PSVT', 'SVARR'],
    'vt': [],
    'tachsr': ['STACH'],
    'lqt': ['LNGQT'],
    'wpw': ['WPW'],
    'pace': ['PACE']
}


# Execute metadata processing based on selected datasets
if args.datasets in ['mimic-iv', 'all']:
    print("Processing mimic-iv dataset...")
    lh.metadata_extract_save(ds_name='mimic-iv', data_input_dir=data_input_dir+'mimic-iv/', metadata_output_dir='output/metadata/', label_kwds=labels_mimic_iv)

if args.datasets in ['ptb-xl', 'all']:
    print("Processing ptb-xl dataset...")
    lh.metadata_extract_save(ds_name='ptb-xl', data_input_dir=data_input_dir+'ptb-xl/', metadata_output_dir='output/metadata/', label_kwds=labels_ptb_xl)
