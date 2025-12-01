# EvoBrain: Dynamic Multi-channel EEG Graph Modeling for Time-Evolving Brain Network (NeurIPS 2025, Spotlight)


---
## Data

Temple University Seizure Corpus (TUSZ) v1.5.2 dataset is publicly available [here](https://isip.piconepress.com/projects/tuh_eeg/).
Once your request form is accepted, you can access the dataset.

---

## Setup

You can install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

---

## Preprocessing
The preprocessing step resamples all EEG signals to 200Hz, and saves the resampled signals in 19 EEG channels as `h5` files.

On terminal, run the following:
```bash
python ./data/resample_signals.py --raw_edf_dir <tusz-data-dir> --save_dir <resampled-dir>
```
where `<tusz-data-dir>` is the directory where the downloaded TUSZ v1.5.2 data are located, and `<resampled-dir>` is the directory where the resampled signals will be saved.

## Experiments
### Configurations
You can modify settings and training parameters by editing the 'args.py' file. 
This includes adjusting the task, model, number of epochs, learning rate, batch size, and other model training parameters. 
Alternatively, you can specify them during execution using flags like '--num_epochs'.

### RUN
To train and test, you can run: 
```bash
python main.py --dataset TUSZ --input_dir <resampled-dir> --raw_data_dir <tusz-data-dir> --save_dir results --model_name evobrain --num_epochs 100 
```
where `<save-dir>` is the directory where the results are located.



## Citation

If you find this work useful, please cite our paper:
```bibtex
@inproceedings{
kotoge2025evobrain,
title={EvoBrain: Dynamic Multi-Channel {EEG} Graph Modeling for Time-Evolving Brain Networks},
author={Rikuto Kotoge and Zheng Chen and Tasuku Kimura and Yasuko Matsubara and Takufumi Yanagisawa and Haruhiko Kishima and Yasushi Sakurai},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025}
}
```