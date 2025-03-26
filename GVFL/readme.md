# GVFL
### GVFL-EEG: Guided Visual Feature Learning from Randomized EEG Trials for Object Recognition

##### Core idea: basic constrastive learning for image and EEG. Interesting analysis from neuroscience perspective! 🤣

## Abstract
![alt text](image-28.png)

- We introduce GVFL-EEG, a novel two-stage learning
framework designed for decoding visual object cate-
gories from randomized EEG trials. This framework uti-
lizes a pre-trained image encoder to guide the training of
an EEG encoder, enabling effective extraction of visual
representations from EEG signals.
- We develop EEGMambaformer, an EEG encoder incor-
porating a residual Mamba block to capture the temporal
dynamics of EEG signals. This is further enhanced by an
inverted transformer encoder that applies self-attention
across EEG channels to elucidate spatial dependencies
inherent in the EEG data.
- Our experimental results demonstrate that the proposed
framework successfully extracts visual representations of
perceived objects from EEG responses, achieving state-
of-the-art performance for object recognition using ran-
domized EEG trials.
- We conduct extensive experiments to investigate how dif-
ferent oscillation rhythms and time ranges affect decod-
ing performance, thereby enhancing our understanding
of the neural encoding of visual information.

## Datasets
many thanks for sharing good datasets!
[EEG40000](https://ieee-dataport.org/open-access/dataset-object-classification-randomized-eeg-trials)


## EEG pre-processing
### Script path
- `./GVFL/data_preprocess/bin_self/`
### Data path 
- raw data: `./Data/Things-EEG2/Raw_data/`
- proprocessed eeg data: `./Data/Things-EEG2/Preprocessed_data_250Hz/`
### Steps(尚未编辑)
1. pre-processing EEG data 
   - modify `preprocessing_utils.py` as you need.
     - choose channels
     - epoching
     - baseline correction
     - resample to 250 Hz
     - sort by condition
     - Multivariate Noise Normalization (z-socre is also ok)
   - `python preprocessing.py` for each subject. 

2. get the center images of each test condition (for testing, contrast with EEG features)
   - get images from original Things dataset but discard the images used in EEG test sessions.


### Annotation
please ensure that you have file "sensor_dataframe.xlsx" ,~~"eeg_vit.csv" and "img_vit.csv"~~ in your root_path
  
## Image features from pre-trained models



## Prepare Environment
~~~
pip install -r requirements.txt
~~~


## Training and testing

To run the code, follow these steps:

### Modify the Dataset Path
Open the configuration file (e.g.,  config.py) and update the dataset path to point to your preprocessed file. details please see config.py


### Run the Code
Use the following command to start the program:
for pretrain:
~~~
bash shell/pretrain.sh
~~~

for finetune classification:
~~~
bash shell/train_classification.sh
~~~





## Citation(尚未编辑)
Hope this code is helpful. I would appreciate you citing us in your paper. 😊
```
@inproceedings{song2024decoding,
  title = {Decoding {{Natural Images}} from {{EEG}} for {{Object Recognition}}},
  author = {Song, Yonghao and Liu, Bingchuan and Li, Xiang and Shi, Nanlin and Wang, Yijun and Gao, Xiaorong},
  booktitle = {International {{Conference}} on {{Learning Representations}}},
  year = {2024},
}
```
<!-- ## Acknowledgement

## References

## License -->