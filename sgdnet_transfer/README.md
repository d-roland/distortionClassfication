# Transfer learning from SGDNet

Source : SGDNet, An End-to-End Saliency-Guided Deep Neural Network for  No-Reference Image Quality Assessment
The pdf can be found in [this link], original code on: https://github.com/ysyscool/SGDNet.
```
@inproceedings{yang2019sgdnet,
  title={SGDNet: An End-to-End Saliency-Guided Deep Neural Network for No-Reference Image Quality Assessment},
  author={Yang, Sheng and Jiang, Qiuping and Lin, Weisi and Wang, Yongtao},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  year={2019},
  organization={ACM}
}
```

## Basline Model
The SGDNet base code is in the folder sgdnet_base for reference

## Transfer learning
The fine tuning of SGDNet base model is controlled via sgdnet_transfer.py (with 8 classes or  with 3 classes)\
Usage:\
python3 transfer_sgdnet.py --data_dir <DATASET_DIR> --label_scheme <0 for 8 classes or 1 for 3 classes>

## Evaluation of transfered models
The evaluation can be run via evaluate_sgdnet.py (both for model with 8 classes or model with 3 classes)\
Usage:\
python3 evaluate_sgdnet.py --data_dir <DATASET_DIR> --label_scheme <0 for 8 classes or 1 for 3 classes>

## Error analysis on transfered models
The data necessary to perform error analysis can be obtained via error_analysis_sgdnet.py (both for model with 8 classes or model with 3 classes)\
Usage:\
python3 error_analysis_sgdnet.py --data_dir <DATASET_DIR> --label_scheme <0 for 8 classes or 1 for 3 classes>
