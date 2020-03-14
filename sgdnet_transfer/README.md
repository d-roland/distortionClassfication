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
The fine tuning of SGDNet base model is controlled via sgdnet_transfer.py (version 1 with 8 classes) or sgdnet_transfer-2.py (version 2 with 3 classes)

## Evaluation of transfered models
The evaluation can be run via evaluate_sgdnet.py (for model version 1) and evaluate_sgdnet-2.py (for model version 2)

## Error analysis on transfered models
The data necessary to perform error analysis can be obtained via error_analysis_sgdnet.py (for model version 1) and error_analysis_sgdnet-2.py (for model version 2)
