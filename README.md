# Face Recognition
University coursework, a re-implementation of face recognition models in Matlab

<p align="center" float="left">
    <img src="Screenshot/template_matching.png" width="48%"/>
    <img src="Screenshot/vgg16_xml.png" width="49%"/>
</p>
<p align="center" float="left">
    <img src="Screenshot/roc.png" width="75%"/>
</p>

# Reproducibility
1. MATLAB Toolboxes
    - deep learning toolbox
    - parallel computing toolbox
2. Model Params
    - VGGFace Params

# Dataset
- LFW funneled
- a small subset of celeb dataset

# Folder Structure
```
.
└── root/
    ├── Common/
    │   ├── DatabaseFunctions/
    │   ├── GeneralFunctions/
    │   ├── lfw-details/
    │   └── testLabel.mat
    ├── FaceDatabase/
    │   ├── lfw-funneled/
    │   ├── ...
    │   ├── Train/
    │   │   ├── example_subject_1/
    │   │   └── example_subject_2/
    │   └── Test/
    ├── Models/
    ├── Weights/
    ├── Screenshot/
    ├── Evaluation.m
    ├── EvaluationAll.m
    ├── GenerateROC.mlx
    └── README.md
```
`Common` includes all auxiliary functions for dataset manipulation and dlmodel functions   
`lfw-details` extracted text files from LFW Dataset   
`testLabel.mat` true label values for prepared train and test set   
`FaceDatabase` image dataset dump   
`Models` contain subfolders of essential model functions   
`Weights` contain subfolders of essential model pretrained parameters   
`Screenshot` screenshots   
`Evaluation.m` main script   
`EvaluationAll.m` extra main script   
`GenerateROC.mlx` ROC plot live script   

# License
Licensed under the [MIT License](./LICENSE.md).
