## Class-Incremental Exemplar Compression for Class-Incremental Learning

\[[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Luo_Class-Incremental_Exemplar_Compression_for_Class-Incremental_Learning_CVPR_2023_paper.pdf)\]

### Getting Started
To run this repository, we kindly advise you to install python 3.6.8 and other requirements within [Anaconda](https://www.anaconda.com/) environment.
```python
conda create -n CIM-PyTorch python=3.6.8
conda activate CIM-PyTorch
pip install -r requirements.txt
```

### Dataset Preparation
* `Food-101`: download from its official [link](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/);
* `ImageNet-1000`: download from its official [link](https://image-net.org/);
* `ImageNet-100`: split from `ImageNet-1000` following [LUCIR](https://github.com/hshustc/CVPR19_Incremental_Learning).
  
### Running Experiments
* Revise the dataset directory in `utils/data.py`;
* Revise the experimental configurations in `cim-<YOUR DATASET>.json`;
* Run command: `python main.py --config cim-<YOUR DATASET>.json`.

### Acknowledgements
* [FOSTER](https://github.com/G-U-N/ECCV22-FOSTER)
* [Rational Activations](https://github.com/ml-research/rational_activations)
