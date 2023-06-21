## Class-Incremental Exemplar Compression for Class-Incremental Learning

\[[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Luo_Class-Incremental_Exemplar_Compression_for_Class-Incremental_Learning_CVPR_2023_paper.pdf)\] \[[Poster](https://github.com/xfflzl/CIM-CIL/blob/main/materials/poster.pdf)\]

### Introduction
Exemplar-based class-incremental learning (CIL) finetunes the model with all samples of new classes but few-shot exemplars of old classes in each incremental phase, where the "few-shot" abides by the limited memory budget. In this paper, we break this "few-shot" limit based on a simple yet surprisingly effective idea: compressing exemplars by downsampling non-discriminative pixels and saving "many-shot" compressed exemplars in the memory. Without needing any manual annotation, we achieve this compression by generating 0-1 masks on discriminative pixels from class activation maps (CAM). We propose an adaptive mask generation model called class-incremental masking (CIM) to explicitly resolve two difficulties of using CAM: 1) transforming the heatmaps of CAM to 0-1 masks with an arbitrary threshold leads to a trade-off between the coverage on discriminative pixels and the quantity of exemplars, as the total memory is fixed; and 2) optimal thresholds vary for different object classes, which is particularly obvious in the dynamic environment of CIL. We optimize the CIM model alternatively with the conventional CIL model through a bilevel optimization problem. We conduct extensive experiments on high-resolution CIL benchmarks including Food-101, ImageNet-100, and ImageNet-1000, and show that using the compressed exemplars by CIM can achieve a new state-of-the-art CIL accuracy, e.g., 4.8 percentage points higher than FOSTER on 10-Phase ImageNet-1000.
![framework](https://github.com/xfflzl/CIM-CIL/blob/main/materials/framework.png)
> The proposed compression pipeline. Given an image, we extract its CAM/CIM-based mask, threshold
it to be a 0-1 mask with a fixed threshold, and generate a tight bounding box that covers all masked pixels. Then, we downsample the pixels outside the bounding box and sum the downsampled image up to the masked image to generate the final compressed image.

### Getting Started
To run this repository, we kindly advise you to install python 3.6.8 and other requirements within [Anaconda](https://www.anaconda.com/) environment.
```python
conda create -n CIM-PyTorch python=3.6.8
conda activate CIM-PyTorch
pip install -r requirements.txt
```

### Dataset Preparation
* `Food-101`: download from its official [link](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/); split the training and testing subsets according to its meta files;
* `ImageNet-1000`: download from its official [link](https://image-net.org/);
* `ImageNet-100`: split from `ImageNet-1000` following [LUCIR](https://github.com/hshustc/CVPR19_Incremental_Learning).
  
### Running Experiments
* Revise the dataset directory in `utils/data.py`; 
* Revise the experimental configurations in `cim-<YOUR DATASET>.json`;
* Run command: `python main.py --config cim-<YOUR DATASET>.json`.

### Acknowledgements
* [FOSTER](https://github.com/G-U-N/ECCV22-FOSTER)
* [Rational Activations](https://github.com/ml-research/rational_activations)
