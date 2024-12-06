# Quantifying the Limits of Segment Anything Model for Tree-like and Low-Contrast Objects

#### By [Yixin Zhang*](https://scholar.google.com/citations?user=qElWNMwAAAAJ&hl=en). [Nicholas Konz*](https://nickk124.github.io/), Kevin Kramer, and [Maciej Mazurowski](https://sites.duke.edu/mazurowski/).
**(*= equal contribution)**

arXiv paper link: [![arXiv Paper](https://img.shields.io/badge/arXiv-2412.04243-orange.svg?style=flat)](https://arxiv.org/abs/2412.04243)

<p align="center">
  <img src='https://github.com/mazurowski-lab/segmentation-metrics/blob/main/figs/teaser.png' width='75%'>
</p>

[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) has shown extensive promise in segmenting objects from a wide range of contexts unseen in training, yet still has surprising trouble with certain types of objects such as those with dense, tree-like structures, or low textural contrast with their surroundings.

In our paper, [*Quantifying the Limits of Segment Anything Model: Analyzing Challenges in Segmenting Tree-Like and Low-Contrast Structures*](https://arxiv.org/abs/2412.04243), we propose metrics that quantify the tree-likeness and textural contrast of objects, and show that SAM's ability to segment these objects is noticeably correlates with these metrics (see below). **This codebase provides the code to easily calculate these metrics.**

<p align="center">
  <img src='https://github.com/mazurowski-lab/segmentation-metrics/blob/main/figs/corr.png' width='75%'>
</p>

## Citation

Please cite our paper if you use our code or reference our work:
```bib
@article{zhang2024texturalconfusion,
      title={Quantifying the Limits of Segment Anything Model: Analyzing Challenges in Segmenting Tree-Like and Low-Contrast Structures}, 
      author={Yixin Zhang and Nicholas Konz and Kevin Kramer and Maciej A. Mazurowski},
      year={2024},
      eprint={2412.04243},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.04243}, 
}
```

## 1) Installation
Please run `pip3 install -r requirements.txt` to install the required packages.

## 2) Usage

You can easily compute these metrics for your own segmentation masks as shown in the following example.

```python
import torch
device = "cuda" # or "cpu"

object_mask = torch.load('path/to/your/mask.pt')
assert object_mask.ndim == 4
# ^ ! mask needs to be of shape (1, H, W)
```

### Tree-likeness Metrics

```python
from treelikeness_metrics import get_CPR, get_DoGD
cpr = get_CPR(object_mask, device=device)
dogd = get_DoGD(object_mask, device=device)
```

The hyperparameters of these metrics ($r$ for CPR, and $a$ and $b$ for DoGD) can also be adjusted from their default values, as shown below.

```python
dogd = get_CPR(object_mask, rad=7, device=device)
cpr = get_DoGD(object_mask, a=63, b=5, device=device)
```

### Textural Contrast/Separability Metrics

Note that the textural contrast/separability metrics additionally require the image that the object mask corresponds to:

```python
import torchvision.transforms as transforms
from PIL import Image
from textural_contrast_metrics import TexturalMetric

img = transforms.functional.to_tensor(Image.open('path/to/your/image.png').convert('RGB')).to(device)

metric = TexturalMetric(device)
separability = metric.get_separability_score(img, object_mask)
```
