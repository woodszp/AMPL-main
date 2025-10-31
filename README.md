# AMPL 

This is the PyTorch implementation of ["AMPL: An Adaptive Meta-Prompt Learner for Few-Shot Image Classification"](). 




## Installation

Python 3.8, Pytorch 1.11, CUDA 11.3. The code is tested on Ubuntu 20.04.


We have prepared a conda YAML file which contains all the python dependencies.

```sh
conda env create -f environment.yml
```

To activate this conda environment,

```sh
conda activate ampl
```

We use [wandb](https://wandb.ai/site) to log the training stats (optional). 

## Datasets

- **𝒎𝒊𝒏𝒊ImageNet**

  > The 𝑚𝑖𝑛𝑖ImageNet dataset was proposed by [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) for few-shot learning evaluation. Its complexity is high due to the use of ImageNet images but requires fewer resources and infrastructure than running on the full [ImageNet dataset](https://arxiv.org/pdf/1409.0575.pdf). In total, there are 100 classes with 600 samples of color images per class. These 100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test. To generate this dataset from ImageNet, you may use the repository [𝑚𝑖𝑛𝑖ImageNet tools](https://github.com/y2l/mini-imagenet-tools).


- **𝒕𝒊𝒆𝒓𝒆𝒅ImageNet**

  > The [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet](https://arxiv.org/pdf/1803.00676.pdf) dataset is a larger subset of ILSVRC-12 with 608 classes (779,165 images) grouped into 34 higher-level nodes in the ImageNet human-curated hierarchy. To generate this dataset from ImageNet, you may use the repository 𝑡𝑖𝑒𝑟𝑒𝑑ImageNet dataset: [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet tools](https://github.com/y2l/tiered-imagenet-tools). 


- **CIFAR-FS and FC100**

  > CIFAR-FS and FC100 can be download using the [scripts](https://github.com/icoz69/DeepEMD/tree/master/datasets) from [DeepEMD](https://github.com/icoz69/DeepEMD). 




## Training

Our model are trained on 8 RTX3090 GPUs by default (24GB memory). You can specify the argument ```--nproc_per_node``` in the following ```command``` file as the number of GPUs available in your server, and increase/decrease the argument ```--batch_size_per_gpu``` if your GPU has more/less memory.

- **Pre-training (self-supervised)**

  In this phase, we pretrain our model using the self-supervised learning method [iBOT](https://github.com/bytedance/ibot) and [SMKD](https://github.com/HL-hanlin/SMKD) (use the ). All models are trained for a maximum of 1600 epochs. We evaluate our model on the validation set after training for every 50 epochs, and report the best. 
  1-shot and 5-shot evaluation results with _Prototype_ method is given in the following table. We also provide full checkpoints and test-set features for pretrained models, and command to replicate the results.

  ```--data_path```: need to be set as the location of the training set of dataset XXX (e.g. miniImageNet). 
  ```--output_dir```: location where the phase1 checkpoints and evaluation files to be stored.


  <table>
    <tr>
      <th>Dataset</th>
      <th>1-shot</th>
      <th>5-shot</th>
      <th colspan="3">Download</th>
    </tr>
    <tr>
      <td>𝒎𝒊𝒏𝒊ImageNet</td>
      <td>60.93%</td>
      <td>80.38%</td>
      <td><a href="https://drive.google.com/file/d/1cHRiySKgrgbGqnNvMFY0D9IvWgpO75Jm/view?usp=share_link">checkpoint</a></td>
      <td><a href="https://drive.google.com/drive/folders/1YSxoCnuLidqwXsJCwnuvA3_6JEB4zFm1?usp=share_link">features</a></td>
      <td><a href="https://drive.google.com/file/d/1hJCVuLJQdGbvUjlRv2XzbsxLB6VQvKcm/view?usp=share_link">command</a></td>
    </tr>
    <tr>
      <td>𝒕𝒊𝒆𝒓𝒆𝒅ImageNet</td>
      <td>71.36%</td>
      <td>83.28%</td>
      <td><a href="https://drive.google.com/file/d/1udnoJrpOs5tcfSsGWBsoUQRiGaIZzx29/view?usp=share_link">checkpoint</a></td>
      <td><a href="https://drive.google.com/drive/folders/1i1XHoySqThAm_EOSxB6BhbA6BH6GRat4?usp=share_link">features</a></td>
      <td><a href="https://drive.google.com/file/d/1zjRRRzc_RU_jXcA8YGQVuyTgHeidDPmo/view?usp=share_link">command</a></td>
    </tr>
    <tr>
      <td>CIFAR-FS</td>
      <td>65.70%</td>
      <td>83.45%</td>
      <td><a href="https://drive.google.com/file/d/1tag6WuM9Ps1PnLgt7VoCIcPrxCqVEqO3/view?usp=share_link">checkpoint</a></td>
      <td><a href="https://drive.google.com/drive/folders/1phzC-CuER4QvhP3XTrl7a2g7uks6SCbK?usp=share_link">features</a></td>
      <td><a href="https://drive.google.com/file/d/1dGEUgq0HOJ0nL2jMHxdcNiVkJeOmUCdr/view?usp=share_link">command</a></td>
    </tr>
      <tr>
      <td>FC100</td>
      <td>44.20%</td>
      <td>61.64%</td>
      <td><a href="https://drive.google.com/file/d/1CAWtHJvvVKjptQh07sb9T50UeaqKYdru/view?usp=share_link">checkpoint</a></td>
      <td><a href="https://drive.google.com/drive/folders/1VRZ-McBcHHFwsA-h8QVDNBdSQK5CKrbH?usp=share_link">features</a></td>
      <td><a href="https://drive.google.com/file/d/1KhfZq2OcmTvT-xjCzaEI2NaHkCo45WnD/view?usp=share_link">command</a></td>
    </tr>
  </table>


- **Aaptive Meta-Prompt Learner (supervised)**

  In this second phase, we start from the checkpoint in phase 1 and further train the model using the supervised knowledge distillation method proposed in our paper. All models are trained for a maximum of 150 epochs. We evaluate our model on the validation set after training for every 5 epochs, and report the best. Similarly, 1-shot and 5-shot evaluation results with _Prototype_ method is given in the following table. We also provide checkpoints and features for pretrained models.

  ```--pretrained_dino_path```: should be set as the same location as ```--output_dir``` in phase1. 
  ```--pretrained_dino_file```: which checkpoint file to resume from (e.g. ```checkpoint1250.pth```).
  ```--output_dir```: location where the phase2 checkpoints and evaluation files to be stored.

  <table>
    <tr>
      <th>Dataset</th>
      <th>1-shot</th>
      <th>5-shot</th>
      <th colspan="3">Download</th>
    </tr>
    <tr>
      <td>𝒎𝒊𝒏𝒊ImageNet</td>
      <td>74.82%</td>
      <td>88.47%</td>
      <td><a href="">checkpoint</a></td>
      <td><a href="">features</a></td>
      <td><a href="">command</a></td>
    </tr>
    <tr>
      <td>𝒕𝒊𝒆𝒓𝒆𝒅ImageNet</td>
      <td>78.98%</td>
      <td>91.61%</td>
      <td><a href="">checkpoint</a></td>
      <td><a href="">features</a></td>
      <td><a href="">command</a></td>
    </tr>
    <tr>
      <td>CIFAR-FS</td>
      <td>78.69%</td>
      <td>90.68%</td>
      <td><a href="">checkpoint</a></td>
      <td><a href="">features</a></td>
      <td><a href="">command</a></td>
    </tr>
      <tr>
      <td>FC100</td>
      <td>58.34%</td>
      <td>72.25%</td>
      <td><a href="">checkpoint</a></td>
      <td><a href="">features</a></td>
      <td><a href="">command</a></td>
    </tr>
  </table>



## Evaluation 

We use ```eval_ampl.py``` to evaluate a trained model. Before running the evaluation code, we need to specify the image data path in ```server_dict``` of this python file.

For example, we can use the following code to do 5-way 5-shot evaluation on the model trained in Pre-training on mini-ImageNet:

- **prototype**:
```sh
python eval_ampl.py --server mini --num_shots 5 --ckp_path /root/autodl-nas/FSVIT_results/MINI480_phase2 --ckpt_filename checkpoint0040.pth --output_dir /root/autodl-nas/FSVIT_results/MINI480_prototype --evaluation_method cosine --iter_num 10000
```

- **classifier**:
```sh
python eval_ampl.py --server mini --num_shots 5 --ckp_path /root/autodl-nas/FSVIT_results/MINI480_phase2 --ckpt_filename checkpoint0040.pth --output_dir /root/autodl-nas/FSVIT_results/MINI480_classifier --evaluation_method classifier --iter_num 1000
```




## Citation
```BibTeX
@artile{,
      title={AMPL: An Adaptive Meta-Prompt Learner for Few-Shot Image Classification}, 
      author={Zhiping Wu, Lian Huai, Tong Liu, Zeyu Shangguan, Lei Wang, Jing Huo, Wenbin Li*, Yang Gao, Xingqun Jiang.},
      journal={Neural Networks},
      year={2025}
}
```
