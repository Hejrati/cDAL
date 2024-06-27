# PyTorch implementation of "Conditional diffusion model with spatial attentionand latent embedding" [MICCAI 2024] #

<div align="center">
  <a href="https://github.com/Hejrati" target="_blank">Behzad&nbsp;Hejrati</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://github.com/soumbane" target="_blank">Soumyanil &nbsp;Banerjee</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://www.humonc.wisc.edu/team_member/carri-glide-hurst-phd/" target="_blank">Carri&nbsp;Glide-Hurst</a>&emsp; <b>&middot;</b> &emsp;
  <a href="https://mdong.eng.wayne.edu/" target="_blank">Ming&nbsp;Dong</a> &emsp;
  <br> <br>
</div>
<br>
<br>


Diffusion models have been used extensively for high quality image and video generation tasks. In this paper, we propose a novel conditional diffusion model with spatial attention and latent embedding
(cDAL) for medical image segmentation. In cDAL, a convolutional neural network (CNN) based discriminator is used at every time-step of the diffusion process to distinguish between the generated labels and the real
ones. A spatial attention map is computed based on the features learned by the discriminator to help cDAL generate more accurate segmentation of discriminative regions in an input image. Additionally, we incorporated a random latent embedding into each layer of our model to significantly reduce the number of training and sampling time-steps, thereby making it much faster than other diffusion models for image segmentation

![Architecture](https://github.com/Hejrati/cDAL/assets/123422511/e64bdace-f9a7-4776-855c-e9245a8d8e2f)




## Set up datasets ##
We trained cDAL on several datasets, including MoNuSeg2018, CXR and Hippocampus. 


## Training cDAL ##
We use the following commands on each dataset for training cDAL. Use `parameters_monu.json` for MonuSeg and `parameters_lung.json` for CXR.

#### MoNuSeg ####
Here you can find general [website](https://monuseg.grand-challenge.org/) of the challenge,
download the dataset
[train](https://drive.google.com/file/d/1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA/view?usp=sharing)
and [test](https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view?usp=sharing) sets.

#### CXR ####

This is the [link](https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset) for Lung segmentation from Chest X-Ray dataset.
To preprocess images, we followed the same standard.


#### Hippocampus 3D ####
In this is [link](http://medicaldecathlon.com/) you can find Hippocampus dataset.This dataset can be directly downloaded from this google drive [link](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)


## Pretrained Checkpoints ##
We have already released pretrained checkpoints on MonuSeg and CXR in ['saved_models'](https://github.com/Hejrati/cDAL/tree/master/saved_models).
Simply download the `saved_models` directory to the code directory. Use `parameters_monu.json` for MonuSeg and `parameters_lung.json` for CXR.

## Evaluation ##
After training, samples can be generated by calling ```sampling_xxx.py```. We evaluate the models with a single NVIDIA Quadro RTX 6000 GPU.


We use the [MONAI](https://github.com/Project-MONAI/MONAI) implementation for Hippocampus dataset to process dataset and compute one-hot encoding. Aslo, we use [DDGAN](https://github.com/NVlabs/denoising-diffusion-gan/blob/main/train_ddgan.py) implemention for our difusion model and time-dependent discriminator. 


![Evaluation1](https://github.com/Hejrati/cDAL/assets/123422511/e0d5f0e6-391c-46aa-b275-f65c8b8e1885)

![Evaluation2](https://github.com/Hejrati/cDAL/assets/123422511/fb4b3d23-415f-4bce-a285-e3953007257f)

![Evaluation3](https://github.com/Hejrati/cDAL/assets/123422511/fe553721-d9d2-4864-b08d-d304e02705d6)




## License ##
Please check the LICENSE file.

## Bibtex ##
Cite our paper using the following bibtex item:



