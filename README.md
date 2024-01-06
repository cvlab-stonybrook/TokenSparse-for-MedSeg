# TokenSparse-for-MedSeg
Code for IPMI2023 paper "Token Sparsification for Faster Medical Image Segmentation"
```
@inproceedings{zhou2023token,
  title={Token Sparsification for Faster Medical Image Segmentation},
  author={Zhou, Lei and Liu, Huidong and Bae, Joseph and He, Junjun and Samaras, Dimitris and Prasanna, Prateek},
  booktitle={International Conference on Information Processing in Medical Imaging},
  pages={743--754},
  year={2023},
  organization={Springer}
}
```
## Package Versions
```
python=3.8.12
pytorch=1.7.1=py3.8_cuda10.1.243_cudnn7.6.3_0
torchvision=0.8.2=py38_cu101
monai-weekly==0.9.dev2152
nibabel==3.2.1
omegaconf==2.1.1
timm==0.4.12
torchprofile==0.0.4
```

## Prepare Dataset
1. Download BTCV from https://www.synapse.org/#!Synapse:syn3376386 and MSD BraTS from http://medicaldecathlon.com/
2. Additionally for BTCV, copy the `lib/data/transunet.json` into the data folder.
3. Set the `data_path` in the config file as the above data folder.

## Experiments
### BTCV
#### Training Script
```
python main.py configs/adaseg3d_btcv.yaml \
    --run_name=AdaSegx8v2_3D_vit_base_btcv_enc12l_tploc258_tpratio09_tptau1_compl1l768_scorepredmlp_logsigmoid_fusionnet_dp01_newinnorm \
    --patch_size=8 \
    --a_min=-958 \
    --a_max=326 \
    --b_min=-1 \
    --b_max=1 \
    --dec_arch=UNETRx8_decoderv2 \
    --tp_loc=2-5-8 \
    --tp_ratio=0.9 \
    --compl_embed_dim=768 \
    --compl_depth=1 \
    --compl_num_heads=12 \
    --completion_net=FusionNet \
    --drop_path=0.1 \
    --batch_size=2 \
    --lr=1.72e-2 \
    --warmup_epochs=50 \
    --epochs=5000 \
    --save_freq=100 \
    --eval_freq=100
```
* `config/adaseg3d_btcv.yaml`: The configuration file located at `configs`. It sets all the parameters needed for the training and evalution. Parameters will be overwritten if specified later in the command line.
* `patch_size`: 3D patch size; 8 means patch size of $8\times8\times8$.
* `a_min`, `a_max`, `b_min`, and `b_max`: parameters that specifiy the data preprocessing in MONAI.
* `dec_arch`: As our patch size is smaller than the original UNETR paper, the input feature map of decoder has larger resolution. Thus, we adapt the original decoder (`patch_size=16`) to a new decoder without reducing depth.
* `tp_loc` and `tp_ratio`: where to prune tokens and what is the pruning ratio.
* `completion_net`, `compl_embed_dim`, `compl_depth`, and `compl_num_heads`: Completion network arguments. They specify the completion net architecture, embedding dimension, depth, and number of heads.
* `lr`: This learning rate (1.72e-2) is based on a batch size of 256. That means, with a batch size of 2, the actual learning rate is `1.72e-2 * 2 / 256 = 1.3e-4`.
* `eval_freq`: how frequently the model will evaluate

#### Speed Measure Script
```
python speed_test.py configs/adaseg3d_btcv.yaml \
    --patch_size=8 \
    --dec_arch=UNETRx8_decoderv2 \
    --tp_loc=2-5-8 \
    --tp_ratio=0.9 \
    --compl_embed_dim=768 \
    --compl_depth=1 \
    --compl_num_heads=12 \
    --completion_net=FusionNet \
    --batch_size=1 \
    --num_samples=1
```
You should be able to get a MAC of 146.627 G. Throughputs vary in difference devices. A100 results are reported in our paper.

### MSD BraTS
#### Training Script
```
python main.py configs/adaseg3d_msdbrats.yaml \
    --run_name=AdaSegx8v2_3D_vit_base_msd_brats_enc12l_tploc258_tpratio075_tptau1_compl1l768_scorepredmlp_logsigmoid_fusionnet \
    --patch_size=8 \
    --dec_arch=UNETRx8_decoderv2 \
    --tp_loc=2-5-8 \
    --tp_ratio=0.75 \
    --compl_embed_dim=768 \
    --compl_depth=1 \
    --compl_num_heads=12 \
    --completion_net=FusionNet \
    --batch_size=1 \
    --num_samples=3 \
    --cache_rate=0.5 \
    --lr=1.72e-2 \
    --epochs=533 \
    --warmup_epochs=13 \
    --save_freq=100 \
    --eval_freq=1
```
* Most parameters are same as BTCV
* `cache_rate`: For MONAI to pre-cache data.
* `num_samples`: MONAI can crop multiple samples from one 3D volume. If your GPU is out of memory, try to reduce this to 2 or 1.

#### Speed Measure Script
```
python speed_test.py configs/adaseg3d_msdbrats.yaml \
    --patch_size=8 \
    --dec_arch=UNETRx8_decoderv2 \
    --tp_loc=2-5-8 \
    --tp_ratio=0.75 \
    --compl_embed_dim=768 \
    --compl_depth=1 \
    --compl_num_heads=12 \
    --completion_net=FusionNet \
    --batch_size=1 \
    --num_samples=1
```
You should be able to get a MAC of 439.42 G.
*Note that our paper reports 428.28 G as we used `UNETRx8_decoder` as the `dec_arch` which is a bit shallower than `v2`. This small difference does not change our conclusion.