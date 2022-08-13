# RCAN


link: https://github.com/yjn870/RCAN-pytorch

This repository is implementation of the "Image Super-Resolution Using Very Deep Residual Channel Attention Networks".

<center><img src="./figs/fig2.png"></center>
<center><img src="./figs/fig3.png"></center>
<center><img src="./figs/fig4.png"></center>

## Requirements
- PyTorch
- Tensorflow
- tqdm
- Numpy
- Pillow

**Tensorflow** is required for quickly fetching image in training phase.

## Results

For below results, we set the number of residual groups as 6, the number of RCAB as 12, the number of features as 64. <br />
In addition, we use a intermediate weights because training process need to take a looong time on my computer. 😭<br />

<table>
    <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x2</center></td>
        <td><center>RCAN x2</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/monarch.bmp" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_x2_bicubic.png" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_x2_RCAN.png" height="300"></center>
    	</td>
    </tr>
</table>

## Usages

### Train

When training begins, the model weights will be saved every epoch. <br />
If you want to train quickly, you should use **--use_fast_loader** option.

```bash
python main.py --scale 2 \
               --num_rg 10 \
               --num_rcab 20 \ 
               --num_features 64 \              
               --images_dir "" \
               --outputs_dir "" \               
               --patch_size 48 \
               --batch_size 16 \
               --num_epochs 20 \
               --lr 1e-4 \
               --threads 8 \
               --seed 123 \
               --use_fast_loader              
```
activate torch-pan


for scale = 2:

python main.py --arch RCAN --scale 2 --num_rg 10 --num_rcab 20 --num_features 64 --outputs_dir "./ckpt2/RCAN_DFC_BIX2_G10R20P128/" --patch_size 128 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123   --dataset WV3

python main.py --arch HAN --scale 2 --num_rg 10 --num_rcab 20 --num_features 64 --outputs_dir "./ckpt2/HAN_DFC_BIX2_G10R20P128/" --patch_size 128 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123   --dataset WV3



scale = 4:


python main.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --images_dir "./dataset/DIV2K/DIV2K_train_HR" --outputs_dir "./checkpoint/RCAN_BIX4_G10R20P48/" --patch_size 48 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123


python main.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --outputs_dir "./checkpoint/RCAN_DFC_BIX4_G10R20P64/" --patch_size 64 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123   --dataset WV3  --pre_trained --contEpoch 10

python main.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --outputs_dir "./checkpoint/RCAN_pls_BIX4_G10R20P64/" --patch_size 64 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123   --dataset pleiades

python main.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --images_dir "./dataset/DIV2K/DIV2K_train_HR" --outputs_dir "./checkpoint/HAN_BIX4_G10R20P48/" --patch_size 48 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123


python main.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --outputs_dir "./checkpoint/HAN_pls_BIX4_G10R20P64/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123   --dataset pleiades          

python main.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --outputs_dir "./checkpoint/HAN_DFC_BIX4_G10R20P64/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --pre_trained --contEpoch 8 --dataset WV3


python main.py --arch HAN --scale 4 --num_rg 5 --num_rcab 20 --num_features 64 --outputs_dir "./checkpoint/HAN_DFC_BIX4_G5R20P64/" --patch_size 64 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3


python main.py --arch HANRS --scale 4 --num_rg 5 --num_rcab 3 --res_scale 0.2 --num_features 64 --outputs_dir "./checkpoint/HANRS_DFC_BIX4_G5R3P64/" --patch_size 64 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3

python main.py --arch HANRS --scale 4 --num_rg 5 --num_rcab 3 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HANRS_DFC_BIX4_G5R3P64S1/" --patch_size 64 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3

python main.py --arch HANRS --scale 4 --num_rg 10 --num_rcab 15 --res_scale 0.2 --num_features 64 --outputs_dir "./ckpt4/HANRS_DFC_BIX4_G10R15P64S.2_per/HANRS_DFC_BIX4_G10R15P64S.2_per/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3 --per_loss


python main.py --arch HANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HANRS_DFC_BIX4_G10R5P64S1_TV/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3 --pre_trained --contEpoch 8


python main.py --arch HANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HANRS_pls_BIX4_G10R5P64S1_TV2/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset pleiades --pre_trained --contEpoch 9


python main.py --arch HANRS_Hetconv --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HANRS_Hetconv_DFC_BIX4_G10R5P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3 --pre_trained --contEpoch 8

python main.py --arch HANRS_Hetconv --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HANRS_Hetconv_pls_BIX4_G10R5P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset pleiades --pre_trained --contEpoch 13

5311273


python main.py --arch HANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HANRS_pls_BIX4_G10R5P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset pleiades --pre_trained --contEpoch 19 

python main.py --arch HANRSv2 --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HANRSv2_DFC_BIX4_G10R5P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3



python main.py --arch HANRSv3 --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HANRSv3_DFC_BIX4_G10R5P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3

python main.py --arch HANRS --scale 4 --num_rg 10 --num_rcab 8 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HANRS_DFC_BIX4_G10R8P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3     

18941025

python main.py --arch HPANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HPANRS_DFC_BIX4_G10R5P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3

python main.py --arch HPANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HPANRS_pls_BIX4_G10R5P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset pleiades --pre_trained --contEpoch 5

python main.py --arch HSCNRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HSCNRS_DFC_BIX4_G10R5P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3

python main.py --arch HSCNRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HSCNRS_pls_BIX4_G10R5P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset pleiades


python main.py --arch HSCANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/HSCANRS_DFC_BIX4_G10R5P64S1/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3

python main.py --arch ABPN --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --outputs_dir "./checkpoint/ABPNv5_DFC_BIX4/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset WV3

### Test

Output results consist of restored images by the BICUBIC and the RCAN.

```bash
python example.py --scale 2 \
               --num_rg 10 \
               --num_rcab 20 \
               --num_features 64 \
               --weights_path "" \
               --image_path "" \
               --outputs_dir ""                              
```


python example.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --weights_path "./checkpoint/RCAN_BIX4_G10R20P48/RCAN_epoch_19.pth" --image_path "dataset/Set5/HR/" --outputs_dir "./result/RCAN_BIX4_G10R20P48/"

python example.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --weights_path "./checkpoint/RCAN_DFC_BIX4_G10R20P64/RCAN_epoch_9.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/RCAN_DFC_BIX4_G10R20P64_epoch9/"
parameters: 15887779

python example.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --weights_path "./checkpoint/RCAN_pls_BIX4_G10R20P64/RCAN_epoch_16.pth" --image_path "./result/pleiades_test/" --outputs_dir "./result/RCAN_pls_BIX4_G10R20P64_epoch16/"
parameters: 15887779


python example.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --weights_path "./checkpoint/HAN_BIX4_G10R20P48/HAN_epoch_19.pth" --image_path "dataset/Set5/HR/" --outputs_dir "./result/HAN_BIX4_G10R20P48/"


python example.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --weights_path "./checkpoint/HAN_DFC_BIX4_G10R20P64/HAN_epoch_9.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HAN_DFC_BIX4_G10R20P64_epoch9/"


python example.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --weights_path "./checkpoint/HAN_pleiades_BIX4_G10R20P64/HAN_epoch_9.pth" --image_path "D:/Dataset/Pleiades/test_pleiades/" --outputs_dir "./result/HAN_pleiades_BIX4_G10R20P64_epoch9/"

parameters: 16071745

python example.py --arch HAN --scale 4 --num_rg 5 --num_rcab 20 --num_features 64 --weights_path "./checkpoint/HAN_DFC_BIX4_G5R20P64/HAN_epoch_19.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HAN_DFC_BIX4_G5R20P64_epoch19/"

python example.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --weights_path "./checkpoint/HAN_pls_BIX4_G10R20P64/HAN_epoch_18.pth" --image_path "./result/pleiades_test/" --outputs_dir "./result/HAN_pls_BIX4_G10R20P64_epoch18/"

python example.py --arch HANRS --scale 4 --num_rg 5 --num_rcab 3 --res_scale 0.2 --num_features 64 --weights_path "./checkpoint/HANRS_DFC_BIX4_G5R3P64/HANRS_epoch_19.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HANRS_DFC_BIX4_G5R3P64_epoch19/"

python example.py --arch HANRS --scale 4 --num_rg 5 --num_rcab 3 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRS_DFC_BIX4_G5R3P64S1/HANRS_epoch_18.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HANRS_DFC_BIX4_G5R3P64S1_epoch18/"

python example.py --arch HANRS --scale 4 --num_rg 10 --num_rcab 3 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRS_DFC_BIX4_G10R3P64S1/HANRS_epoch_18.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HANRS_DFC_BIX4_G10R3P64S1_epoch18/"

python example.py --arch HANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRS_DFC_BIX4_G10R5P64S1/HANRS_epoch_0.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HANRS_DFC_BIX4_G10R5P64S1_epoch0/" 

python example.py --arch HANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRS_DFC_BIX4_G10R5P64S1_TV/HANRS_epoch_19.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HANRS_DFC_BIX4_G10R5P64S1_TV_epoch19/"


python example.py --arch HANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRS_pls_BIX4_G10R5P64S1_TV2/HANRS_epoch_14.pth" --image_path "./result/pleiades_test/" --outputs_dir "./result/HANRS_pleiades_BIX4_G10R5P64S1_TV_epoch14/"



python example.py --arch HANRS_Hetconv --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRS_Hetconv_DFC_BIX4_G10R5P64S1/HANRS_Hetconv_epoch_9.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HANRS_Hetconv_DFC_BIX4_G10R5P64S1_epoch9/"

python example.py --arch HANRS_Hetconv --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRS_Hetconv_pls_BIX4_G10R5P64S1/HANRS_Hetconv_epoch_19.pth" --image_path "./result/pleiades_test/" --outputs_dir "./result/HANRS_Hetconv_pleiades_BIX4_G10R5P64S1_epoch19/"

python example.py --arch HANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRS_pls_BIX4_G10R5P64S1/HANRS_epoch_17.pth" --image_path "./result/pleiades_test/" --outputs_dir "./result/HANRS_pleiades_BIX4_G10R5P64S1_epoch17/"     

parameters: 12282345

python example.py --arch HANRSv2 --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRSv2_DFC_BIX4_G10R5P64S1/HANRSv2_epoch_2.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HANRSv2_DFC_BIX4_G10R5P64S1_epoch2/" 

parameters: 12319209

python example.py --arch HANRSv3 --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRSv3_DFC_BIX4_G10R5P64S1/HANRSv3_epoch_15.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HANRSv3_DFC_BIX4_G10R5P64S1_epoch15/"

python example.py --arch HANRSv3 --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRSv3_pls_BIX4_G10R5P64S1/HANRSv3_epoch_19.pth" --image_path "./result/pleiades_test/" --outputs_dir "./result/HANRSv3_pleiades_BIX4_G10R5P64S1_epoch19/"

parameters: 12282319

python example.py --arch HANRS --scale 4 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HANRS_DFC_BIX4_G5R10P64S1/HANRS_epoch_17.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HANRS_DFC_BIX4_G5R10P64S1_epoch17/"
parameters: 11913385

python example.py --arch HPANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HPANRS_DFC_BIX4_G10R5P64S1/HPANRS_epoch_15.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HPANRS_DFC_BIX4_G10R5P64S1_epoch15/"


python example.py --arch HPANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HPANRS_pls_BIX4_G10R5P64S1/HPANRS_epoch_19.pth" --image_path "./result/pleiades_test/" --outputs_dir "./result/HPANRS_pleiades_BIX4_G10R5P64S1_epoch19/"

parameters: 12461345

python example.py --arch HSCNRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HSCNRS_DFC_BIX4_G10R5P64S1/HSCNRS_epoch_10.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HSCNRS_DFC_BIX4_G10R5P64S1_epoch10/"


python example.py --arch HSCNRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HSCNRS_pls_BIX4_G10R5P64S1/HSCNRS_epoch_16.pth" --image_path "./result/pleiades_test/" --outputs_dir "./result/HSCNRS_pleiades_BIX4_G10R5P64S1_epoch16/"

parameters: 14098145


python example.py --arch HSCANRS --scale 4 --num_rg 10 --num_rcab 5 --res_scale 1 --num_features 64 --weights_path "./checkpoint/HSCANRS_DFC_BIX4_G10R5P64S1/HSCANRS_epoch_6.pth" --image_path "D:/Dataset/DFC2019/Train-Track1-RGB/test_rename/" --outputs_dir "./result/HSCANRS_DFC_BIX4_G10R5P64S1_epoch6/"