#!/bin/bash
# 
# CompecTA (c) 2017
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-
#SBATCH --job-name=THPs4RD
#SBATCH --nodes=1
# #SBATCH --nodelist=ai05
#SBATCH --constraint=tesla_v100
#SBATCH --gres=gpu:1
# #SBATCH --gres=gpu:tesla_k80=1

##SBATCH --partition=ai
##SBATCH --account=ai
##SBATCH --qos=ai

#SBATCH --time=0-00:10:00
#SBATCH --partition=short
#SBATCH --mem=30G
# #SBATCH --ntasks-per-node=10
# #SBATCH --mem-per-cpu=30G
#SBATCH --output=p-python_THPs4RD.out
#SBATCH --mail-type=ALL
# #SBATCH --mail-user=rji19@ku.edu.tr


################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

## Load Python 3.6.3
#echo "Activating Python 3.6.3..."
#srun -N 1 --gres=gpu:1 --mem=200G --time=2:00:00  --pty bash
#R/4.0.2

module load anaconda/3.6
#source activate py3 
source activate torchw_10
#source activate pytf
#source activate tf_deinter
#source activate HW
#cd HW5
#source activate py37
#source activate frupcv
#module load cuda/8.0
#module load cudnn/6.0/cuda-8.0
#module load cuda/9.1
#module load cudnn/7.0.5/cuda-9.1 
#module load cuda/10.0
#module load cudnn/7.6.2/cuda-10.0
#module load cudnn/7.0.6/cuda-10.1
module load cudnn/7.6.5/cuda-10.1   
module load cuda/10.1
#module load gcc/6.3.0
#export OMP_NUM_THREADS=1
#srun -N 1 --constraint=tesla_t4 --gres=gpu:1 --mem=200G --time=2:00:00  --pty bash
#srun --x11=all -N 1 --gres=gpu:1 --mem=200G --time=2:00:00 --pty bash

#cd /scratch/users/rji19/sepconv-pytorch/EQVI_single
#cd /scratch/users/rji19/sepconv-pytorch/EDVR_old
#cd WPJ/BasicSR
#cd WPJ/MFSR/SNet7
#cd
#cd WPJ/RCAN-pytorch
#cd /scratch/users/rji19/EDVR/codes/
#cd /scratch/users/rji19/EDVR/codes/data_scripts/
#cd /scratch/users/rji19/EDVR/datasets/ucf101/
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo
nvidia-smi
#PYTHONPATH="./:${PYTHONPATH}" \
#CUDA_VISIBLE_DEVICES=0 \
#python basicsr/train.py -opt options/train/EDVR/train_EDSC_L_x4_SR_Vimeo_woTSA.yml

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_EQVI_lap_l1.py --config configs/config_train_EQVI_VTSR.py


### Koc project

#python example2.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/HAN_BIX4/HAN_epoch_19.pth" --image_path "../data/test/AksuKestel_npy/" --outputs_dir "./test/test_MSx4/HAN_BIX4_epoch19/"


#python example2.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/RCAN_BIX4/RCAN_epoch_19.pth" --image_path "../data/test/AksuKestel_npy/" --outputs_dir "./test/test_MSx4/RCAN_BIX4_epoch19/"



#python example2.py --arch RDN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/RDN_BIX4/RDN_epoch_19.pth" --image_path "../data/test/AksuKestel_npy/" --outputs_dir "./test/test_MSx4/RDN_BIX4_epoch19/"


#python example2.py --arch DDBPN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/DDBPN_BIX4/DDBPN_epoch_19.pth" --image_path "../data/test/AksuKestel_npy/" --outputs_dir "./test/test_MSx4/DDBPN_BIX4_epoch19/"


#python example2.py --arch EEGAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/EEGAN_BIX4/EEGAN_epoch_49.pth" --image_path "../data/test/AksuKestel_npy/" --outputs_dir "./test/test_MSx4/EEGAN_BIX4_epoch49/"


python example2.py --arch CAFRN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/CAFRN_BIX4/CAFRN_epoch_19.pth" --image_path "../data/test/AksuKestel_npy/" --outputs_dir "./test/test_MSx4/CAFRN_BIX4_epoch19/"





####    Koc project




#python example.py --arch LGCNet --scale 2 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt2/LGCNet_BIX2/LGCNet_epoch_99.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx2/LGCNet_pls_BIX2_epoch99/"











#python example.py --arch RCAN --scale 4 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/RSRCAN_BIX4/RCAN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/RSRCAN_pls_BIX4_epoch19/"


#python example.py --arch RCAN --scale 2 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt2/RSRCAN_BIX2/RCAN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx2/RSRCAN_pls_BIX2_epoch19/"


#python example.py --arch RCAN --scale 8 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt8/RSRCAN_BIX8/RCAN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx8/RSRCAN_pls_BIX8_epoch19/"


#python example.py --arch RCAN --scale 4 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/RSRCAN_BIX4/RCAN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/RSRCAN_WV3_BIX4_epoch19/"


#python example.py --arch RCAN --scale 2 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt2/RSRCAN_BIX2/RCAN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx2/RSRCAN_WV3_BIX2_epoch19/"


#python example.py --arch RCAN --scale 8 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt8/RSRCAN_BIX8/RCAN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx8/RSRCAN_WV3_BIX8_epoch19/"



#python example2.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./ckpt4/HAN_pls_BIX4_G10R20P64S1_new/HAN_epoch_19.pth" --image_path "../data/test/MS_npy/pls_MS_npy/" --outputs_dir "./test/test_MSx4/HAN_pls_BIX4_G10R20P64S1_epoch19_new/"


#python example2.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./ckpt4/RCAN_pls_BIX4_G10R20P64S1_new/RCAN_epoch_19.pth" --image_path "../data/test/MS_npy/pls_MS_npy/" --outputs_dir "./test/test_MSx4/RCAN_pls_BIX4_G10R20P64S1_epoch19_new/"

#python example2.py --arch CAFRN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./ckpt4/CAFRN_pls_BIX4_new/CAFRN_epoch_19.pth" --image_path "../data/test/MS_npy/pls_MS_npy/" --outputs_dir "./test/test_MSx4/CAFRN_pls_BIX4_epoch19_new/"

#python example2.py --arch RDN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./ckpt4/RDN_pls_BIX4_new/RDN_epoch_19.pth" --image_path "../data/test/MS_npy/pls_MS_npy/" --outputs_dir "./test/test_MSx4/RDN_pls_BIX4_epoch19_new/"

#python example2.py --arch SAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./ckpt4/SAN_pls_BIX4_new/SAN_epoch_19.pth" --image_path "../data/test/MS_npy/pls_MS_npy/" --outputs_dir "./test/test_MSx4/SAN_pls_BIX4_epoch19_new/"

#python example2.py --arch MHAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./ckpt4/MHAN_pls_BIX4_new/MHAN_epoch_19.pth" --image_path "../data/test/MS_npy/pls_MS_npy/" --outputs_dir "./test/test_MSx4/MHAN_pls_BIX4_epoch19_new/"

#python example2.py --arch VDSR --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./ckpt4/VDSR_pls_BIX4_new/VDSR_epoch_99.pth" --image_path "../data/test/MS_npy/pls_MS_npy/" --outputs_dir "./test/test_MSx4/VDSR_pls_BIX4_epoch99_new/"


#python example2.py --arch SRCNN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./ckpt4/SRCNN_pls_BIX4_new/SRCNN_epoch_99.pth" --image_path "../data/test/MS_npy/pls_MS_npy/" --outputs_dir "./test/test_MSx4/SRCNN_pls_BIX4_epoch99_new/"


#python example2.py --arch LGCNet --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./ckpt4/LGCNet_pls_BIX4_new/LGCNet_epoch_99.pth" --image_path "../data/test/MS_npy/pls_MS_npy/" --outputs_dir "./test/test_MSx4/LGCNet_pls_BIX4_epoch99_new/"


#python example.py --arch HANRSv5 --scale 4 --num_rg 10 --num_rcab 15 --res_scale 0.2 --num_features 64 --weights_path "./ckpt4/HANRSv5_DFC_BIX4_G10R15P64S.2/HANRSv5_epoch_19.pth" --image_path "../data/test/DFC_test/" --outputs_dir "./test/test_MSx4/HANRSv5_DFC_BIX4_G10R15P64S.2_epoch19/"

#python example.py --arch HANRSv6 --scale 4 --num_rg 10 --num_rcab 15 --res_scale 0.2 --num_features 64 --weights_path "./ckpt4/HANRSv6_DFC_BIX4_G10R15P64S.2/HANRSv6_epoch_19.pth" --image_path "../data/test/DFC_test/" --outputs_dir "./test/test_MSx4/HANRSv6_DFC_BIX4_G10R15P64S.2_epoch19/"


#python example.py --arch NLSN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/NLSN_BIX4/NLSN_epoch_49.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/NLSN_WV3_BIX4_epoch49/"

#python example.py --arch NLSN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/NLSN_BIX4/NLSN_epoch_49.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/NLSN_pls_BIX4_epoch49/"

#python example.py --arch EEGAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/EEGAN_BIX4/EEGAN_epoch_49.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/EEGAN_pls_BIX4_epoch49/"

#python example.py --arch EEGAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/EEGAN_BIX4/EEGAN_epoch_49.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/EEGAN_WV3_BIX4_epoch49/"

#python example.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/HAN_BIX4/HAN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/HAN_WV3_BIX4_epoch19/"

#python example.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/HAN_BIX4/HAN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/HAN_pls_BIX4_epoch49/"

#python example.py --arch MHAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/MHAN_BIX4/MHAN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/MHAN_WV3_BIX4_epoch19/"


#python example.py --arch MHAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/MHAN_BIX4/MHAN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/MHAN_pls_BIX4_epoch19/"

#python example.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/RCAN_BIX4/RCAN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/RCAN_WV3_BIX4_epoch19/"


#python example.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/RCAN_BIX4/RCAN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/RCAN_pls_BIX4_epoch19/"

#python example.py --arch RDN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/RDN_BIX4/RDN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/RDN_WV3_BIX4_epoch19/"


#python example.py --arch RDN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/RDN_BIX4/RDN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/RDN_pls_BIX4_epoch19/"

#python example.py --arch SAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/SAN_BIX4/SAN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/SAN_WV3_BIX4_epoch19/"

#python example.py --arch CAFRN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/CAFRN_BIX4/CAFRN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/CAFRN_WV3_BIX4_epoch19/"

#python example.py --arch CAFRN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/CAFRN_BIX4/CAFRN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/CAFRN_pls_BIX4_epoch19/"


#python example.py --arch RCAN --scale 4 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/RSRCAN_BIX4/RCAN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/RSRCAN_WV3_BIX4_epoch19/"


#python example.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/RSRCAN_BIX4/RSRCAN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/RSRCAN_pls_BIX4_epoch19/"

#python example.py --arch DDBPN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/DDBPN_BIX4/DDBPN_epoch_19.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/DDBPN_WV3_BIX4_epoch19/"

#python example.py --arch DDBPN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/DDBPN_BIX4/DDBPN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/DDBPN_pls_BIX4_epoch19/"


#python example.py --arch VDSR --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/VDSR_BIX4/VDSR_epoch_99.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/VDSR_WV3_BIX4_epoch99/"


#python example.py --arch VDSR --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/VDSR_BIX4/VDSR_epoch_99.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/VDSR_pls_BIX4_epoch99/"


#python example.py --arch SRCNN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/SRCNN_BIX4/SRCNN_epoch_99.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/SRCNN_WV3_BIX4_epoch99/"

#python example.py --arch SRCNN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/SRCNN_BIX4/SRCNN_epoch_99.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/SRCNN_pls_BIX4_epoch99/"

#python example.py --arch LGCNet --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/LGCNet_BIX4/LGCNet_epoch_99.pth" --image_path "../data/test/val_RGB/" --outputs_dir "./test/testx4/LGCNet_WV3_BIX4_epoch99/"

#python example.py --arch LGCNet --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/LGCNet_BIX4/LGCNet_epoch_99.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/LGCNet_pls_BIX4_epoch99/"





















#python example.py --arch SAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --weights_path "./Mdata_ckpt4/SAN_BIX4/SAN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx4/SAN_pls_BIX4_epoch19/"



#python example.py --arch MHAN --scale 2 --num_rg 10 --num_rcab 12 --res_scale 0.2 --num_features 64 --weights_path "./ckpt2/MHAN_pls_BIX2_new3/MHAN_epoch_19.pth" --image_path "../data/test/pls_test/" --outputs_dir "./test/testx2/MHAN_pls_BIX2_epoch19_new3/"