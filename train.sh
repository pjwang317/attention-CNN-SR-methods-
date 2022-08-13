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
#SBATCH --job-name=MSRFBNs2
#SBATCH --nodes=1
# #SBATCH --nodelist=ai05
#SBATCH --constraint=tesla_t4
#SBATCH --gres=gpu:1
# #SBATCH --gres=gpu:tesla_k80=1

##SBATCH --partition=ai
##SBATCH --account=ai
##SBATCH --qos=ai

#SBATCH --time=1-00:00:00
#SBATCH --partition=mid
#SBATCH --mem=50G
# #SBATCH --ntasks-per-node=10
# #SBATCH --mem-per-cpu=30G
#SBATCH --output=p-python_MSRFBNs2.out
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


#python main_trail_corona.py --arch RDN --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/RDN_BIX8_corona/" --patch_size 32 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Corona


#python main_trail_corona.py --arch DDBPN --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/DDBPN_BIX8_corona/" --patch_size 32 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Corona


python main_trail_corona.py --arch CAFRN --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/CAFRN_BIX8_corona/" --patch_size 32 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Corona



#python main_trail.py --arch RCAN --scale 2 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/RSRCAN_BIX2/" --patch_size 64 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch RCAN --scale 4 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/RSRCAN_BIX4/" --patch_size 32 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch RCAN --scale 8 --num_rg 5 --num_rcab 10 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/RSRCAN_BIX8/" --patch_size 16 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata


#python main_trail.py --arch DDBPN --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/DDBPN_BIX8/" --patch_size 16 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata



#python main_trail.py --arch DDBPN --scale 4 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/DDBPN_BIX4/" --patch_size 32 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata



#python main_trail.py --arch DDBPN --scale 2 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/DDBPN_BIX2/" --patch_size 64 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata



#python main_trail.py --arch HAN --scale 2 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/HAN_BIX2/" --patch_size 64 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata


#python main_trail.py --arch HAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/HAN_BIX4/" --patch_size 32 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata



#python main_trail.py --arch HAN --scale 8 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/HAN_BIX8/" --patch_size 16 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch SRFBN --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/SRFBN_BIX8/" --patch_size 16 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata


#python main_trail.py --arch SAN --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/SAN_BIX8/" --patch_size 16 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata


#python main_trail.py --arch CAFRN --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/CAFRN_BIX8/" --patch_size 16 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch MHAN --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/MHAN_BIX8/" --patch_size 16 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch RDN --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/RDN_BIX8/" --patch_size 16 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata


#python main_trail.py --arch LGCNet --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/LGCNet_BIX8/" --patch_size 16 --batch_size 2 --num_epochs 100 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch SRCNN --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/SRCNN_BIX8/" --patch_size 16 --batch_size 8 --num_epochs 100 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch VDSR --scale 8 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/VDSR_BIX8/" --patch_size 16 --batch_size 8 --num_epochs 100 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch RCAN --scale 8 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/RCAN_BIX8/" --patch_size 16 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch EEGAN --scale 8 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/EEGAN_BIX8/" --patch_size 16 --batch_size 8 --num_epochs 50 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch NLSN --scale 8 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt8/NLSN_BIX8/" --patch_size 16 --batch_size 8 --num_epochs 50 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata





#python main_trail.py --arch SAN --scale 2 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/SAN_BIX2/" --patch_size 64 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata


#python main_trail.py --arch CAFRN --scale 2 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/CAFRN_BIX2/" --patch_size 64 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch MHAN --scale 2 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/MHAN_BIX2/" --patch_size 64 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch RDN --scale 2 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/RDN_BIX2/" --patch_size 64 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata


#python main_trail.py --arch LGCNet --scale 2 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/LGCNet_BIX2/" --patch_size 64 --batch_size 2 --num_epochs 100 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch SRCNN --scale 2 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/SRCNN_BIX2/" --patch_size 64 --batch_size 8 --num_epochs 100 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch VDSR --scale 2 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/VDSR_BIX2/" --patch_size 64 --batch_size 8 --num_epochs 100 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch RCAN --scale 2 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/RCAN_BIX2/" --patch_size 64 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch EEGAN --scale 2 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/EEGAN_BIX2/" --patch_size 64 --batch_size 8 --num_epochs 50 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch NLSN --scale 2 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt2/NLSN_BIX2/" --patch_size 64 --batch_size 8 --num_epochs 50 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata































#python main_trail.py --arch SAN --scale 4 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/SAN_BIX4/" --patch_size 32 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata


#python main_trail.py --arch CAFRN --scale 4 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/CAFRN_BIX4/" --patch_size 32 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch MHAN --scale 4 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/MHAN_BIX4/" --patch_size 32 --batch_size 4 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch RDN --scale 4 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/RDN_BIX4/" --patch_size 32 --batch_size 2 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata


#python main_trail.py --arch LGCNet --scale 4 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/LGCNet_BIX4/" --patch_size 32 --batch_size 2 --num_epochs 100 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch SRCNN --scale 4 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/SRCNN_BIX4/" --patch_size 32 --batch_size 8 --num_epochs 100 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch VDSR --scale 4 --num_rg 10 --num_rcab 15 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/VDSR_BIX4/" --patch_size 32 --batch_size 8 --num_epochs 100 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch RCAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/RCAN_BIX4/" --patch_size 32 --batch_size 8 --num_epochs 20 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch EEGAN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/EEGAN_BIX4/" --patch_size 32 --batch_size 8 --num_epochs 50 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata

#python main_trail.py --arch NLSN --scale 4 --num_rg 10 --num_rcab 20 --res_scale 1 --num_features 64 --outputs_dir "./Mdata_ckpt4/NLSN_BIX4/" --patch_size 32 --batch_size 8 --num_epochs 50 --lr 1e-4 --threads 8 --seed 123 --dataset Mdata