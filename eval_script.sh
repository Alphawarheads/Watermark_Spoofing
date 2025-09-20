#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=8
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=16
#SBATCH --partition=kshdnormal
#SBATCH --time=5-00:00:00
#SBATCH --chdir=/public/home/qinxy/zhangzs/Watermark_Spoofing
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append
set -eo pipefail
mkdir -p logs

module purge
source /public/home/qinxy/anaconda3/bin/activate base 
conda activate SLS

module rm  compiler/rocm/dtk-22.10.1
module load compiler/rocm/dtk-23.04

export LD_LIBRARY_PATH=~/anaconda3/pkgs/openmpi-4.0.2-hb1b8bf9_1/lib/:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0

export NCCL_IB_DISABLE=1


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip   
NODE_RANK=$SLURM_NODEID
echo NODE_RANK: $NODE_RANK
COMMENT=test11_ori
###########################################LA21###########################################
#original

# TRACK=LA
# PROTOCOLS=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/ASVspoof_DF_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt
# DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/ASVspoof2021_LA_eval/flac/
# EVAL_OUTPUT=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test2/${COMMENT}_train_eval_ori_SLS_LA21.txt
#seen data

# TRACK=LA
# PROTOCOLS=/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged/LA21/protocol_wm_75_only.txt 
# DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/LA21_100/ 
# EVAL_OUTPUT=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test2/${COMMENT}_train_eval_wm_SLS_LA21.txt
#unseendata

# TRACK=LA
# PROTOCOLS=/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/LA21/split/protocol_wm_75_only.txt
# DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/LA21_phase2/ 
# EVAL_OUTPUT=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test2/${COMMENT}_train_eval_wm_SLS_LA21.txt


###########################################In The Wild###########################################

# TRACK=In-the-Wild
# PROTOCOLS=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/ASVspoof_DF_cm_protocols/in_the_wild.eval.txt
# DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/release_in_the_wild/
# EVAL_OUTPUT=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test2/${COMMENT}_train_eval_ori_SLS_ITW.txt

# TRACK=In-the-Wild
# PROTOCOLS=/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged/ITW/protocol_wm_75_only.txt
# DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/ITW_100/
# EVAL_OUTPUT=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test2/${COMMENT}_train_eval_wm_SLS_ITW.txt


# TRACK=In-the-Wild
# PROTOCOLS=/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/ITW/split/protocol_wm_75_only.txt
# DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/ITW_phase2/
# EVAL_OUTPUT=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test2/${COMMENT}_train_eval_wm_SLS_ITW.txt
########################################### Deep Fake 2021 ###########################################

# TRACK=DF
# PROTOCOLS=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt
# DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/ASVspoof2021_DF_eval/flac/
# EVAL_OUTPUT=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test2/${COMMENT}_train_eval_ori_SLS_DF21.txt

# TRACK=DF
# PROTOCOLS=/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged/DF21/protocol_wm_75_only.txt
# DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/DF21_100/
# EVAL_OUTPUT=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test2/${COMMENT}_train_eval_wm_SLS_DF21.txt

# TRACK=DF
# PROTOCOLS=/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/split/protocol_wm_75_only.txt
# DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/DF21_phase2/
# EVAL_OUTPUT=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test2/${COMMENT}_train_eval_wm_SLS_DF21.txt
########################################### Deep Fake 2021 ###########################################


srun torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  --rdzv_id=$RANDOM \
  main_multinodes.py --track=${TRACK} --is_eval --eval \
  --model_path=/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/Submissions/4dcu_enhanced.pth \
  --protocols_path=${PROTOCOLS} \
  --database_path=${DATABASE_PATH} \
  --eval_output=${EVAL_OUTPUT}

  
  