#!/bin/bash
#SBATCH --job-name=DF21part12
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1                       
#SBATCH --cpus-per-task=32           
#SBATCH --gres=dcu:4                         
#SBATCH --partition=kshdnormal         
##SBATCH --time=5-00:00:00 
##SBATCH --output=./logs/%x_%j.out          
##SBATCH --error=./logs/%x_%j.err          

export LD_LIBRARY_PATH=/anaconda3/envs/ywliang_py39_audioldm_train/lib/python3.9/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH
# source /public/home/qinxy/anaconda3/bin/activate base 
# conda activate SVDD
source /public/home/qinxy/anaconda3/bin/activate base 


# module load compiler/dtk/24.04
# source /public/home/qinxy/anaconda3/bin/activate base 
conda activate watermark

export MIOPEN_USER_DB_PATH="${SLURM_TMPDIR:-/tmp}/miopen-db-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/miopen-cache-${SLURM_JOB_ID}"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
#!/bin/bash

# === 1. 全攻击模式（科研评估用） ===
python main.py \
    --mode all_attacks \
    --threads 32 \
    --new_protocol_path /public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged/ITW/protocol_wm_75.txt \
    --audio_dir /public/home/qinxy/AudioData/Antispoofing/release_in_the_wild \
    --embedded_path /public/home/qinxy/AudioData/Antispoofing/The-A-Files-master/test1 \
    --output Test.csv \
    --limit -1

# /public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/ASVspoof2021_LA_eval/flac
# /public/home/qinxy/AudioData/Antispoofing/release_in_the_wild

# === 5. 嵌入无攻击模式（按 protocol 跑攻击） ===
python main.py \
    --mode embed_only \
    --threads 32 \
    --new_protocol_path /public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/ori_split/protocol_part_8.txt \
    --audio_dir /public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/ASVspoof2021_DF_eval/flac/ \
    --embedded_path /public/home/qinxy/AudioData/Antispoofing/DF21_phase2 \
    --save_embedded \
    --output watermarked file \
    --limit -1

# /public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/ASVspoof2021_LA_eval/flac
# /public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/ASVspoof2021_DF_eval/flac