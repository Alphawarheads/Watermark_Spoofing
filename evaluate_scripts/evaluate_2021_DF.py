#!/usr/bin/env python
import sys, os
sys.path.append(os.path.dirname(__file__))
import sys, os.path
import numpy as np
import pandas
import eval_metrics_DF as em
from glob import glob

# if len(sys.argv) != 4:
#     print("CHECK: invalid input arguments. Please read the instruction below:")
#     print(__doc__)
#     exit(1)


truth_dir = "/public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/DF-keys-full/keys/DF"
phase = "eval"

cm_key_file = os.path.join(truth_dir, 'CM/trial_metadata.txt')


def eval_to_score_file(score_file, cm_key_file="/public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/DF-keys-full/keys/DF/CM/trial_metadata.txt"):

    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    if len(submission_scores.columns) > 2:
        print('CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces.' % len(submission_scores.columns))
        exit(1)
            
    cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0, right_on=1, how='inner')  # check here for progress vs eval set
    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    out_data = "eer: %.2f\n" % (100*eer_cm)
    print(out_data)
    return eer_cm
def evaluate_all_files_in_folder(folder_path: str, cm_key_file="/public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/DF-keys-full/keys/DF/CM/trial_metadata.txt" ):
    """
    遍历文件夹中所有以 _LA.txt 结尾的文件，并依次调用 eval_to_score_file 评估

    参数:
        folder_path: 包含提交结果的文件夹路径
        cm_key_file: 固定的 key 文件路径
    """
    # 列出所有 _LA.txt 文件
    all_files = [f for f in os.listdir(folder_path) if f.endswith("_DF21.txt")]
    all_files.sort()

    if not all_files:
        print("未找到任何 _LA.txt 文件")
        return

    for filename in all_files:
        submit_path = os.path.join(folder_path, filename)
        print(f"Evaluating: {submit_path}")
        try:
            eer = eval_to_score_file(submit_path, cm_key_file)
            print(f"  -> EER = {eer:.4f}")
        except Exception as e:
            print(f"  [Error] 评估 {submit_path} 时出错: {str(e)}")

if __name__ == "__main__":

    # if not os.path.isfile(submit_file):
    #     print("%s doesn't exist" % (submit_file))
    #     exit(1)
        
    # if not os.path.isdir(truth_dir):
    #     print("%s doesn't exist" % (truth_dir))
    #     exit(1)

    # if phase != 'progress' and phase != 'eval' and phase != 'hidden_track':
    #     print("phase must be either progress, eval, or hidden_track")
    #     exit(1)

    # _ = eval_to_score_file(submit_file, cm_key_file)
    # evaluate_all_files_in_folder("")
    print(1)