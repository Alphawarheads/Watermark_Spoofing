#!/usr/bin/env python
import sys, os
sys.path.append(os.path.dirname(__file__))
import sys, os.path
import numpy as np
import pandas
import eval_metrics_DF as em
from glob import glob



def strip_ext(col):
    s = col.astype("string").str.strip()
    # 仅去掉结尾处的后缀，大小写不敏感
    return s.str.replace(r'(?i)\.(wav|flac|mp3|txt)$', '', regex=True)

phase = "eval"

cm_key_file = "/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/ASVspoof_DF_cm_protocols/ITW_systems.txt"


def eval_to_score_file(score_file, cm_key_file):
    
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    

    # 如果你的第一列还是数字索引：
    cm_data[0] = strip_ext(cm_data[0])
    submission_scores[0] = strip_ext(submission_scores[0])
    # print(cm_data)
    # print(submission_scores)
    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    if len(submission_scores.columns) > 2:
        print('CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces.' % len(submission_scores.columns))
        exit(1)
            
    cm_scores = submission_scores.merge(cm_data, left_on=0, right_on=0, how='inner')  # check here for progress vs eval set
    # print(cm_scores)
    # print("key总数:", len(cm_data[0].unique()))
    # print("score总数:", len(submission_scores[0].unique()))
    # print("交集数量:", len(set(cm_data[0]) & set(submission_scores[0])))

    bona_cm = cm_scores[cm_scores["1_y"] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores["1_y"] == 'spoof']['1_x'].values
    # print(bona_cm,spoof_cm)
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    out_data = "eer: %.2f\n" % (100*eer_cm)
    print(out_data)
    return eer_cm
def evaluate_all_files_in_folder(folder_path: str, cm_key_file="/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/ASVspoof_DF_cm_protocols/ITW_systems.txt"):
    """
    遍历文件夹中所有以 _LA.txt 结尾的文件，并依次调用 eval_to_score_file 评估

    参数:
        folder_path: 包含提交结果的文件夹路径
        cm_key_file: 固定的 key 文件路径
    """
    # 列出所有 _LA.txt 文件
    all_files = [f for f in os.listdir(folder_path) if f.endswith("_ITW.txt")]
    all_files.sort()

    if not all_files:
        print("未找到任何 _LA.txt 文件")
        return

    for filename in all_files:
        submit_path = os.path.join(folder_path, filename)
        print(f"Evaluating: {submit_path}")
        try:
            eer = eval_to_score_file(submit_path, cm_key_file)
            # print(f"  -> EER = {eer:.4f}")
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
    print(1)