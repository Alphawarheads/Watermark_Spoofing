#!/usr/bin/env python
import sys, os
sys.path.append(os.path.dirname(__file__))
import sys, os.path
import numpy as np
import pandas
import eval_metric_LA as em
from glob import glob



truth_dir = "/public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/LA-keys/"
phase = "eval"

asv_key_file = os.path.join(truth_dir, 'LA/ASV/trial_metadata.txt')
asv_scr_file = os.path.join(truth_dir, 'LA/ASV/ASVTorch_Kaldi/score.txt')
cm_key_file = os.path.join(truth_dir, 'LA/CM/trial_metadata.txt')


Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
    'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
    'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
}


def load_asv_metrics():
    # Load organizers' ASV scores
    asv_key_data = pandas.read_csv(asv_key_file, sep=' ', header=None)
    asv_scr_data = pandas.read_csv(asv_scr_file, sep=' ', header=None)[asv_key_data[7] == phase]
    idx_tar = asv_key_data[asv_key_data[7] == phase][5] == 'target'
    idx_non = asv_key_data[asv_key_data[7] == phase][5] == 'nontarget'
    idx_spoof = asv_key_data[asv_key_data[7] == phase][5] == 'spoof'

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scr_data[2][idx_tar]
    non_asv = asv_scr_data[2][idx_non]
    spoof_asv = asv_scr_data[2][idx_spoof]
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=False):
    bona_cm = cm_scores[cm_scores[5]=='bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5]=='spoof']['1_x'].values

    if invert==False:
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = em.compute_eer(-bona_cm, -spoof_cm)[0]

    if invert==False:
        tDCF_curve, _ = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)
    else:
        tDCF_curve, _ = em.compute_tDCF(-bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm


def eval_to_score_file(score_file, cm_key_file=cm_key_file):
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics()
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)

    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    # check here for progress vs eval set
    cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0, right_on=1, how='inner')
    min_tDCF, eer_cm = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv)

    out_data = "min_tDCF: %.4f\n" % min_tDCF
    out_data += "eer: %.2f\n" % (100*eer_cm)
    print(out_data, end="")

    # just in case that the submitted file reverses the sign of positive and negative scores
    min_tDCF2, eer_cm2 = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=True)

    if min_tDCF2 < min_tDCF:
        print(
            'CHECK: we negated your scores and achieved a lower min t-DCF. Before: %.3f - Negated: %.3f - your class labels are swapped during training... this will result in poor challenge ranking' % (
            min_tDCF, min_tDCF2))

    if min_tDCF == min_tDCF2:
        print(
            'WARNING: your classifier might not work correctly, we checked if negating your scores gives different min t-DCF - it does not. Are all values the same?')

    return min_tDCF
def evaluate_all_files_in_folder(folder_path: str, cm_key_file= "/public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/LA-keys/LA/CM/trial_metadata.txt"):
    """
    遍历文件夹中所有以 _LA.txt 结尾的文件，并依次调用 eval_to_score_file 评估

    参数:
        folder_path: 包含提交结果的文件夹路径
        cm_key_file: 固定的 key 文件路径
    """
    # 列出所有 _LA.txt 文件
    all_files = [f for f in os.listdir(folder_path) if f.endswith("_LA21.txt")]
    all_files.sort()

    if not all_files:
        print("未找到任何 _LA21.txt 文件")
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
    print(1)