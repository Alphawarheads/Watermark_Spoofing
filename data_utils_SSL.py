import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import random
import torchaudio
from typing import List, Dict, Tuple
import os

def _parse_proto_line(line: str) -> Tuple[str, str, int]:
    """
    解析协议行，返回 (utt_id, cls_label_str, wm_label_int)
    兼容：
      1) ASVspoof 5列：_ key _ _ label [wm?]   -> utt=col[1], label=col[-1]或col[-2]
      2) 简单两列：utt label [wm?]
      3) 竖线分隔：path|label[|wm]
    wm 缺省则为 0
    """
    line = line.strip()
    if '|' in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) == 2:
            utt, lab = parts
            wm = 0
        else:
            utt, lab, wm = parts[0], parts[1], int(parts[2])
        # 去掉可能的开头 ./ 
        if utt.startswith('./'):
            utt = utt[2:]
        return utt, lab, wm

    cols = line.split()
    if len(cols) >= 6 and cols[-1] in ('0', '1'):
        # 末尾带 wm 列
        utt = cols[1] if len(cols) >= 2 else cols[0]
        lab = cols[-2]
        wm  = int(cols[-1])
        return utt, lab, wm
    elif len(cols) >= 5:
        utt = cols[1]
        lab = cols[-1]
        wm  = 0
        return utt, lab, wm
    elif len(cols) >= 2:
        utt = cols[0]
        lab = cols[1]
        wm  = 0
        return utt, lab, wm
    else:
        raise ValueError(f"Protocol line not understood: {line}")


def genSpoof_list_multi(
    datasets: List[Dict[str, str]]
) -> Tuple[Dict[str, int], List[str], Dict[str, int], Dict[str, str]]:
    """
    datasets: 列表，每个元素形如：
      {"protocol": ".../train.txt", "base_dir": ".../ASVspoof2019_LA_train/flac/"}
    返回：
      d_label:   {uid -> 1(bonafide)/0(spoof)}
      file_list: [uid, ...]，uid 全局唯一（前缀标域）
      d_wm:      {uid -> 0/1}   （协议无 wm 列时自动置 0）
      d_base:    {uid -> base_dir}  每条样本自己的根目录
    说明：
      uid = f"d{di}:{utt_id}"  避免不同数据集 utt 重名
    """
    d_label, d_wm, d_base = {}, {}, {}
    file_list: List[str] = []

    for di, cfg in enumerate(datasets):
        proto = cfg["protocol"]
        base  = cfg["base_dir"]
        assert os.path.isdir(base), f"Base dir not found: {base}"
        with open(proto, 'r') as f:
            for ln in f:
                if not ln.strip():
                    continue
                utt, lab_str, wm = _parse_proto_line(ln)
                uid = f"d{di}:{utt}"
                d_label[uid] = 1 if lab_str.lower() == 'bonafide' else 0
                d_wm[uid]    = int(wm)
                d_base[uid]  = base if base.endswith(os.sep) else base + os.sep
                file_list.append(uid)

    return d_label, file_list, d_wm, d_base




def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            
            #  key,label = line.strip().split()
             _, key, _, _, label = line.strip().split()
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            #_, key, _, _, _ = line.strip().split()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list




def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	


class Dataset_MultiTrain(Dataset):
    def __init__(
        self,
        args,
        list_IDs,                      # 来自 genSpoof_list_multi 的 file_list (uid 列表)
        labels,                        # d_label  (uid -> 0/1)
        base_dir_map,                  # d_base   (uid -> base_dir)
        wm_labels=None,                # d_wm     (uid -> 0/1)
        algo: int = 0,
        try_exts=(".flac", ".wav", "") # 依次尝试
    ):
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir_map = base_dir_map
            self.wm_labels = wm_labels or {}
            self.args = args
            self.algo = algo
            self.cut = 64600
            self.try_exts = try_exts

    def __len__(self):
            return len(self.list_IDs)

    def _load_wave(self, base_dir: str, utt_id_raw: str):
            """
            uid: d{di}:{utt}; 这里只取冒号后的原始 utt
            """
            utt = utt_id_raw.split(":", 1)[1]
            # 若 utt 已经自带后缀，直接尝试绝对路径
            if utt.endswith(".wav") or utt.endswith(".flac"):
                path = base_dir + utt
                wav, fs = torchaudio.load(path)
                return wav, fs

            # 否则依次尝试补后缀
            last_err = None
            for ext in self.try_exts:
                try:
                    path = base_dir + utt + ext
                    wav, fs = torchaudio.load(path)
                    return wav, fs
                except Exception as e:
                    last_err = e
                    continue
            raise last_err if last_err is not None else RuntimeError(f"Cannot load {utt} under {base_dir}")

    def __getitem__(self, index):
            uid = self.list_IDs[index]
            base_dir = self.base_dir_map[uid]

            waveform, fs = self._load_wave(base_dir, uid)

            # 重采样到 16k
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                waveform = resampler(waveform)
                fs = 16000

            # 取单通道
            waveform = waveform[0].numpy()

            # RawBoost
            Y = process_Rawboost_feature(waveform, fs, self.args, self.algo)

            # 截断/补齐
            X_pad = pad(Y, self.cut)
            x_inp = Tensor(X_pad)

            cls_target = self.labels[uid]
            wm_target  = int(self.wm_labels.get(uid, 0))
            # 返回 (wav, cls_label, wm_label)
            # return x_inp, cls_target, wm_target
            return x_inp,cls_target

class Dataset_ASVspoof2019_train(Dataset):
	def __init__(self,args,list_IDs, labels, base_dir,algo):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.algo=algo
            self.args=args
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
           return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            waveform, fs = torchaudio.load(self.base_dir + '' + utt_id + '.flac')

            # 若不是 16000 Hz，则重采样
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                waveform = resampler(waveform)
                fs = 16000

            # 只取单通道（第一个通道）
            waveform = waveform[0].numpy()

            Y = process_Rawboost_feature(waveform, fs, self.args, self.algo)
            X_pad = pad(Y, self.cut)
            x_inp = Tensor(X_pad).unsqueeze(-1)
            target = self.labels[utt_id]
            return x_inp, target

class Dataset_train(Dataset):
	def __init__(self,args,list_IDs, labels, base_dir,algo):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.algo=algo
            self.args=args
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
           return len(self.list_IDs)


	def __getitem__(self, index):
            utt_id = self.list_IDs[index]
            waveform, fs = torchaudio.load(self.base_dir + utt_id+"")

            # 若不是 16000 Hz，则重采样
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                waveform = resampler(waveform)
                fs = 16000

            # 只取单通道（第一个通道）
            waveform = waveform[0].numpy()

            Y = process_Rawboost_feature(waveform, fs, self.args, self.algo)
            X_pad = pad(Y, self.cut)
            x_inp = Tensor(X_pad)
            target = self.labels[utt_id]
            return x_inp, target
            
            
class Dataset_ASVspoof2021_eval(Dataset):
	def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            utt_id = self.list_IDs[index].split(" ")[0]
            try:
                waveform, fs = torchaudio.load(self.base_dir + '' + utt_id + '.flac')
            except RuntimeError:
                waveform, fs = torchaudio.load(self.base_dir + '' + utt_id + '.wav')

            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                waveform = resampler(waveform)
                fs = 16000

            waveform = waveform[0].numpy()
            X_pad = pad(waveform, self.cut)
            x_inp = Tensor(X_pad)
            return x_inp, utt_id



class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
               '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
            utt_id = self.list_IDs[index]
            utt_id = self.list_IDs[index].split(" ")[0]
            try:
                waveform, fs = torchaudio.load(self.base_dir + utt_id+"")
            except RuntimeError:
                try:
                    waveform, fs = torchaudio.load(self.base_dir + utt_id + ".flac")
                except RuntimeError:
                    waveform, fs = torchaudio.load(self.base_dir + utt_id + ".wav")


            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                waveform = resampler(waveform)
                fs = 16000

            waveform = waveform[0].numpy()
            X_pad = pad(waveform, self.cut)
            x_inp = Tensor(X_pad)
            return x_inp, utt_id




        #--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature