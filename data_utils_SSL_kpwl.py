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

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        d_wm = {}
        for line in l_meta:
            
            #  key,label = line.strip().split()
            #  _, key, _, _, label, wm_label= line.strip().split()
             key, label, _,wm_label= line.strip().split()
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
             d_wm[key]=int(wm_label)
        
        return d_meta,file_list,d_wm
    
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
			

class Dataset_ASVspoof2019_train(Dataset):
	def __init__(self,args,list_IDs, labels, base_dir,algo,wm_labels=None):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.wm_labels = wm_labels if wm_labels is not None else {}
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
            wm_label = int(self.wm_labels.get(utt_id, 0))
            return x_inp, target, wm_label

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
            x_inp = Tensor(X_pad).unsqueeze(-1)
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
            waveform, fs = torchaudio.load(self.base_dir + '' + utt_id + '.flac')

            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                waveform = resampler(waveform)
                fs = 16000

            waveform = waveform[0].numpy()
            X_pad = pad(waveform, self.cut)
            x_inp = Tensor(X_pad).unsqueeze(-1)
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
                    waveform, fs = torchaudio.load(self.base_dir + utt_id+".flac")
                except RuntimeError:
                    waveform, fs = torchaudio.load(self.base_dir + utt_id+".wav")

            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                waveform = resampler(waveform)
                fs = 16000

            waveform = waveform[0].numpy()
            X_pad = pad(waveform, self.cut)
            x_inp = Tensor(X_pad).unsqueeze(-1)
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