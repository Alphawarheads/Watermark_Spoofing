import torch
from typing import List
import typing as tp
import random
from Attack_Simulator_B import (
    attack_random_noise_torch, attack_sample_suppression_torch, attack_low_pass_filter_torch,
    attack_median_filter_torch, attack_resample_torchaudio, attack_amplitude_scaling_torch,
    attack_lossy_compression_torch, attack_quantization_torch, attack_echo_torch,
    attack_time_stretch_torch, attack_reverberation_torch, attack_aac_compression_torch,
    attack_speed_torch, attack_band_filter_torch, attack_pitch_shift_ffmpeg_asetrate_atempo,
    attack_opus_compression_torch, align_with_offset_torch, pink_noise, highpass_filter,
    smooth, boost_audio, duck_audio, attack_crop_like, attack_speed_torch_pitch,
    apply_random_phase_shift_torch, attack_spec_augment_torch
)


def apply_configurable_attacks(watermarked_signals: torch.Tensor, sample_rate: int, attack_names: List[str],mask: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
    """根据攻击列表顺序依次对信号进行扰动。"""
    ffmpeg_path = "/public/home/qinxy/bin/ffmpeg"

    for attack_name in attack_names:
        try:
            if attack_name == 'noise':
                snr_db = 0.001
                signal= attack_random_noise_torch(watermarked_signals, snr_db, mask)
            elif attack_name == 'suppression':
                p = random.uniform(0.005, 0.02)
                signal = attack_sample_suppression_torch(watermarked_signals, p=p)
            elif attack_name == 'lp':
                cutoff = random.uniform(3000, 6000)
                signal = attack_low_pass_filter_torch(watermarked_signals, sample_rate, cutoff=cutoff)
            elif attack_name == 'pink':
                noise_std = 0.01
                signal= pink_noise(watermarked_signals, noise_std, mask)
            elif attack_name == 'hp':
                cutoff_freq = 500
                signal = highpass_filter(watermarked_signals, cutoff_freq, sample_rate)
            elif attack_name == 'smooth':
                signal = smooth(watermarked_signals,(2,10))
            elif attack_name == 'boost':
                signal = boost_audio(watermarked_signals,20)
            elif attack_name == 'duck':
                signal = duck_audio(watermarked_signals,20)
            elif attack_name == 'median':
                kernel_size = random.choice([3, 5])
                signal = attack_median_filter_torch(watermarked_signals, kernel_size=kernel_size)
            elif attack_name == 'resample':
                signal = attack_resample_torchaudio(watermarked_signals, sample_rate)
            elif attack_name == 'amp':
                factor = random.uniform(0.3, 1.2)
                signal = attack_amplitude_scaling_torch(watermarked_signals, factor=factor)
            elif attack_name == 'quant':
                levels = random.choice([256, 512, 1024])
                signal = attack_quantization_torch(watermarked_signals, levels=levels)
            elif attack_name == 'echo':
                delay_ms = (0.1, 0.5)
                factor = (0.1, 0.5)
                signal= attack_echo_torch(watermarked_signals, delay_ms, factor, sample_rate, mask)
            elif attack_name == 'stretch':
                rate = random.uniform(0.85, 1.15)
                signal = attack_time_stretch_torch(watermarked_signals, rate=rate, sample_rate=sample_rate)
            elif attack_name == 'lossy': # MP3
                bitrate = random.choice([32, 64, 96, 128])
                signal = attack_lossy_compression_torch(watermarked_signals, sample_rate, bitrate=bitrate, ffmpeg_path=ffmpeg_path)
            elif attack_name == 'reverb':
                rir_length = sample_rate // 100  # 0.1 秒
                current_device = watermarked_signals.device # 获取当前批次的设备
                # 在正确的设备上生成 RIR
                rir_tensor = torch.exp(-torch.linspace(0, 1, steps=rir_length, device=current_device, dtype=watermarked_signals.dtype))
                if rir_tensor is not None and rir_tensor.numel() > 0:
                    signal = attack_reverberation_torch(watermarked_signals, rir=rir_tensor)
                    del rir_tensor
                else:
                    print(f"Warning: RIR tensor not provided or empty for attack '{attack_name}'. Using original signal.")
                    signal = watermarked_signals # 保持不变
            elif attack_name == 'aac':
                signal = attack_aac_compression_torch(watermarked_signals, sample_rate, ffmpeg_path=ffmpeg_path)
            elif attack_name == 'opus':
                signal = attack_opus_compression_torch(watermarked_signals, sample_rate, ffmpeg_path=ffmpeg_path)
            elif attack_name == 'speed':
                speed_factor = (0.80, 1.20)
                # factor = random.choice(speed_factor)
                signal= attack_speed_torch(watermarked_signals, speed_factor).detach()
            elif attack_name == 'pitch':
                signal = attack_pitch_shift_ffmpeg_asetrate_atempo(watermarked_signals)
            elif attack_name == 'band':
                cutoff_freq_low=random.randint(300,400)
                cutoff_freq_high=random.randint(7000,9000)
                signal = attack_band_filter_torch(watermarked_signals, cutoff_freq_low, cutoff_freq_high, sample_rate)
            elif attack_name == 'speed_pitch':
                signal = attack_speed_torch_pitch(watermarked_signals)
            elif attack_name == 'phaseshift':
                signal = apply_random_phase_shift_torch(watermarked_signals)
            elif attack_name == 'specaug':
                signal = attack_spec_augment_torch(watermarked_signals, sample_rate)
            elif attack_name == 'original':
                signal = watermarked_signals
            else:
                    # 这个分支理论上不会执行，因为 attack_name 是从列表中选的
                print(f"Error: Reached unexpected attack name '{attack_name}'.")
                signal = watermarked_signals
        except Exception as e:
            print(f"[!] Failed on attack {attack_name}: {e}")
            signal = watermarked_signals
    return signal
