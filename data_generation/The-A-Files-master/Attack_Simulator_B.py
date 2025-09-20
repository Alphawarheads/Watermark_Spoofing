# -*- coding: utf-8 -*-
import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
import torch.nn.functional as F
import torchaudio.sox_effects as sox_effects
import subprocess
import tempfile
import numpy as np
import random
import os
import time
import math # 引入 math 用于 ceil
from multiprocessing import Pool, cpu_count # 导入多进程库
from julius import fft_conv1d, resample_frac
import julius
import typing as tp
import uuid # 用于生成唯一文件名
import torchaudio.transforms as T
import soundfile as sf

_offset_rng = random.Random()
_offset_rng.seed(os.urandom(16))
################################################################################
# Utilities
################################################################################

def attack_crop_like(
    signal_encoded: torch.Tensor, # 已经加了水印的信号 [B, T] 或 [B, 1, T]
    signal_i: torch.Tensor       # 对应的原始信号   [B, T] 或 [B, 1, T]
) -> torch.Tensor:
    """
    对已加水印的信号应用类似 'crop' 的攻击，模拟部分信号丢失或被原始信号替换。

    攻击类型 (随机选择):
        - Zeroing: 将信号的一部分置零。
        - Replacement: 将信号的一部分替换为对应的原始信号。
        - No attack: 信号保持不变。

    Args:
        signal_encoded (torch.Tensor): 已加水印的信号。
        signal_i (torch.Tensor): 对应的原始（未加水印）信号。

    Returns:
        torch.Tensor: 经过类 crop 攻击后的信号，形状与输入相同。
    """
    assert signal_encoded.shape == signal_i.shape, "输入信号和原始信号形状必须一致"
    assert signal_encoded.device == signal_i.device, "输入信号和原始信号必须在同一设备"
    # print("crop_like")
    # 确定概率 (可以调整)
    p_zeroing = 0.0
    p_replacement = 0.5 # 包含原 shuffle 和 crop 的概率
    # p_no_attack = 1.0 - p_zeroing - p_replacement = 0.4

    B, T = signal_encoded.shape[0], signal_encoded.shape[-1]
    device = signal_encoded.device

    # --- 适配输入形状，确保操作在 [B, T] 上进行 ---
    input_dim = signal_encoded.dim()
    if input_dim == 3: # [B, 1, T] -> [B, T]
        signal_encoded_proc = signal_encoded.squeeze(1)
        signal_i_proc = signal_i.squeeze(1)
    elif input_dim == 2: # [B, T]
        signal_encoded_proc = signal_encoded
        signal_i_proc = signal_i
    else:
        raise ValueError("输入信号维度应为 2 ([B, T]) 或 3 ([B, 1, T])")

    # --- 随机选择攻击类型 ---
    p = torch.rand(1, device=device).item()
    signal_attacked = signal_encoded_proc.clone() # 默认不攻击

    if p < p_zeroing + p_replacement: # 需要生成 mask
        mask = torch.ones_like(signal_encoded_proc) # 默认保留所有信号

        if p < p_zeroing: # 应用 Zeroing 攻击
            # --- 生成类似 Padding 的 mask ---
            start = int(torch.rand(1).item() * 0.22 * T)
            finish = int((0.77 + torch.rand(1).item() * 0.22) * T)
            mask[:, :start] = 0
            mask[:, finish:] = 0
            if torch.rand(1).item() > 0.5: # 一半概率反转 mask
                mask = 1 - mask
            # print("[attack_crop_like] Applying Zeroing attack.")
            signal_attacked = signal_encoded_proc * mask

        else: # p_zeroing <= p < p_zeroing + p_replacement，应用 Replacement 攻击
            # --- 生成类似 Crop/Shuffle 的 mask (窗口置零) ---
            mask_size_ratio = 0.5 # 最多替换/置零 50%
            mask_samples = round(T * mask_size_ratio)
            n_windows = torch.randint(1, 5 + 1, (1,), device=device).item()
            window_size = max(1, int(mask_samples / n_windows)) # 确保 window_size 至少为 1

            for _ in range(n_windows):
                if T > window_size:
                    mask_start = torch.randint(0, T - window_size, (1,), device=device).item()
                    mask[:, mask_start : mask_start + window_size] = 0
                else: # 如果窗口比信号还长，则全部置零
                    mask[:, :] = 0
                    break

            if torch.rand(1).item() > 0.5: # 一半概率反转 mask
                mask = 1 - mask

            # print("[attack_crop_like] Applying Replacement attack.")
            # 应用替换：保留 mask 为 1 的 watermarked 部分，替换 mask 为 0 的部分为原始信号
            signal_attacked = signal_encoded_proc * mask + signal_i_proc * (1 - mask)

    else: # p >= p_zeroing + p_replacement
        # print("[attack_crop_like] No crop-like attack applied.")
        # signal_attacked 保持为 signal_encoded_proc.clone()
        pass

    # --- 恢复原始维度 ---
    if input_dim == 3:
        signal_attacked = signal_attacked.unsqueeze(1)

    return signal_attacked

def to_equal_length_torch(original: torch.Tensor, watermarked: torch.Tensor):
    """
    若 original 和 watermarked 长度不同，则截断到最短长度。
    适用于 1D 或 2D( batch=1 ) 情况。
    (此函数在新的需求下可能较少使用，但保留以备不时之需)
    """
    if original.shape != watermarked.shape:
        min_len = min(original.shape[-1], watermarked.shape[-1])
        original = original[..., :min_len]
        watermarked = watermarked[..., :min_len]
    return original, watermarked


def align_with_offset_torch(
    signal_primary: torch.Tensor,
    signal_fill: torch.Tensor,
    target_len: int,
    max_offset_ratio: float = 0
) -> torch.Tensor:
    """
    【批处理对齐函数 - 双源带随机偏移】
    将 primary 信号批次中的每个信号调整到 target_len，使用对应的 fill 信号进行填充，并应用独立随机时间偏移。
    能够处理 signal_primary 和 signal_fill 长度不一致的情况。

    Args:
        signal_primary (torch.Tensor): 主要内容的信号批次 [B, T_primary]。
        signal_fill (torch.Tensor): 用于填充的信号批次 [B, T_fill]。
        target_len (int): 目标输出长度。
        max_offset_ratio (float, optional): 最大偏移比例 (相对于 target_len)。默认为 0.05 (5%)。

    Returns:
        torch.Tensor: 长度为 target_len 且带有随机偏移的组合信号批次 [B, target_len]。
    """
    if signal_primary.dim() != 2 or signal_fill.dim() != 2:
        raise ValueError("align_with_offset_torch 要求输入为 [B, T] 格式")
    if signal_primary.shape[0] != signal_fill.shape[0]:
         raise ValueError("Batch size mismatch between signal_primary and signal_fill")

    batch_size = signal_primary.shape[0]
    device = signal_primary.device
    dtype = signal_primary.dtype

    aligned_batch = torch.zeros((batch_size, target_len), dtype=dtype, device=device)

    max_offset = int(target_len * max_offset_ratio)
    if max_offset < 0: max_offset = 0

    for b in range(batch_size):
        primary = signal_primary[b] # [T_primary]
        fill = signal_fill[b]       # [T_fill]
        len_primary = primary.shape[-1]
        len_fill = fill.shape[-1]

        current_fill = fill
        current_len_fill = len_fill
        if current_len_fill == 0:
            current_fill = primary
            current_len_fill = len_primary
            if current_len_fill == 0:
                 if target_len > 0:
                     aligned_batch[b] = torch.zeros(target_len, dtype=dtype, device=device)
                     continue
                 else:
                     continue

        if max_offset == 0:
            random_offset = 0
        else:
            random_offset = _offset_rng.randint(-max_offset, max_offset)

        conceptual_start_index = (len_primary // 2) - (target_len // 2) + random_offset
        output_indices = torch.arange(target_len, device=device)
        conceptual_sample_indices = conceptual_start_index + output_indices

        current_aligned_signal = torch.zeros(target_len, dtype=dtype, device=device)

        primary_mask = (conceptual_sample_indices >= 0) & (conceptual_sample_indices < len_primary)
        fill_before_mask = conceptual_sample_indices < 0
        fill_after_mask = conceptual_sample_indices >= len_primary

        indices_from_primary = conceptual_sample_indices[primary_mask]
        if indices_from_primary.numel() > 0:
            # 确保 primary 索引不越界 (可能因为 primary 很短而发生)
            valid_primary_indices = indices_from_primary.long()
            valid_primary_indices = valid_primary_indices[(valid_primary_indices >= 0) & (valid_primary_indices < len_primary)]
            if valid_primary_indices.numel() > 0:
                 values_from_primary = primary[valid_primary_indices]
                 # 将值放回对应的 mask 位置
                 mask_subset = primary_mask.clone() # 复制掩码
                 # 将不在 valid_primary_indices 中的掩码位置设为 False
                 mask_subset[primary_mask] = torch.isin(indices_from_primary, valid_primary_indices)
                 current_aligned_signal[mask_subset] = values_from_primary


        indices_for_fill_before = conceptual_sample_indices[fill_before_mask]
        if indices_for_fill_before.numel() > 0 and current_len_fill > 0:
            indices_in_fill_before = indices_for_fill_before % current_len_fill
            values_from_fill_before = current_fill[indices_in_fill_before.long()]
            current_aligned_signal[fill_before_mask] = values_from_fill_before

        indices_for_fill_after = conceptual_sample_indices[fill_after_mask]
        if indices_for_fill_after.numel() > 0 and current_len_fill > 0:
            indices_in_fill_after = (indices_for_fill_after - len_primary) % current_len_fill
            values_from_fill_after = current_fill[indices_in_fill_after.long()]
            current_aligned_signal[fill_after_mask] = values_from_fill_after

        aligned_batch[b] = current_aligned_signal

    return aligned_batch

def generate_pink_noise(length: int) -> torch.Tensor:
    """Generate pink noise using Voss-McCartney algorithm with PyTorch."""
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
    # Normalize
    pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
    return pink_noise

################################################################################
# Attack Functions - Simplified Signature, No Internal Alignment/STE
################################################################################

# 1. Additive White Gaussian Noise (AWGN)
def attack_random_noise_torch(
        waveform: torch.Tensor,
        noise_std: float = 0.001,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Add Gaussian noise to the waveform."""
        noise = torch.randn_like(waveform) * noise_std
        noisy_waveform = waveform + noise
        del noise
        if mask is None:
            return noisy_waveform
        else:
            return noisy_waveform,mask


def pink_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.01,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Add pink background noise to the waveform."""
        noise = generate_pink_noise(waveform.shape[-1]) * noise_std
        noise = noise.to(waveform.device)
        # Assuming waveform is of shape (bsz, channels, length)
        noisy_waveform = waveform + noise.unsqueeze(0).to(waveform.device)
        del noise
        if mask is None:
            return noisy_waveform
        else:
            return noisy_waveform, mask
    
# 2. Sample Suppression
def attack_sample_suppression_torch(signal: torch.Tensor, p: float = 0.001) -> torch.Tensor:
    """
    [可微分] 按概率 p 将部分样本置零 (支持批处理 [B, T])
    Args:
        signal (torch.Tensor): 输入信号 [B, T]。
        p (float): 置零概率。
    Returns:
        torch.Tensor: 置零后的信号 [B, T] (长度不变)。
    """
    if signal.dim() != 2: raise ValueError("需要 [B, T] 输入")
    mask = torch.bernoulli(torch.full_like(signal, 1 - p))
    signal_suppressed = signal * mask
    return signal_suppressed


def attack_low_pass_filter_torch(
    signal_batch: torch.Tensor,
    sample_rate: int = 48000,
    cutoff: float = 6000.0,
    num_cascades: int = 5  # 大幅增加级联次数以获得更陡峭的截止, 例如 6-10 次
) -> torch.Tensor:
    """
    [纯Pytorch, 可微分] 通过多次级联二阶低通滤波器(biquad)实现非常陡峭的频率截止。
    总等效阶数约为 2 * num_cascades。
    例如, num_cascades=6 约等于12阶滤波器。
    num_cascades=10 约等于20阶滤波器。

    请注意：
    1. 陡峭度增加的同时，计算量也会增加。
    2. 实际的-3dB截止点会比单级滤波器在`cutoff_freq`处的截止点更低，
       或者说，在指定的`cutoff_freq`处的衰减会比单级滤波器大得多
       (衰减大约是 单级衰减dB值 * num_cascades)。
    3. 可能会引入更大的群延迟和相位失真。

    Args:
        signal_batch (torch.Tensor): 输入信号, 形状为 [B, T] (批次数, 时间点数)。
        sample_rate (int): 采样率 (Hz)。
        cutoff_freq (float): 每个二阶低通滤波器的目标截止频率 (Hz)。
        num_cascades (int): biquad滤波器的级联次数。次数越多，截止越陡峭。
    Returns:
        torch.Tensor: 低通滤波后的信号, 形状为 [B, T]。
    """
    if signal_batch.dim() != 2:
        raise ValueError("输入信号必须是二维张量 [B, T]")
    if signal_batch.numel() == 0:
        return signal_batch # 处理空张量

    # 确保在与输入数据相同的设备上操作
    current_signal = signal_batch.clone() # 操作克隆以避免修改原始张量
    cutoff = random.uniform(3000, 6000)
    for i in range(num_cascades):
        current_signal = AF.lowpass_biquad(
            current_signal,
            sample_rate=sample_rate,
            cutoff_freq=cutoff
        )
    
    return current_signal


def attack_spec_augment_torch(
    signal_batch: torch.Tensor,
    sample_rate: int = 48000,
    n_fft: int = 2048,
    hop_length: int = 512,
    freq_mask_param: int = 80,
    time_mask_param: int = 80,
    mask_gain: float = 1e-9  # 默认使用一个极小值而非0，以保证数值稳定性
) -> torch.Tensor:
    """
    [健壮版, 纯Pytorch, 可微分] 通过在频谱图上应用参数化滤波器来模拟SpecAugment攻击。
    此版本经过加固，以最大程度保证训练过程中的数值稳定性。

    1. 将时域信号转换为频域的复数频谱图 (STFT)。
    2. 将遮盖操作实现为对特定区域应用一个极低的、非零的增益（乘法）。
    3. 将修改后的幅度谱与原始相位信息结合。
    4. 使用可微分的逆短时傅里叶变换 (iSTFT) 将频谱图转换回时域信号。
    """
    # --- 0. 输入校验和准备 ---
    if signal_batch.dim() != 2:
        raise ValueError("输入信号必须是二维张量 [B, T]")
    if signal_batch.numel() == 0:
        return signal_batch

    device = signal_batch.device
    batch_size = signal_batch.shape[0]
    original_length = signal_batch.shape[1]
    
    # 【健壮性措施 1】确保增益不是真正的零，防止梯度完全消失
    # 即使输入 mask_gain=0.0，我们也用一个极小值代替，这对结果影响微乎其微，但对梯度计算更安全。
    stable_mask_gain = max(mask_gain, 1e-9)

    # --- 1. 定义可微分的 STFT 和 iSTFT ---
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=None,
        center=True,
        pad_mode="reflect"
    ).to(device)

    inverse_spectrogram_transform = T.InverseSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        center=True
    ).to(device)

    # --- 2. 波形 -> 复数频谱图 -> 分离幅度和相位 ---
    complex_spec = spectrogram_transform(signal_batch)
    
    # 【健壮性措施 2】在计算幅度和相位时增加一个极小量，避免输入为 0+0j 的罕见情况
    # 这可以提高 torch.angle 在零点附近的数值稳定性
    epsilon = 1e-9
    complex_spec_stable = complex_spec + epsilon * 1j
    
    magnitude_spec = torch.abs(complex_spec_stable)
    phase_spec = torch.angle(complex_spec_stable)

    num_freq_bins = magnitude_spec.shape[1]
    num_time_frames = magnitude_spec.shape[2]
    
    masked_magnitude_spec = magnitude_spec.clone()

    # --- 3. 应用参数化滤波器（等效于 SpecAugment）---
    # 对批处理中的每个样本独立应用随机遮盖
    for i in range(batch_size):
        # 频率遮盖
        # f_width = torch.randint(0, freq_mask_param + 1, (1,)).item()
        # if f_width > 0:
        #     f_start = torch.randint(0, num_freq_bins - f_width + 1, (1,)).item()
        #     masked_magnitude_spec[i, f_start:f_start+f_width, :] *= stable_mask_gain

        # # 时间遮盖
        # t_width = torch.randint(0, time_mask_param + 1, (1,)).item()
        # if t_width > 0:
        #     t_start = torch.randint(0, num_time_frames - t_width + 1, (1,)).item()
        #     masked_magnitude_spec[i, :, t_start:t_start+t_width] *= stable_mask_gain
        # 频率遮蔽（健壮性防御）
        max_f_width = min(freq_mask_param, num_freq_bins)
        if max_f_width > 0:
            f_width = torch.randint(0, max_f_width + 1, (1,)).item()
            if f_width > 0 and num_freq_bins - f_width > 0:
                f_start = torch.randint(0, num_freq_bins - f_width + 1, (1,)).item()
                masked_magnitude_spec[i, f_start:f_start+f_width, :] *= stable_mask_gain

        # 时间遮蔽（健壮性防御）
        max_t_width = min(time_mask_param, num_time_frames)
        if max_t_width > 0:
            t_width = torch.randint(0, max_t_width + 1, (1,)).item()
            if t_width > 0 and num_time_frames - t_width > 0:
                t_start = torch.randint(0, num_time_frames - t_width + 1, (1,)).item()
                masked_magnitude_spec[i, :, t_start:t_start+t_width] *= stable_mask_gain

    # --- 4. 频谱图 -> 波形 ---
    reconstructed_complex_spec = torch.polar(masked_magnitude_spec, phase_spec)
    
    reconstructed_signal = inverse_spectrogram_transform(
        reconstructed_complex_spec, length=original_length
    )
    
    return reconstructed_signal


def highpass_filter(
    waveform: torch.Tensor,
    cutoff_freq: float = 500,
    sample_rate: int = 16000,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Filter the highpass frequency from the waveform"""
    return julius.highpass_filter(waveform, cutoff=cutoff_freq / sample_rate)



# 4. Median Filter
def attack_median_filter_torch(signal: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    [不可微分] 1D 中值滤波 (通过循环支持批处理 [B, T])
    Args:
        signal (torch.Tensor): 输入信号 [B, T]。
        kernel_size (int): 滤波器核大小。
    Returns:
        torch.Tensor: 中值滤波后的信号 [B, T] (长度不变)。
    """
    if signal.dim() != 2: raise ValueError("需要 [B, T] 输入")
    batch_size = signal.shape[0]
    medianed_batch = torch.zeros_like(signal)
    pad = kernel_size // 2

    for b in range(batch_size):
        current_signal = signal[b:b+1] # [1, T]
        signal_padded = F.pad(current_signal, (pad, pad), mode='reflect')
        unfolded = signal_padded.unfold(dimension=-1, size=kernel_size, step=1)
        medianed_val = unfolded.median(dim=-1).values
        medianed_batch[b] = medianed_val.squeeze(0)

    return medianed_batch # 直接返回中值滤波结果，无 STE


# 5. Re-Sample
def attack_resample_torchaudio(signal: torch.Tensor, sample_rate: int = 48000) -> torch.Tensor:
    """
    [可微分] 随机下采样再上采样 (支持批处理 [B, T])
    Args:
        signal (torch.Tensor): 输入信号 [B, T]。
        sample_rate (int): 原始采样率。
    Returns:
        torch.Tensor: 重采样后的信号 [B, T_new] (长度可能改变)。
    """
    if signal.dim() != 2: raise ValueError("需要 [B, T] 输入")
    possible_rates = [44100, 24000, 22050, 16000]
    available_rates = [r for r in possible_rates if r != sample_rate]
    if not available_rates: available_rates = [sample_rate]
    new_sr = random.choice(available_rates)
    if new_sr < 1: new_sr = 1

    signal_unsqueezed = signal.unsqueeze(1) # [B, 1, T]
    downsampler = AT.Resample(orig_freq=sample_rate, new_freq=new_sr, dtype=signal.dtype).to(signal.device)
    down = downsampler(signal_unsqueezed)
    upsampler = AT.Resample(orig_freq=new_sr, new_freq=sample_rate, dtype=signal.dtype).to(signal.device)
    up_unsqueezed = upsampler(down)
    up = up_unsqueezed.squeeze(1) # [B, T_new]
    return up # 返回重采样结果，长度可能不同


# 6. Amplitude Scaling
def attack_amplitude_scaling_torch(signal: torch.Tensor, factor: float = 0.9) -> torch.Tensor:
    """
    [可微分] 幅度缩放 (支持批处理 [B, T])
    Args:
        signal (torch.Tensor): 输入信号 [B, T]。
        factor (float): 缩放因子。
    Returns:
        torch.Tensor: 幅度缩放后的信号 [B, T] (长度不变)。
    """
    if signal.dim() != 2: raise ValueError("需要 [B, T] 输入")
    scaled_signal = signal * factor
    clamped_signal = torch.clamp(scaled_signal, -1.0, 1.0)
    return clamped_signal


# --- Helper for FFmpeg based attacks (通过文件传递) ---
def _run_ffmpeg_via_files(input_wav_path, output_wav_path, sample_rate, target_suffix, encode_cmd_list_template, decode_cmd_list_template, ffmpeg_path):
    """
    内部工作函数，通过文件路径接收输入和指定输出，执行 FFmpeg 编解码。
    直接接受 7 个解包后的参数。
    只执行命令，不加载结果。返回 True 表示成功，False 表示失败。
    """
    # 参数现在是直接传入的，不再需要从元组解包

    temp_target_name = None
    try:
        temp_dir = os.path.dirname(output_wav_path)
        # 使用 mktemp 在指定目录安全获取临时文件名
        temp_target_name = tempfile.mktemp(suffix=target_suffix, dir=temp_dir)

        # 替换命令模板中的路径
        encode_cmd = [item.replace("INPUT_WAV", input_wav_path).replace("OUTPUT_TARGET", temp_target_name) for item in encode_cmd_list_template]
        decode_cmd = [item.replace("INPUT_TARGET", temp_target_name).replace("OUTPUT_WAV", output_wav_path) for item in decode_cmd_list_template]

        # --- 编码 ---
        # 使用 check=True，让它在失败时抛出异常
        result_encode = subprocess.run(encode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # --- 解码 ---
        result_decode = subprocess.run(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # 如果两步都成功，返回 True
        return True

    except subprocess.CalledProcessError as e:
        # 打印错误信息，帮助调试
        # print(f"FFmpeg command failed in worker (Input: {input_wav_path}, Suffix: {target_suffix}).")
        # print(f"Stderr: {e.stderr}")
        # 命令失败，返回 False
        return False
    except Exception as proc_err:
         print(f"Unexpected error in worker process for {target_suffix} (Input: {input_wav_path}): {proc_err}")
         return False # 其他错误也算失败
    finally:
        # 清理编码后的中间文件 (aac/mp3/opus)
        if temp_target_name and os.path.exists(temp_target_name):
            try:
                os.remove(temp_target_name)
            except OSError:
                pass
        # 注意：输入和输出 WAV 文件由主进程管理

# 7. Lossy Compression (MP3) - 使用多进程和文件传递
def attack_lossy_compression_torch(signal: torch.Tensor, sample_rate: int = 48000, bitrate: int = 64, ffmpeg_path: str = "ffmpeg", num_workers: int = 16) -> torch.Tensor:
    """
    [不可微分] 使用 FFmpeg 进行 MP3 压缩 (多进程并行，通过文件传递数据)
    Args:
        signal (torch.Tensor): 输入信号 [B, T]。
        sample_rate (int): 采样率。
        bitrate (int): MP3 码率 (kbps)。
        ffmpeg_path (str): ffmpeg 可执行文件路径。
        num_workers (int): 使用的进程数。0 表示使用所有可用核心。
    Returns:
        torch.Tensor: MP3 压缩后的信号 [B, T_padded]。
    """
    if signal.dim() != 2: raise ValueError("需要 [B, T] 输入")
    batch_size = signal.shape[0]
    device = signal.device
    orig_dtype = signal.dtype

    if not os.path.exists(ffmpeg_path):
        print(f"Warning: FFmpeg not found at {ffmpeg_path}. Skipping MP3 attack.")
        return signal.clone()

    max_cpus = cpu_count()
    if num_workers <= 0: workers = max(1, max_cpus + num_workers)
    else: workers = min(num_workers, max_cpus)
    # print(f"[MP3 Attack (File I/O)] Using {workers} worker processes.")

    # --- 准备：移动到 CPU，创建临时目录和文件路径 ---
    signal_cpu = signal.cpu()
    encode_cmd_template = [ffmpeg_path, "-y", "-i", "INPUT_WAV", "-c:a", "libmp3lame", "-b:a", f"{bitrate}k", "-ac", "1", "-ar", str(sample_rate), "OUTPUT_TARGET"]
    decode_cmd_template = [ffmpeg_path, "-y", "-i", "INPUT_TARGET", "-ar", str(sample_rate), "-ac", "1", "-f", "wav", "OUTPUT_WAV"]

    # 使用 with TemporaryDirectory 管理输入和输出临时目录
    with tempfile.TemporaryDirectory() as input_temp_dir, \
         tempfile.TemporaryDirectory() as output_temp_dir:

        tasks_args = []
        input_wav_paths = []
        output_wav_paths = []

        # --- 1. 主进程：保存输入文件，准备任务参数 ---
        # print(f"[MP3 Attack (File I/O)] Saving input WAVs to {input_temp_dir}...")
        save_start = time.time()
        for b in range(batch_size):
            # 生成唯一的文件名
            unique_id = uuid.uuid4()
            input_wav_path = os.path.join(input_temp_dir, f"input_{unique_id}.wav")
            output_wav_path = os.path.join(output_temp_dir, f"output_{unique_id}.wav")

            # 保存单个样本到输入文件
            # torchaudio.save 需要 [C, T] 或 [T]
            current_signal_to_save = signal_cpu[b] # [T]
            try:
                 torchaudio.save(input_wav_path, current_signal_to_save.unsqueeze(0), sample_rate) # 保存为 [1, T]
            except Exception as save_err:
                 print(f"Error saving input WAV for sample {b}: {save_err}. Skipping this sample.")
                 # 添加占位符或标记，以便后续处理
                 input_wav_paths.append(None)
                 output_wav_paths.append(None)
                 tasks_args.append(None) # 添加 None 标记任务失败
                 continue

            input_wav_paths.append(input_wav_path)
            output_wav_paths.append(output_wav_path)
            tasks_args.append(
                (input_wav_path, output_wav_path, sample_rate, ".mp3", encode_cmd_template, decode_cmd_template, ffmpeg_path)
            )
        # print(f"[MP3 Attack (File I/O)] Input WAVs saved in {time.time() - save_start:.4f} seconds.")

        # 过滤掉失败的任务
        valid_tasks_args = [args for args in tasks_args if args is not None]
        if not valid_tasks_args:
            print("[MP3 Attack (File I/O)] No valid tasks to process after saving inputs.")
            return signal.clone()

        # --- 2. 使用进程池执行 FFmpeg 命令 ---
        # print(f"[MP3 Attack (File I/O)] Starting FFmpeg processing for {len(valid_tasks_args)} tasks...")
        process_start = time.time()
        task_success_flags = [] # 存储每个任务的成功标志
        try:
            with Pool(processes=workers) as pool:
                # map 只接受一个参数的函数，所以需要一个包装器或使用 starmap
                # results 是一个布尔值列表 [True, False, True, ...]
                results = pool.starmap(_run_ffmpeg_via_files, valid_tasks_args)
                task_success_flags = results
        except Exception as pool_err:
             print(f"[MP3 Attack (File I/O)] Error during multiprocessing pool execution: {pool_err}")
             # 即使池出错，仍然尝试加载已成功处理的文件
             task_success_flags = [False] * len(valid_tasks_args) # 假设全部失败

        # print(f"[MP3 Attack (File I/O)] FFmpeg processing finished in {time.time() - process_start:.4f} seconds.")
        # print(f"[MP3 Attack (File I/O)] Success flags: {task_success_flags}")


        # --- 3. 主进程：加载输出文件 ---
        # print(f"[MP3 Attack (File I/O)] Loading output WAVs from {output_temp_dir}...")
        load_start = time.time()
        reconstructed_batch_list = []
        valid_task_index = 0
        for b in range(batch_size):
            # 检查原始任务是否有效且对应进程是否成功
            if tasks_args[b] is not None and valid_task_index < len(task_success_flags) and task_success_flags[valid_task_index]:
                output_wav_path = output_wav_paths[b]
                try:
                    # 尝试加载输出文件 (CPU Tensor)
                    reconstructed_cpu, sr_rec = torchaudio.load(output_wav_path)
                    if sr_rec != sample_rate: # 理论上解码命令指定了SR，但这层检查更安全
                        resampler = AT.Resample(orig_freq=sr_rec, new_freq=sample_rate, dtype=reconstructed_cpu.dtype).to('cpu')
                        reconstructed_cpu = resampler(reconstructed_cpu)
                    reconstructed_batch_list.append(reconstructed_cpu.squeeze(0)) # [T_new]
                except Exception as load_err:
                    # print(f"Error loading output WAV for sample {b}: {load_err}. Using original signal.")
                    reconstructed_batch_list.append(signal_cpu[b]) # 加载失败用原始信号
                valid_task_index += 1
            else:
                # 如果原始保存失败，或进程执行失败
                # print(f"Using original signal for sample {b} due to previous error or failed processing.")
                reconstructed_batch_list.append(signal_cpu[b]) # 使用原始信号

        # print(f"[MP3 Attack (File I/O)] Output WAVs loaded in {time.time() - load_start:.4f} seconds.")


    # --- 4. 后处理：Padding, Stacking, Device Transfer ---
    post_start = time.time()
    try:
        valid_results = [t for t in reconstructed_batch_list if isinstance(t, torch.Tensor) and t.numel() > 0]
        if not valid_results:
             print("[MP3 Attack (File I/O)] No valid signals loaded. Returning original.")
             return signal.clone()

        max_len = max(t.shape[0] for t in valid_results)
        padded_list = []
        for t in reconstructed_batch_list:
            if isinstance(t, torch.Tensor):
                 current_len = t.shape[0]
                 if current_len == 0: padded_list.append(torch.zeros(max_len, dtype=orig_dtype, device='cpu'))
                 elif current_len > max_len: padded_list.append(t[:max_len].to(orig_dtype))
                 else: padded_list.append(F.pad(t, (0, max_len - current_len)).to(orig_dtype))
            else: # 不应发生，因为失败情况已填入原始信号
                 padded_list.append(torch.zeros(max_len, dtype=orig_dtype, device='cpu'))

        reconstructed_batch_cpu = torch.stack(padded_list, dim=0)
        reconstructed_batch = reconstructed_batch_cpu.to(device)
        # print(f"[MP3 Attack (File I/O)] Post-processing took {time.time() - post_start:.4f} seconds.")
        return reconstructed_batch

    except Exception as post_err:
        print(f"[MP3 Attack (File I/O)] Error during post-processing: {post_err}")
        return signal.clone()


# --- attack_aac_compression_torch 和 attack_opus_compression_torch ---
# 可以用几乎完全相同的方式修改，只需改变：
# - 打印的攻击名称 ("[AAC Attack (File I/O)]", "[Opus Attack (File I/O)]")
# - target_suffix (".aac", ".opus")
# - encode_cmd_template 中的编码器和参数 ("-c:a aac -b:a bitrate", "-c:a libopus -b:a bitrate -ar 48000")
# - 随机比特率列表 (bitrates)

# 这里提供 AAC 的修改版本作为示例：

def attack_aac_compression_torch(signal: torch.Tensor, sample_rate: int = 48000, ffmpeg_path: str = "ffmpeg", num_workers: int = 16) -> torch.Tensor:
    """
    [不可微分] 使用 FFmpeg 进行 AAC 压缩 (随机比特率, 使用多进程和文件传递)
    Args:
        signal (torch.Tensor): 输入信号 [B, T]。
        sample_rate (int): 采样率。
        ffmpeg_path (str): ffmpeg 可执行文件路径。
        num_workers (int): 使用的进程数。0 表示使用所有可用核心。
    Returns:
        torch.Tensor: AAC 压缩后的信号 [B, T_padded]。
    """
    if signal.dim() != 2: raise ValueError("需要 [B, T] 输入")
    batch_size = signal.shape[0]
    device = signal.device
    orig_dtype = signal.dtype

    if not os.path.exists(ffmpeg_path):
        print(f"Warning: FFmpeg not found at {ffmpeg_path}. Skipping AAC attack.")
        return signal.clone()

    max_cpus = cpu_count()
    if num_workers <= 0: workers = max(1, max_cpus + num_workers)
    else: workers = min(num_workers, max_cpus)
    # print(f"[AAC Attack (File I/O)] Using {workers} worker processes.")

    signal_cpu = signal.cpu()
    bitrates = ["160k", "128k", "96k", "64k"]
    decode_cmd_template = [ffmpeg_path, "-y", "-i", "INPUT_TARGET", "-ar", str(sample_rate), "-ac", "1", "-f", "wav", "OUTPUT_WAV"]

    with tempfile.TemporaryDirectory() as input_temp_dir, \
         tempfile.TemporaryDirectory() as output_temp_dir:

        tasks_args = []
        input_wav_paths = []
        output_wav_paths = []

        save_start = time.time()
        for b in range(batch_size):
            unique_id = uuid.uuid4()
            input_wav_path = os.path.join(input_temp_dir, f"input_{unique_id}.wav")
            output_wav_path = os.path.join(output_temp_dir, f"output_{unique_id}.wav")
            current_signal_to_save = signal_cpu[b]
            try:
                 torchaudio.save(input_wav_path, current_signal_to_save.unsqueeze(0), sample_rate)
            except Exception as save_err:
                 print(f"Error saving input WAV for sample {b}: {save_err}.")
                 input_wav_paths.append(None)
                 output_wav_paths.append(None)
                 tasks_args.append(None)
                 continue

            input_wav_paths.append(input_wav_path)
            output_wav_paths.append(output_wav_path)
            # 为每个任务随机选择比特率
            bitrate = random.choice(bitrates)
            encode_cmd_template = [ffmpeg_path, "-y", "-i", "INPUT_WAV", "-c:a", "aac", "-b:a", bitrate, "-ac", "1", "-ar", str(sample_rate), "OUTPUT_TARGET"]
            tasks_args.append(
                (input_wav_path, output_wav_path, sample_rate, ".aac", encode_cmd_template, decode_cmd_template, ffmpeg_path)
            )
        # print(f"[AAC Attack (File I/O)] Input WAVs saved in {time.time() - save_start:.4f} seconds.")

        valid_tasks_args = [args for args in tasks_args if args is not None]
        if not valid_tasks_args:
            print("[AAC Attack (File I/O)] No valid tasks to process.")
            return signal.clone()

        process_start = time.time()
        task_success_flags = []
        try:
            with Pool(processes=workers) as pool:
                results = pool.starmap(_run_ffmpeg_via_files, valid_tasks_args)
                task_success_flags = results
        except Exception as pool_err:
             print(f"[AAC Attack (File I/O)] Error during multiprocessing pool execution: {pool_err}")
             task_success_flags = [False] * len(valid_tasks_args)
        # print(f"[AAC Attack (File I/O)] FFmpeg processing finished in {time.time() - process_start:.4f} seconds.")

        load_start = time.time()
        reconstructed_batch_list = []
        valid_task_index = 0
        for b in range(batch_size):
            if tasks_args[b] is not None and valid_task_index < len(task_success_flags) and task_success_flags[valid_task_index]:
                output_wav_path = output_wav_paths[b]
                try:
                    reconstructed_cpu, sr_rec = torchaudio.load(output_wav_path)
                    if sr_rec != sample_rate:
                        resampler = AT.Resample(orig_freq=sr_rec, new_freq=sample_rate, dtype=reconstructed_cpu.dtype).to('cpu')
                        reconstructed_cpu = resampler(reconstructed_cpu)
                    reconstructed_batch_list.append(reconstructed_cpu.squeeze(0))
                except Exception as load_err:
                    reconstructed_batch_list.append(signal_cpu[b])
                valid_task_index += 1
            else:
                reconstructed_batch_list.append(signal_cpu[b])
        # print(f"[AAC Attack (File I/O)] Output WAVs loaded in {time.time() - load_start:.4f} seconds.")

    post_start = time.time()
    try:
        valid_results = [t for t in reconstructed_batch_list if isinstance(t, torch.Tensor) and t.numel() > 0]
        if not valid_results: return signal.clone()
        max_len = max(t.shape[0] for t in valid_results)
        padded_list = []
        for t in reconstructed_batch_list:
            if isinstance(t, torch.Tensor):
                 current_len = t.shape[0]
                 if current_len == 0: padded_list.append(torch.zeros(max_len, dtype=orig_dtype, device='cpu'))
                 elif current_len > max_len: padded_list.append(t[:max_len].to(orig_dtype))
                 else: padded_list.append(F.pad(t, (0, max_len - current_len)).to(orig_dtype))
            else: padded_list.append(torch.zeros(max_len, dtype=orig_dtype, device='cpu'))
        reconstructed_batch_cpu = torch.stack(padded_list, dim=0)
        reconstructed_batch = reconstructed_batch_cpu.to(device)
        # print(f"[AAC Attack (File I/O)] Post-processing took {time.time() - post_start:.4f} seconds.")
        return reconstructed_batch
    except Exception as post_err:
        print(f"[AAC Attack (File I/O)] Error during post-processing: {post_err}")
        return signal.clone()
    

# 13. Opus Compression - 使用多进程和文件传递
def attack_opus_compression_torch(signal: torch.Tensor, sample_rate: int = 48000, ffmpeg_path: str = "ffmpeg", num_workers: int = 16) -> torch.Tensor:
    """
    [不可微分] 使用 FFmpeg 进行 Opus 压缩 (随机比特率, 使用多进程和文件传递)
    Args:
        signal (torch.Tensor): 输入信号 [B, T]。
        sample_rate (int): 采样率。
        ffmpeg_path (str): ffmpeg 可执行文件路径。
        num_workers (int): 使用的进程数。0 表示使用所有可用核心。
    Returns:
        torch.Tensor: Opus 压缩后的信号 [B, T_padded]。
    """
    if signal.dim() != 2: raise ValueError("需要 [B, T] 输入")
    batch_size = signal.shape[0]
    device = signal.device
    orig_dtype = signal.dtype

    if not os.path.exists(ffmpeg_path):
        print(f"Warning: FFmpeg not found at {ffmpeg_path}. Skipping Opus attack.")
        return signal.clone()

    max_cpus = cpu_count()
    if num_workers <= 0: workers = max(1, max_cpus + num_workers)
    else: workers = min(num_workers, max_cpus)
    # print(f"[Opus Attack (File I/O)] Using {workers} worker processes.")

    # --- 准备：移动到 CPU，定义命令模板，创建临时目录 ---
    signal_cpu = signal.cpu()
    # Opus 编码推荐使用 48kHz 内部处理，但解码可以指定回原始采样率
    decode_cmd_template = [ffmpeg_path, "-y", "-i", "INPUT_TARGET", "-ar", str(sample_rate), "-ac", "1", "-f", "wav", "OUTPUT_WAV"]
    bitrates = ["96k", "64k", "48k", "32k"] # Opus 常用比特率

    with tempfile.TemporaryDirectory() as input_temp_dir, \
         tempfile.TemporaryDirectory() as output_temp_dir:

        tasks_args = []
        input_wav_paths = []
        output_wav_paths = []

        # --- 1. 主进程：保存输入文件，准备任务参数 ---
        save_start = time.time()
        for b in range(batch_size):
            unique_id = uuid.uuid4()
            input_wav_path = os.path.join(input_temp_dir, f"input_{unique_id}.wav")
            output_wav_path = os.path.join(output_temp_dir, f"output_{unique_id}.wav")
            current_signal_to_save = signal_cpu[b]
            try:
                 torchaudio.save(input_wav_path, current_signal_to_save.unsqueeze(0), sample_rate)
            except Exception as save_err:
                 print(f"Error saving input WAV for sample {b}: {save_err}.")
                 input_wav_paths.append(None)
                 output_wav_paths.append(None)
                 tasks_args.append(None)
                 continue

            input_wav_paths.append(input_wav_path)
            output_wav_paths.append(output_wav_path)
            # 为每个任务随机选择比特率
            bitrate = random.choice(bitrates)
            # 定义 Opus 编码命令模板 (注意 -ar 48000)
            encode_cmd_template = [ffmpeg_path, "-y", "-i", "INPUT_WAV", "-c:a", "libopus", "-b:a", bitrate, "-ac", "1", "-ar", "48000", "OUTPUT_TARGET"]
            tasks_args.append(
                (input_wav_path, output_wav_path, sample_rate, ".opus", encode_cmd_template, decode_cmd_template, ffmpeg_path)
            )
        # print(f"[Opus Attack (File I/O)] Input WAVs saved in {time.time() - save_start:.4f} seconds.")

        valid_tasks_args = [args for args in tasks_args if args is not None]
        if not valid_tasks_args:
            print("[Opus Attack (File I/O)] No valid tasks to process.")
            return signal.clone()

        # --- 2. 使用进程池执行 FFmpeg 命令 ---
        process_start = time.time()
        task_success_flags = []
        try:
            with Pool(processes=workers) as pool:
                results = pool.starmap(_run_ffmpeg_via_files, valid_tasks_args)
                task_success_flags = results
        except Exception as pool_err:
             print(f"[Opus Attack (File I/O)] Error during multiprocessing pool execution: {pool_err}")
             task_success_flags = [False] * len(valid_tasks_args)
        # print(f"[Opus Attack (File I/O)] FFmpeg processing finished in {time.time() - process_start:.4f} seconds.")


        # --- 3. 主进程：加载输出文件 ---
        load_start = time.time()
        reconstructed_batch_list = []
        valid_task_index = 0
        for b in range(batch_size):
            if tasks_args[b] is not None and valid_task_index < len(task_success_flags) and task_success_flags[valid_task_index]:
                output_wav_path = output_wav_paths[b]
                try:
                    reconstructed_cpu, sr_rec = torchaudio.load(output_wav_path)
                    if sr_rec != sample_rate:
                        resampler = AT.Resample(orig_freq=sr_rec, new_freq=sample_rate, dtype=reconstructed_cpu.dtype).to('cpu')
                        reconstructed_cpu = resampler(reconstructed_cpu)
                    reconstructed_batch_list.append(reconstructed_cpu.squeeze(0))
                except Exception as load_err:
                    reconstructed_batch_list.append(signal_cpu[b])
                valid_task_index += 1
            else:
                reconstructed_batch_list.append(signal_cpu[b])
        # print(f"[Opus Attack (File I/O)] Output WAVs loaded in {time.time() - load_start:.4f} seconds.")

    # --- 4. 后处理：Padding, Stacking, Device Transfer ---
    post_start = time.time()
    try:
        valid_results = [t for t in reconstructed_batch_list if isinstance(t, torch.Tensor) and t.numel() > 0]
        if not valid_results:
             print("[Opus Attack (File I/O)] No valid signals loaded. Returning original.")
             return signal.clone()
        max_len = max(t.shape[0] for t in valid_results)
        padded_list = []
        for t in reconstructed_batch_list:
            if isinstance(t, torch.Tensor):
                 current_len = t.shape[0]
                 if current_len == 0: padded_list.append(torch.zeros(max_len, dtype=orig_dtype, device='cpu'))
                 elif current_len > max_len: padded_list.append(t[:max_len].to(orig_dtype))
                 else: padded_list.append(F.pad(t, (0, max_len - current_len)).to(orig_dtype))
            else: padded_list.append(torch.zeros(max_len, dtype=orig_dtype, device='cpu'))

        reconstructed_batch_cpu = torch.stack(padded_list, dim=0)
        reconstructed_batch = reconstructed_batch_cpu.to(device)
        # print(f"[Opus Attack (File I/O)] Post-processing took {time.time() - post_start:.4f} seconds.")
        return reconstructed_batch
    except Exception as post_err:
        print(f"[Opus Attack (File I/O)] Error during post-processing: {post_err}")
        return signal.clone()
    
# 8. Quantization
def attack_quantization_torch(signal: torch.Tensor, levels=512) -> torch.Tensor:
    """
    [不可微分] 模拟均匀量化 (支持批处理 [B, T])
    Args:
        signal (torch.Tensor): 输入信号 [B, T]。
        levels (int): 量化级别数。
    Returns:
        torch.Tensor: 量化后的信号 [B, T] (长度不变)。
    """
    if signal.dim() != 2: raise ValueError("需要 [B, T] 输入")
    levels = max(1, int(levels))
    scale = (levels - 1) / 2.0
    quantized = torch.round(signal * scale) / scale
    quantized = torch.clamp(quantized, -1.0, 1.0)
    return quantized #直接返回量化结果，无 STE


# 9. Echo
def attack_echo_torch(
        tensor: torch.Tensor,
        volume_range: tuple = (0.1, 0.5),
        duration_range: tuple = (0.1, 0.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Attenuating the audio volume by a factor of 0.4, delaying it by 100ms,
    and then overlaying it with the original.

    Args:
        tensor: 3D Tensor representing the audio signal [bsz, channels, frames]
        volumne range: volume range of the echo signal
        duration range: duration range of the echo signal
        sample_rate: Sample rate of the audio signal.
    Returns:
        Audio signal with reverb.
    """
    added_channel = False
    if tensor.dim() == 2:  # [B, T] → [B, 1, T]
        tensor = tensor.unsqueeze(1)
        added_channel = True
        
    # Create a simple impulse response
    # Duration of the impulse response in seconds
    duration = torch.FloatTensor(1).uniform_(*duration_range)
    volume = torch.FloatTensor(1).uniform_(*volume_range)

    n_samples = int(sample_rate * duration)
    impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

    # Define a few reflections with decreasing amplitude
    impulse_response[0] = 1.0  # Direct sound

    impulse_response[
        int(sample_rate * duration) - 1
    ] = volume  # First reflection after 100ms

    # Add batch and channel dimensions to the impulse response
    impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

    # Convolve the audio signal with the impulse response
    reverbed_signal = fft_conv1d(tensor, impulse_response)
    del impulse_response
    # Normalize to the original amplitude range for stability
    reverbed_signal = (
        reverbed_signal
        / torch.max(torch.abs(reverbed_signal))
        * torch.max(torch.abs(tensor))
    )

    # Ensure tensor size is not changed
    tmp = torch.zeros_like(tensor)
    tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
    reverbed_signal = tmp

    # Remove channel dim if originally not present
    if added_channel:
        reverbed_signal = reverbed_signal.squeeze(1)
    if mask is not None:
        return reverbed_signal, mask
    else:
        return reverbed_signal


# 10. Time Stretch
def attack_time_stretch_torch(signal: torch.Tensor, rate=1.1, sample_rate=48000) -> torch.Tensor:
    """
    [可微分] 使用 F.interpolate 模拟时间拉伸 (支持批处理 [B, T])
    Args:
        signal (torch.Tensor): 输入信号 [B, T]。
        rate (float): 拉伸速率 (>1 加快, <1 减慢)。
        sample_rate (int): 采样率 (未使用)。
    Returns:
        torch.Tensor: 时间拉伸后的信号 [B, T] (长度不变)。
    """
    if signal.dim() != 2: raise ValueError("需要 [B, T] 输入")
    target_len = signal.shape[-1]

    if rate <= 0:
        print(f"Warning: Time stretch rate ({rate}) must be positive. Returning original.")
        stretched = signal.clone()
    elif rate == 1.0:
        stretched = signal.clone()
    else:
        new_len = int(target_len / rate)
        if new_len <= 0: new_len = 1
        tmp = signal.unsqueeze(1) # [B, 1, T]
        tmp_scaled = F.interpolate(tmp, size=new_len, mode='linear', align_corners=False)
        tmp_stretched_back = F.interpolate(tmp_scaled, size=target_len, mode='linear', align_corners=False)
        stretched = tmp_stretched_back.squeeze(1)  # [B, T]
    return stretched


# 11. Reverberation
def attack_reverberation_torch(signal: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """
    [可微分] 使用 RIR 进行卷积混响 (支持批处理 [B, T])
    Args:
        signal (torch.Tensor): 输入信号 [B, T]。
        rir (torch.Tensor): 房间脉冲响应 [R]。
    Returns:
        torch.Tensor: 添加混响后的信号 [B, T_new] (长度改变 T + R - 1)。
    """
    if signal.dim() != 2: raise ValueError("需要 [B, T] 输入")
    if rir.dim() != 1: raise ValueError("RIR 需要是 1D 张量 [R]")
    device = signal.device
    rir = rir.to(device)
    rir_len = rir.shape[-1]
    s = signal.unsqueeze(1) # [B, 1, T]
    r = rir.unsqueeze(0).unsqueeze(0) # [1, 1, R]
    padding = rir_len - 1
    out = F.conv1d(s, r, padding=padding) # 输出 shape [B, 1, T + R - 1]
    reverbed_full = out.squeeze(1)  # [B, T + R - 1]
    return reverbed_full # 返回原始卷积结果，长度增加


# 14. Speed (Tempo) Change
def attack_speed_torch(
        wav: torch.Tensor,
        speed_range=(0.5, 1.5),
        n_fft: int = 1024,
        hop_length: int = 256,
) -> torch.Tensor:
    """
    可微分的时间伸缩（保持音高）——Phase-Vocoder 实现
    wav        : [B,T] or [B,1,T]  (float-32, -1~1)
    speed_range: 随机伸缩比例区间  (0.8~1.2 代表慢/快)
    return     : 时长≈ T/ratio     (同 dtype / device)
    """
    if wav.dim() == 3:                   # [B,1,T] → [B,T]
        wav = wav.squeeze(1)
    B, T = wav.shape
    device, dtype = wav.device, wav.dtype

    speed = random.uniform(*speed_range)
    phase_advance = torch.linspace(
        0, math.pi * hop_length, n_fft // 2 + 1,
        device=device, dtype=dtype
    )[None, :, None]                    # [1,F,1]

    # STFT  ->  [B,F,τ]  (complex)
    spec = torch.stft(
        wav, n_fft=n_fft, hop_length=hop_length,
        window=torch.hann_window(n_fft, device=device, dtype=dtype),
        return_complex=True
    )

    # Phase-Vocoder (torchaudio 自带，支持 Autograd)
    spec_stretch = AF.phase_vocoder(spec, speed, phase_advance)  # :contentReference[oaicite:1]{index=1}

    # 目标长度  ≈ T / speed  (四舍五入)
    T_target = int(T / speed + 0.5)

    # iSTFT  ->  [B,T_target]
    wav_out = torch.istft(
        spec_stretch, n_fft=n_fft, hop_length=hop_length,
        window=torch.hann_window(n_fft, device=device, dtype=dtype),
        length=T_target
    )
    # print(f"wav_out is : {wav_out.shape}")
    return wav_out


def attack_speed_torch_pitch(
        wav: torch.Tensor,
        speed_range=(0.8, 1.2),            # 与前一个函数保持一致
        target_len: int = 96_000
) -> torch.Tensor:
    """
    线性插值式时间伸缩（保持音高）——批量版  
    wav        : [B,T]  或 [B,1,T]  或 [T]  
    speed_range: (min,max)；从区间随机采样一个 speed，整批共享  
    return     : [B,T_pad] (长度统一为 target_len，若不足补 0，超出则裁剪)
    """
    # -------- 统一形状到 [B,1,T] --------
    if wav.dim() == 1:          # [T] → [1,1,T]
        wav = wav.unsqueeze(0).unsqueeze(0)
    elif wav.dim() == 2:        # [B,T] → [B,1,T]
        wav = wav.unsqueeze(1)
    elif wav.dim() == 3 and wav.size(1) == 1:
        pass                    # 已是 [B,1,T]
    else:
        raise ValueError("支持 [T], [B,T] 或 [B,1,T] 三种输入")

    B, _, T = wav.shape
    device, dtype = wav.device, wav.dtype

    # -------- 采样 / 计算新长度 --------
    speed = random.uniform(*speed_range)
    
    new_T = max(2, int(T / speed+0.5))

    # -------- 线性插值拉伸 / 压缩 --------
    wav_stretch = F.interpolate(
        wav, size=new_T, mode="linear", align_corners=False
    )                            # [B,1,new_T]

    #-------- 对齐到 target_len --------
    # if new_T < target_len:                          # 不足则后补 0
    #     pad_sz = target_len - new_T
    #     wav_stretch = F.pad(wav_stretch, (0, pad_sz))
    # else:                                           # 过长直接裁剪
    #     wav_stretch = wav_stretch[..., :target_len]

    return wav_stretch.squeeze(1)               


def apply_random_phase_shift_torch(
        wav: torch.Tensor
) -> torch.Tensor:
    """
    对原始语音进行随机相位偏移。

    Args:
        wav (torch.Tensor): 输入音频张量，支持 [T], [B,T] 或 [B,1,T] 形状。
                            其中 T 是时间维度（采样点数），B 是批次大小。

    Returns:
        torch.Tensor: 施加随机相位偏移后的音频张量。
                      如果输入是 [T] 或 [B,T]，则输出为 [T] 或 [B,T]。
                      如果输入是 [B,1,T]，则输出为 [B,T]。
                      输出长度与输入长度一致。
    """
    original_dim = wav.dim()

    # -------- 统一形状到 [B,1,T] --------
    if original_dim == 1:  # [T] → [1,1,T]
        wav = wav.unsqueeze(0).unsqueeze(0)
    elif original_dim == 2:  # [B,T] → [B,1,T]
        wav = wav.unsqueeze(1)
    elif original_dim == 3 and wav.size(1) == 1:
        pass  # 已是 [B,1,T]
    else:
        raise ValueError("支持 [T], [B,T] 或 [B,1,T] 三种输入张量形状。")

    B, _, T = wav.shape
    device, dtype = wav.device, wav.dtype

    # -------- 提取批次维度并进行FFT --------
    # 将 [B, 1, T] 压缩为 [B, T]，因为 torch.fft.rfft 通常期望批次维度在最前面
    wav_batch = wav.squeeze(1) # Shape: [B, T]

    # 对实数信号执行快速傅里叶变换 (Real FFT)
    # n=T 确保 FFT 的长度与原始信号长度一致
    Y_freq = torch.fft.rfft(wav_batch, n=T) # Shape: [B, T//2 + 1] (复数)

    # 2. 分离幅度和相位
    magnitudes = torch.abs(Y_freq)  # 幅度谱
    phases = torch.angle(Y_freq)    # 相位谱 (每个频率分量的相位)

    # -------- 采样一个随机相位偏移量 --------
    # 从 [-π, π] 之间随机采样一个浮点数，作为全局相位偏移量
    # 使用 random.uniform 而非 torch.rand().item() 是为了保持与原始代码对 speed 的处理风格一致
    random_shift = random.uniform(-np.pi, np.pi)

    # -------- 应用相位偏移 --------
    # 将随机偏移量加到所有频率分量的相位上
    shifted_phases = phases + random_shift

    # -------- 重构复数频谱 --------
    # Y_shifted = magnitude * e^(j * shifted_phase)
    Y_shifted = magnitudes * torch.exp(1j * shifted_phases)

    # -------- 执行逆傅里叶变换 (Inverse Real FFT) --------
    # n=T 确保逆变换后的信号长度与原始信号长度一致
    wav_shifted = torch.fft.irfft(Y_shifted, n=T) # Shape: [B, T] (实数)

    # -------- 确保音频在 [-1, 1] 范围内，防止溢出或削波失真 --------
    wav_shifted = torch.clamp(wav_shifted, -1.0, 1.0)

    # -------- 根据原始输入维度调整输出形状 --------
    if original_dim == 1:
        return wav_shifted.squeeze(0) # 如果输入是 [T]，则输出为 [T]
    else: # original_dim == 2 ([B,T]) or 3 ([B,1,T])
        return wav_shifted # 输出为 [B,T]
    
  


def attack_pitch_shift_ffmpeg_asetrate_atempo(
    wav: torch.Tensor,
    sample_rate: int = 48000,
    n_steps: float = 0.0, # n_steps is not directly used for factor here, factor is random
    # n_fft and hop_length are not directly used by ffmpeg in this way,
    # but kept for signature consistency if needed for other logic.
    n_fft: int = 1024,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    使用 FFmpeg 的 asetrate 和 atempo 滤波器实现变调（不变速），支持批量处理。
    n_steps: 正值表示升调，负值表示降调 (单位：半音).
             在这个版本中，我们直接使用一个随机的 pitch_factor。
             如果想用 n_steps, 你需要计算 pitch_factor = 2**(n_steps / 12.0)
    """
    B, T = wav.shape
    device = wav.device
    dtype = wav.dtype # Remember original dtype

    # Pitch factor:
    # pitch_factor > 1.0 means higher pitch
    # pitch_factor < 1.0 means lower pitch
    # 如果你想基于 n_steps 来决定 pitch_factor:
    if n_steps != 0.0:
        pitch_factor = 2**(n_steps / 12.0)
    else:
        # 保持原来的随机因子作为示例
        pitch_factor = random.uniform(0.95, 1.05) # 例如 0.8 ~ 1.2

    print(f'Target pitch factor: {pitch_factor}')

    # atempo filter has a range of [0.5, 100.0].
    # If pitch_factor makes 1/pitch_factor outside this range, it might cause issues.
    # For typical pitch shifts (e.g., +/- 1 octave, factor 0.5 to 2.0), this is fine.
    tempo_factor = 1.0 / pitch_factor
    if not (0.5 <= tempo_factor <= 100.0):
        print(f"Warning: Calculated tempo_factor {tempo_factor} is outside FFmpeg's atempo typical range [0.5, 100.0]. Clamping or skipping.")
        # 你可以选择如何处理，例如：
        # 1. 跳过这个音高变换
        # 2. 将 tempo_factor 限制在范围内（但这会改变最终音高效果）
        # 3. 使用 rubberband 作为备选方案
        # 这里我们简单地返回原始音频如果因子超出预期范围
        if tempo_factor < 0.5: pitch_factor = 1/0.5 # Clamp
        elif tempo_factor > 100.0: pitch_factor = 1/100.0 # Clamp
        tempo_factor = 1.0 / pitch_factor
        print(f"Adjusted pitch_factor to {pitch_factor} and tempo_factor to {tempo_factor}")


    shifted_wavs = []

    for i in range(B):
        waveform_single = wav[i].detach().cpu().numpy() # Move to CPU and convert to NumPy

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as infile, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as outfile:
            infilepath = infile.name
            outfilepath = outfile.name

        try:
            # Save input tensor to a temporary WAV file
            sf.write(infilepath, waveform_single, sample_rate)

            # Construct FFmpeg command
            # -y: overwrite output files without asking
            # -i: input file
            # -af: audio filter
            # asetrate changes pitch and speed.
            # atempo changes speed back to original, keeping the new pitch.
            # The order is important.
            cmd = [
                "ffmpeg",
                "-y",
                "-i", infilepath,
                "-af", f"asetrate={sample_rate*pitch_factor},atempo={tempo_factor}",
                # 你也可以指定输出采样率，尽管asetrate的输出应该是原始采样率
                # 如果不指定，FFmpeg通常会保持与asetrate阶段后的采样率一致，
                # 但由于atempo不改变采样率，最终输出采样率应与输入一致。
                # 为了保险起见，可以明确指定：
                "-ar", str(sample_rate),
                outfilepath
            ]

            # Execute FFmpeg command
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"FFmpeg Error for item {i}:")
                print("STDOUT:", stdout.decode())
                print("STDERR:", stderr.decode())
                shifted_wavs.append(wav[i].unsqueeze(0)) # Keep on original device
                continue

            # Load the processed audio file
            shifted_waveform_single, sr_out = sf.read(outfilepath, dtype='float32') # Read as float32

            if sr_out != sample_rate:
                print(f"Warning: Output sample rate {sr_out} doesn't match target {sample_rate}. This might indicate an issue.")
                # 你可能需要在这里添加重采样逻辑，如果FFmpeg没有按预期工作
                # shifted_waveform_single = torchaudio.functional.resample(
                # torch.from_numpy(shifted_waveform_single), sr_out, sample_rate
                # ).numpy()

            shifted_waveform_tensor = torch.from_numpy(shifted_waveform_single).to(device)

            current_len = shifted_waveform_tensor.shape[0]
            if current_len < T:
                padding = T - current_len
                shifted_waveform_tensor = F.pad(shifted_waveform_tensor, (0, padding))
            elif current_len > T:
                shifted_waveform_tensor = shifted_waveform_tensor[:T]

            shifted_wavs.append(shifted_waveform_tensor.unsqueeze(0))
        finally:
            if os.path.exists(infilepath):
                os.remove(infilepath)
            if os.path.exists(outfilepath):
                os.remove(outfilepath)

    if not shifted_wavs:
        return torch.empty_like(wav)
    output_tensor = torch.cat(shifted_wavs, dim=0)
    return output_tensor.to(dtype)


# 16. Band Filter (Bandpass / Bandreject)
def attack_band_filter_torch(
        waveform: torch.Tensor,
        cutoff_freq_low: float = 300,
        cutoff_freq_high: float = 7900,  # 👈 修改为 < sample_rate / 2
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    # Ensure high cutoff < 0.5
    if cutoff_freq_high >= sample_rate / 2:
        cutoff_freq_high = sample_rate / 2 - 1  # 小于 Nyquist 限
    """Apply a bandpass filter to the waveform by cascading
    a high-pass filter followed by a low-pass filter.

    Args:
        waveform (torch.Tensor): Input audio waveform.
        low_cutoff (float): Lower cutoff frequency.
        high_cutoff (float): Higher cutoff frequency.
        sample_rate (int): The sample rate of the waveform.

    Returns:
        torch.Tensor: Filtered audio waveform.
    """
    return julius.bandpass_filter(
            waveform,
            cutoff_low=cutoff_freq_low / sample_rate,
            cutoff_high=cutoff_freq_high / sample_rate,
        )

        

def smooth(
        tensor: torch.Tensor,
        window_size_range: tuple = (2, 10),
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Smooths the input tensor (audio signal) using a moving average filter with the
    given window size.

    Args:
        tensor (torch.Tensor): Input audio tensor. Assumes tensor shape is (batch_size,
        channels, time).
        window_size (int): Size of the moving average window.
        mask: Masks for the input wave

    Returns:
        torch.Tensor: Smoothed audio tensor.
    """

    # 👉 自动处理 [B, T] → [B, 1, T]
    added_channel = False
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)  # Add channel dim
        added_channel = True

    window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))
    kernel = torch.ones(1, 1, window_size, dtype=tensor.dtype, device=tensor.device) / window_size

    # 👇 使用 padding 保持长度
    padding = window_size // 2
    smoothed = F.conv1d(tensor, kernel, padding=padding, groups=1)

    # 👈 如果原始输入是 [B, T]，去掉 channel
    if added_channel:
        smoothed = smoothed.squeeze(1)

    return smoothed
    
def boost_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Filter the lowpass frequency from the waveform"""
    return tensor * (1 + amount / 100)

def duck_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Mask input wav with some ducked signnals"""
    return tensor * (1 - amount / 100)



    