import torch
def calculate_snr(clean, noisy):
    noise = noisy - clean
    signal_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-9))
    return snr.item()

def calculate_pesq_score(clean, noisy, sr):
    # 降采样到 16000（pesq 只支持 8000 or 16000）
    transform = torchaudio.transforms.Resample(sr, 16000).to(clean.device)
    if sr != 16000:
        clean = transform(clean)
        noisy = transform(noisy)
        sr = 16000

    clean_np = clean[0].squeeze(0).cpu().numpy().astype(np.float32)
    noisy_np = noisy[0].squeeze(0).cpu().numpy().astype(np.float32)

    # clip to avoid overflow in PESQ
    clean_np = np.clip(clean_np, -1.0, 1.0)
    noisy_np = np.clip(noisy_np, -1.0, 1.0)

    min_len = min(len(clean_np), len(noisy_np))
    print(clean_np[:min_len].shape)
    return pesq(sr, clean_np[:min_len], noisy_np[:min_len], 'wb')

def multi_resolution_stft_loss(x, y, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
    """
    计算多分辨率 STFT 损失，用于衡量原音与生成音在时频域上的差异，
    损失采用 log-magnitude 的 L1 距离。
    x, y: shape 为 [B, 1, T] 的时域信号
    """
    total_loss = 0.0
    for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
        window = torch.hann_window(win_length).to(x.device)
        # STFT 得到复数谱，取绝对值后加对数再计算 L1 距离
        x_stft = torch.stft(x.squeeze(1), n_fft=fft_size, hop_length=hop_size, win_length=win_length,
                              window=window, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), n_fft=fft_size, hop_length=hop_size, win_length=win_length,
                              window=window, return_complex=True)
        loss = torch.mean(torch.abs(torch.log(torch.abs(x_stft) + 1e-7) -
                                      torch.log(torch.abs(y_stft) + 1e-7)))
        total_loss += loss
    return total_loss / len(fft_sizes)
    

def stft(data):
    """
    data: [B, 1, T] 或 [B, T]
    返回:
        complex_stft: [B, freq, time, 2]
        magnitude:    [B, freq, time]
    """
    n_fft = n_fft
    hop_length = hop_length
    window = torch.hann_window(n_fft).to(data.device)
    if data.dim() == 3:
        data = data.squeeze(1)
    tmp = torch.stft(data, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    complex_stft = torch.view_as_real(tmp)  # [B, freq, time, 2]

    # 计算 magnitude
    magnitude = torch.linalg.norm(complex_stft, dim=-1)  # 等价于 sqrt(real^2 + imag^2)
    magnitude = magnitude.permute(0,2,1)
    return tmp,magnitude
    
def istft(data_fft):
    """
    data_fft: [B, freq, time, 2]
    返回: [B, T] 的时域信号
    """
    n_fft = n_fft
    hop_length = hop_length
    data_fft = data_fft.contiguous()
    watermarked_mag = data_fft.permute(0, 2, 1)  
    window = torch.hann_window(n_fft).to(data_fft.device)
    return torch.istft(torch.view_as_complex(data_fft),
                       n_fft=n_fft,
                       hop_length=hop_length,
                       window=window,
                       return_complex=False)

def bce_loss(msg_decoded_logits,key_input_vector):
    loss_bce_fn = torch.nn.BCEWithLogitsLoss()
    Lm_bce = loss_bce_fn(msg_decoded_logits, key_input_vector)
    return Lm_bce
def evaluate_watermark(
    clean_wav: torch.Tensor,           # [1, 1, T] or [1, T]
    watermarked_wav: torch.Tensor,     # [1, 1, T] or [1, T]
    sample_rate: int,
    loss_tf=None,                      # Optional: TFLoudnessRatio module
    loss_spec=None,                   # Optional: MultiScaleMelSpectrogramLoss module
    original_bits: list = None,       # 添加：原始嵌入的比特
    decoded_bits: list = None         # 添加：从水印中提取出的比特
) -> dict:
    """
    对一对音频进行统一水印评估，输出 PESQ, SNR, STFT loss, MSSpec loss, BER 等。
    """
    import torch.nn.functional as F
    from pesq import pesq
    import torchaudio

    # === Step 0: 保证形状一致 ===
    if clean_wav.dim() == 2:
        clean_wav = clean_wav.unsqueeze(0)
    if watermarked_wav.dim() == 2:
        watermarked_wav = watermarked_wav.unsqueeze(0)

    # === Step 1: 对齐长度 ===
    T = min(clean_wav.shape[-1], watermarked_wav.shape[-1])
    clean_wav = clean_wav[..., :T]
    watermarked_wav = watermarked_wav[..., :T]

    # === Step 2: SNR ===
    snr_val = calculate_snr(clean_wav, watermarked_wav)

    # === Step 3: PESQ（仅限8k/16k）===
    pesq_val = -1.0
    # try:
    #     transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(clean_wav.device)
    #     c_res = transform(clean_wav)
    #     w_res = transform(watermarked_wav)
    #     pesq_val = pesq(16000,
    #                     c_res[0, 0].cpu().numpy().clip(-1, 1).astype('float32'),
    #                     w_res[0, 0].cpu().numpy().clip(-1, 1).astype('float32'),
    #                     'wb')
    # except Exception as e:
    #     print(f"[evaluate_watermark] PESQ 计算失败: {e}")

    # === Step 4: STFT Loss ===
    # stft_loss = multi_resolution_stft_loss(watermarked_wav, clean_wav).item()

    # # === Step 5: MSSpec Loss（可选）===
    # msspec_val = loss_spec(watermarked_wav, clean_wav).item() if loss_spec else -1.0

    # # === Step 6: Loudness Loss（可选）===
    # loudness_val = loss_tf(watermarked_wav, clean_wav).item() if loss_tf else -1.0

    # === Step 7: BER ===
    ber = -1.0
    if original_bits is not None and decoded_bits is not None:
        length = min(len(original_bits), len(decoded_bits))
        if length > 0:
            errors = sum(o != r for o, r in zip(original_bits[:length], decoded_bits[:length]))
            ber = errors / length

    return {
        "pesq": pesq_val,
        "snr": snr_val,
        # "stft_loss": stft_loss,
        # "msspec_loss": msspec_val,
        # "loudness_loss": loudness_val,
        "ber": ber
    }
