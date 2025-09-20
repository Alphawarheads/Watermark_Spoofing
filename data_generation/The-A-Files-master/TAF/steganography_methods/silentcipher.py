from typing import List, Optional
import numpy as np
import bitstring

# 依赖：pip install silentcipher
import silentcipher

# 可选重采样后端：优先 torchaudio，其次 librosa，否则报错
def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav
    # 保证 1D
    wav = np.asarray(wav).squeeze()
    try:
        import torchaudio
        import torch
        tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        res = torchaudio.functional.resample(tensor, src_sr, dst_sr)
        return res.squeeze(0).numpy()
    except Exception:
        try:
            import librosa
            return librosa.resample(wav.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr)
        except Exception as e:
            raise RuntimeError(
                "Need torchaudio or librosa for resampling. Please install one of them."
            ) from e

def _ensure_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:  # (channels, time) or (time, channels)
        # 尝试统一到 (time,)
        if x.shape[0] < x.shape[1]:
            x = x.mean(axis=0)
        else:
            x = x.mean(axis=1)
    return x.squeeze()

def _clip_to_unit(wav: np.ndarray) -> np.ndarray:
    return np.clip(wav, -1.0, 1.0)

def _bits_to_five_bytes(bits: List[int]) -> List[int]:
    """把 bit 列表(0/1)转成 5 个 8bit 整数；不足 40bit 自动 0 填充，多余截断。"""
    b = [1 if v else 0 for v in bits[:40]]
    if len(b) < 40:
        b += [0] * (40 - len(b))
    out = []
    for i in range(5):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | b[i * 8 + j]
        out.append(byte_val)
    return out

def _five_bytes_to_bits(bytes5: List[int], out_len: int) -> List[int]:
    """把 5 个 8bit 整数还原为 bit 列表，并按 out_len 截断/填充。"""
    ba = bitstring.BitArray()
    for v in bytes5[:5]:
        ba.append(bitstring.Bits(uint=v, length=8))
    bits = [int(b) for b in ba.bin]  # 长度 40
    if out_len <= 40:
        return bits[:out_len]
    else:
        return bits + [0] * (out_len - 40)


class SilentCipherMethod(SteganographyMethod):
    """
    TAF 适配器：
    - 输入/输出严格符合 TAF 的 SteganographyMethod 接口
    - 内部自动做采样率适配（到 16k 或 44.1k），再还原
    - 消息：输入为 bit 列表（0/1），内部转成 SilentCipher 需要的 5×8bit 整数
    """

    def __init__(
        self,
        input_sr: int = 16000,
        model_type: str = "16k",     # 备选："44.1k"
        device: str = "cuda",        # 或 "cpu"
        message_sdr: Optional[float] = None,
        phase_shift_decoding: bool = False,
    ):
        self.input_sr = int(input_sr)
        if model_type not in ("16k", "44.1k"):
            raise ValueError("model_type must be '16k' or '44.1k'")
        self.model_type = model_type
        self.model_sr = 16000 if model_type == "16k" else 44100
        self.device = device
        self.message_sdr = message_sdr
        self.phase_shift_decoding = phase_shift_decoding

        # 加载 SilentCipher 模型
        self.model = silentcipher.get_model(model_type=self.model_type, device=self.device)

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """
        data: np.ndarray, 单声道或多声道，值域建议[-1,1]，采样率 = self.input_sr
        message: List[int]，bit 列表(0/1)，长度任意；内部按 40bit 截断/填充
        return: 加水印后的音频，采样率与输入相同
        """
        wav_in = _ensure_mono(np.asarray(data, dtype=np.float32))
        # 重采样到模型采样率
        wav_model = _resample(wav_in, self.input_sr, self.model_sr)

        # bits → 5字节
        bytes5 = _bits_to_five_bytes(message)

        # 调用 SilentCipher
        if self.message_sdr is None:
            encoded, _sdr = self.model.encode_wav(wav_model, self.model_sr, bytes5)
        else:
            encoded, _sdr = self.model.encode_wav(
                wav_model, self.model_sr, bytes5, message_sdr=float(self.message_sdr)
            )

        # 还原到输入采样率
        encoded_out = _resample(encoded, self.model_sr, self.input_sr)
        return _clip_to_unit(encoded_out).astype(data.dtype if isinstance(data, np.ndarray) else np.float32)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """
        data_with_watermark: np.ndarray，采样率 = self.input_sr
        watermark_length: 需要输出的 bit 数（返回严格匹配这个长度；>40 则补零）
        return: List[int] of {0,1}
        """
        wav_in = _ensure_mono(np.asarray(data_with_watermark, dtype=np.float32))
        wav_model = _resample(wav_in, self.input_sr, self.model_sr)

        result = self.model.decode_wav(
            wav_model, self.model_sr, phase_shift_decoding=bool(self.phase_shift_decoding)
        )

        # 容错：若未解出，返回全 0
        if not result.get("messages") or len(result["messages"]) == 0:
            return [0] * int(watermark_length)

        bytes5 = result["messages"][0]  # 长度应为 5
        bits = _five_bytes_to_bits(bytes5, int(watermark_length))
        return bits

    def type(self) -> str:
        return "silentcipher"
