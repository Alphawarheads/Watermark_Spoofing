from typing import List
import numpy as np
from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography_methods.common.mixer import mixer

class DsssMethod(SteganographyMethod):
    """
    Direct Sequence Spread Spectrum (DSSS) audio watermarking.
    """

    # ==== 新增：把扩频码生成方法放进类里 ====
    def _generate_spreading_code(self, length: int, seed: int = 42) -> np.ndarray:
        """
        生成长度为 length 的 ±1 扩频序列（float32）。
        """
        rng = np.random.default_rng(seed)
        chips = rng.choice([-1, 1], size=int(length)).astype(np.float32, copy=False)
        return chips

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """
        Args:
            data: 1D audio waveform, 任意 dtype（会转 float32 做计算）
            message: 比特序列（0/1）
        Returns:
            加水印后的音频，长度与 data 相同
        """
        x = np.asarray(data, dtype=np.float32, order="C")

        L_min = 8 * 1024  # 每段最小长度
        if len(message) == 0:
            # 空消息：直接返回原音频
            return x.copy()

        L2 = np.floor(len(x) / len(message))
        L = int(max(L_min, L2))

        nframe_f = np.floor(len(x) / L)
        nframe = int(nframe_f)
        # 取 8 的倍数，防止后续某些实现里按8帧对齐
        N = nframe - (nframe % 8)
        if N <= 0:
            # 音频太短或参数不合理
            return x.copy()

        alpha = 0.005  # 嵌入强度

        # 准备 N 个比特：不足就循环重复消息（比全填0更稳妥）
        if len(message) >= N:
            bits = np.array(message[:N], dtype=np.int8)
        else:
            reps = int(np.ceil(N / len(message)))
            bits = np.tile(np.asarray(message, dtype=np.int8), reps)[:N]

        # 生成长度为 L 的扩频码，并在帧维度上平铺到 N*L
        r = self._generate_spreading_code(L, seed=42)        # shape: [L]
        pr = np.tile(r, N)                                   # shape: [N*L]

        # 用 mixer 把 N 个比特扩展到 N*L 的序列（确保 shape 对齐）
        # mixer 的第一个返回值应为 shape [N*L]（或能 reshape 成 [N, L] 再展平）
        mix = mixer(L, bits.tolist(), -1, 1, 256)[0]         # 你原本的用法
        mix = np.asarray(mix, dtype=np.float32)
        if mix.ndim > 1:
            mix = mix.reshape(-1)
        if mix.size != N * L:
            # 如果 mixer 输出不是 N*L，就退而求其次：按 DSSS 经典做法扩展比特后逐点相乘
            bip_bits = (bits.astype(np.int8) * 2 - 1).astype(np.float32)  # {-1,+1}
            mix = np.repeat(bip_bits, L)  # shape: [N*L]

        # 叠加水印
        host = x[: N * L]
        stego_part = host + alpha * mix * pr
        out = np.concatenate([stego_part, x[N * L :]], axis=0)
        return out

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """
        解码 watermark，比特阈值化：相关 > 0 -> 1，否则 0
        """
        y = np.asarray(data_with_watermark, dtype=np.float32, order="C")
        if watermark_length <= 0:
            return []

        L_min = 8 * 1024
        L2 = np.floor(len(y) / max(watermark_length, 1))
        L = int(max(L_min, L2))

        nframe = len(y) // L
        N = nframe - (nframe % 8)
        if N <= 0:
            return [0] * watermark_length

        # 切成帧做相关
        ys = y[: N * L].reshape(N, L).T          # [L, N]
        r = self._generate_spreading_code(L, 42) # [L]
        c = (r @ ys) / float(L)                  # [N]，相关均值

        bits_hat = (c > 0.0).astype(np.int8)     # 阈值化
        return bits_hat[:watermark_length].tolist()

    def type(self) -> str:
        return "dsss"
