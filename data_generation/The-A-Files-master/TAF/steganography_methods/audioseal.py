from typing import List
import numpy as np
import torch
from TAF.models.SteganographyMethod import SteganographyMethod
from audioseal import AudioSeal

class AudioSealMethod(SteganographyMethod):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = AudioSeal.load_generator("/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/TAF/ckpt/generator_base.pth",nbits= 16).to(self.device)
        self.detector = AudioSeal.load_detector("/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/TAF/ckpt/detector_base.pth",nbits= 16).to(self.device)
        self.max_bits = self.generator.msg_processor.nbits  # 通常为 16

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        assert data.ndim == 1, "Expected mono waveform"
        signal = torch.tensor(data, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, T]

        # 将 message 转换为 tensor，不做补齐或裁剪
        msg_tensor = torch.tensor(message, dtype=torch.float32, device=self.device).unsqueeze(0)

        watermark = self.generator.get_watermark(signal, message=msg_tensor,sample_rate = 16_000)
        watermarked = signal + watermark

        return watermarked.squeeze().detach().cpu().numpy()

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        assert data_with_watermark.ndim == 1, "Expected mono waveform"
        signal = torch.tensor(data_with_watermark, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        _, message = self.detector.detect_watermark(signal, 16000)

        if message is None or message.numel() == 0:
            return [0] * watermark_length

        decoded = (message.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8).tolist()
        return decoded[:watermark_length]

    def type(self) -> str:
        return "audioseal"
