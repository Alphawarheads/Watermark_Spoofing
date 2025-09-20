from typing import List
import numpy as np
import torch
import wavmark
from TAF.models.SteganographyMethod import SteganographyMethod

class WavMarkMethod(SteganographyMethod):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = wavmark.load_model().to(self.device)

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        signal = data  # ⬅️ 数据已由主逻辑强制为 16kHz mono
        if not isinstance(message, np.ndarray):
            message = np.array(message, dtype=np.uint8)
        watermarked_signal, _ = wavmark.encode_watermark(self.model, signal, message, show_progress=False)
        print(message.tolist())
        return watermarked_signal

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        signal = data_with_watermark
        decoded_message, _ = wavmark.decode_watermark(self.model, signal, watermark_length, show_progress=False)
        print(decoded_message.tolist())
        return decoded_message.tolist()

    def type(self) -> str:
        return "wavmark"
