from typing import List
import numpy as np
import torch
import yaml
import os
from TAF.models.SteganographyMethod import SteganographyMethod
from .model.conv2_mel_modules2 import Encoder, Decoder

class TimbreMethod(SteganographyMethod):
    _model_cache = {}  # 缓存已加载的模型，避免重复加载

    def __init__(self):
        self.process_yaml = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/TAF/steganography_methods/config/process.yaml"
        self.model_yaml = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/TAF/steganography_methods/config/model.yaml"
        self.train_yaml = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/TAF/steganography_methods/config/train.yaml"
        self.model_path = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/TAF/steganography_methods/ckpt/"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 标记模型是否已加载
        self.encoder = None
        self.decoder = None
        self._loaded = False

    def _load_model_if_needed(self):
        """懒加载：仅在首次 encode/decode 时加载模型，并做缓存"""
        if self._loaded:
            return

        # 如果缓存里有，直接用缓存
        if "timbre" in TimbreMethod._model_cache:
            self.encoder, self.decoder = TimbreMethod._model_cache["timbre"]
            self._loaded = True
            return

        # === 加载配置文件 ===
        self.process_config = yaml.load(open(self.process_yaml, "r"), Loader=yaml.FullLoader)
        self.model_config = yaml.load(open(self.model_yaml, "r"), Loader=yaml.FullLoader)
        self.train_config = yaml.load(open(self.train_yaml, "r"), Loader=yaml.FullLoader)
        self.model_config["test"] = {"model_path": self.model_path, "index": -1}

        # === 构建模型 ===
        win_dim = self.process_config["audio"]["win_len"]
        embedding_dim = self.model_config["dim"]["embedding"]
        nlayers_encoder = self.model_config["layer"]["nlayers_encoder"]
        nlayers_decoder = self.model_config["layer"]["nlayers_decoder"]
        attention_heads_encoder = self.model_config["layer"]["attention_heads_encoder"]
        attention_heads_decoder = self.model_config["layer"]["attention_heads_decoder"]
        msg_length = self.train_config["watermark"]["length"]

        encoder = Encoder(self.process_config, self.model_config, msg_length, win_dim, embedding_dim,
                          nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(self.device)
        decoder = Decoder(self.process_config, self.model_config, msg_length, win_dim, embedding_dim,
                          nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(self.device)

        # === 只取最新 ckpt，避免遍历全部 ===
        ckpt_list = sorted(os.listdir(self.model_path))
        ckpt_file = os.path.join(self.model_path, ckpt_list[-1])  # 最新文件
        state_dict = torch.load(ckpt_file, map_location=self.device)

        encoder.load_state_dict(state_dict["encoder"])
        decoder.load_state_dict(state_dict["decoder"], strict=False)
        encoder.eval()
        decoder.eval()
        decoder.robust = False

        self.encoder = encoder
        self.decoder = decoder
        self._loaded = True

        # 缓存模型，下次不用重新加载
        TimbreMethod._model_cache["timbre"] = (self.encoder, self.decoder)

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        self._load_model_if_needed()  # 确保模型已加载
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        msg = torch.from_numpy(np.array(message)).float().unsqueeze(0).unsqueeze(1).to(self.device) * 2 - 1
        with torch.no_grad():
            encoded, _ = self.encoder.test_forward(x, msg, strength_factor=1.1)
        return encoded.squeeze().cpu().numpy()

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        self._load_model_if_needed()  # 确保模型已加载
        x = torch.tensor(data_with_watermark, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoded = self.decoder.test_forward(x)
        return ((decoded >= 0).int().cpu().numpy() + 0).flatten().tolist()[:watermark_length]

    def type(self) -> str:
        return "timbre"
