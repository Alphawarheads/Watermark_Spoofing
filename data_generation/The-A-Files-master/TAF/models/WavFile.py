from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List
from tqdm import tqdm
import numpy as np
import soundfile as sf


@dataclass
class WavFile:
    samplerate: int
    samples: np.ndarray
    path: Path

    @staticmethod
    def load(path: Path) -> WavFile:
        samples, fs = sf.read(path, dtype='float32')
        return WavFile(samplerate=fs, samples=samples, path=path)

    def save_steganography_file(self, a_samples: np.ndarray, suffix: str | None = None):
        sf.write(file=self._steganography_filename(suffix), data=a_samples, samplerate=self.samplerate)

    def _steganography_filename(self, suffix: str | None = None):
        return "{0}_{2}{1}".format(self.path.stem, self.path.suffix, "_stego" + (suffix if suffix is not None else ""))


# ✅ 放在类外部的多线程加载函数
def load_files(path: str, num_threads: int = 32) -> List[WavFile]:
    file_paths = list(Path(path).rglob("*.flac"))
    results = []

    def load_one(p: Path):
        try:
            return WavFile.load(p)
        except Exception as e:
            print(f"[!] Failed to load {p}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(load_one, p): p for p in file_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading audio files"):
            result = future.result()
            if result:
                results.append(result)

    return results
