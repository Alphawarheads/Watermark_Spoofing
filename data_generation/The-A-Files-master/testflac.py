import os
from pathlib import Path
from typing import List
import soundfile as sf
import torchaudio
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # ✅ 加入 tqdm

def load_audio(path: Path):
    """使用 torchaudio 读取音频文件。"""
    samples, fs = sf.read(str(path))
    return samples, fs

def validate_file(path: Path):
    """用于线程池中的单个任务，返回 (文件名, 是否成功, 错误信息)"""
    try:
        _ = load_audio(path)
        return (path.name, True, "")
    except Exception as e:
        return (path.name, False, str(e))

def validate_folder_multithreaded(folder_path: str, extensions: List[str] = [".wav", ".flac"], num_threads: int = 16):
    folder = Path(folder_path)
    audio_files = [f for f in folder.rglob("*") if f.suffix.lower() in extensions]

    success_files = []
    failed_files = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(validate_file, file): file for file in audio_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating audio"):
            filename, success, error_msg = future.result()
            if success:
                success_files.append(filename)
            else:
                print(f"[!] Failed to read: {filename} | Error: {error_msg}")
                failed_files.append(filename)

    print("\n✅ 成功读取的文件数:", len(success_files))
    print("❌ 失败的文件数:", len(failed_files))

    if failed_files:
        failed_path = Path("failed_files.txt")
        with open(failed_path, "w") as f:
            for name in failed_files:
                f.write(name + "\n")
        print(f"\n🚨 已保存失败文件列表到: {failed_path.resolve()}")

# ==== 示例使用 ====
if __name__ == "__main__":
    test_folder = "/DATA1/Audiodata/AntiSpoofingData/asvspoof2019/ASVspoof2019_LA_train/flac"  # 替换为你的音频文件夹路径
    validate_folder_multithreaded(test_folder, num_threads=32)
