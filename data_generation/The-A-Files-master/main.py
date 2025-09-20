import os
import argparse
import random
from collections import defaultdict
from typing import List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm
# === 你已有的模块 ===
from evaluateWatermark import evaluate_watermark
from TAF.models.WavFile import WavFile
from TAF.steganography_methods.factory import SteganographyMethodFactory
from TAF.generator.generator import generate_random_message
from performattack import apply_configurable_attacks
from wavmark.utils import file_reader

# -------------------------
# 基础工具
# -------------------------
def load_files(path: str) -> List[WavFile]:
    loader = WavFileLoader(Path(path))
    return [loader.load(os.path.basename(file_path)) for file_path in glob.iglob(path + '/*.wav')]


def resolve_audio_path(audio_dir: str, filename_no_ext: str, extensions=[".flac", ".wav"]) -> str:
    for ext in extensions:
        path = os.path.join(audio_dir, filename_no_ext + ext)
        if os.path.exists(path):
            return path
    return None


def load_protocol(protocol_path: str):
    protocol_entries = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            filename = parts[0]
            # 兼容多种列数：优先第3列，其次第2列，最后给默认标签
            if len(parts) >= 3:
                label = parts[2]
            elif len(parts) >= 2:
                label = parts[2]
            else:
                label = "any"
            protocol_entries.append((filename, label))
            # if len(protocol_entries) >= 1000:   # 你原来的上限；不要就删掉这行
            #     break
    if not protocol_entries:
        raise ValueError(f"[Protocol] 解析到 0 行：{protocol_path}。请检查路径与列格式。")
    return protocol_entries


# ✅ 已有工具可重采样
def load_files_by_protocol(audio_dir: str, protocol_path: str, target_sr=16000, num_threads: int = 32) -> List[Tuple[WavFile, str]]:
    protocol = load_protocol(protocol_path)

    def load_one(entry):
        filename, label = entry
        audio_path = os.path.join(audio_dir, filename)

        if not os.path.exists(audio_path):
            base_name = os.path.splitext(filename)[0]
            audio_path_resolved = resolve_audio_path(audio_dir, base_name)
            if audio_path_resolved is None:
                print(f"[!] File not found: {filename}")
                return None
            audio_path = audio_path_resolved

        try:
            samples = file_reader.read_as_single_channel(audio_path, aim_sr=target_sr)
            wavfile = WavFile(
                path=audio_path,
                samples=samples,
                samplerate=target_sr
            )
            return (wavfile, label)
        except Exception as e:
            print(f"[!] Failed to load {audio_path}: {e}")
            return None

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(load_one, entry) for entry in protocol]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel loading files"):
            result = future.result()
            if result:
                results.append(result)

    return results


def save_audio(base_dir: str, orig_path: str, samples: np.ndarray, samplerate: int, ext: str = ".flac"):
    """按原文件名保存到 base_dir，默认 .flac"""
    filename = os.path.splitext(os.path.basename(orig_path))[0] + ext
    os.makedirs(base_dir, exist_ok=True)
    sf.write(os.path.join(base_dir, filename), np.asarray(samples, dtype=np.float32), samplerate)


# -------------------------
# 处理（嵌入 → 攻击 → 解码 → 评估）
# -------------------------

def process_file(
    file: WavFile,
    label: str,
    mode: str,
    method_cache: dict,
    attack_list: List[str],
    embed_count: dict,
    embedded_path: str = None,
    save_embedded: bool = False,
) -> List[dict]:
    """
    对单个文件执行：按方法生成随机比特 → encode → 多攻击/单攻击 → decode → evaluate
    返回：[{Attack, Method, BER, SNR, PESQ}, ...]
    """
    results = []

    for method in method_cache[file.samplerate]:
        
        if (method.type() != label) and (mode != "all_attacks"):
            continue

        
        print(method.type(),label)
        # --- 随机水印长度规则（保持你原来的逻辑） ---
        if method.type() in ("wavmark", "audioseal"):
            secret_msg = np.random.choice([0, 1], size=16)
        elif method.type() == "timbre":
            secret_msg = np.random.choice([0, 1], size=10)
        else:
            secret_msg = generate_random_message(length=random.randint(8, 32))

        try:
            
            secret_data = method.encode(file.samples, secret_msg)
            # secret_data = method.encode(secret_data, secret_msg)
            
            embed_count[method.type()] += 1
            
            # all_attacks 模式是否保存嵌入音频
            if save_embedded and embedded_path:
                save_audio(embedded_path, file.path, secret_data, file.samplerate, ext=".flac")
            
            # 选择要应用的攻击集合
            attacks_to_apply = attack_list 
            
            if mode == "embed_only":
                attacks_to_apply = ["original"]
                return 0
            for attack_name in attacks_to_apply:
                if attack_name != 'original':
                    attacked_data = apply_configurable_attacks(
                        torch.tensor(secret_data).unsqueeze(0),
                        file.samplerate,
                        [attack_name]
                    ).squeeze(0).cpu().numpy()
                else:
                    attacked_data = secret_data
                
                # protocol_attack 模式下保存攻击后的音频
                
                # 解码 + 评估
                
                decoded_message = method.decode(attacked_data, len(secret_msg))


                with open("/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/message_comparison_log.txt", 'a') as f:
                            orig_str = ''.join(str(bit) for bit in secret_msg)
                            dec_str = ''.join(str(bit) for bit in decoded_message)
                            f.write(f"attack: {attack_name}\n")
                            f.write(f"original: {orig_str}\n")
                            f.write(f"decoded : {dec_str}\n")
                            f.write(f"match   : {['✓' if o == d else 'x' for o, d in zip(orig_str, dec_str)]}\n\n")
                
                metrics = evaluate_watermark(
                    clean_wav=torch.tensor(file.samples).unsqueeze(0).unsqueeze(0).float(),
                    watermarked_wav=torch.tensor(attacked_data).unsqueeze(0).unsqueeze(0).float(),
                    sample_rate=file.samplerate,
                    original_bits=secret_msg,
                    decoded_bits=decoded_message
                )

                # ---- 兼容你原本的 full_metrics 口味（如你仍想保留）----
                # for metric_name in ['ber', 'snr', 'pesq']:
                #     full_metrics[attack_name][method.type()][metric_name].append(metrics[metric_name])

                # ✅ 关键：收集扁平结果行，后面聚合生成与原来一致的 MultiIndex CSV
                results.append({
                    "Attack": attack_name,
                    "Method": method.type(),
                    "BER": metrics["ber"],
                    "SNR": metrics["snr"],
                    "PESQ": metrics["pesq"]
                })

        except Exception as e:
            logger.opt(exception=e).error("Error in method {} ", method.type())


    return results


# -------------------------
# 主程序
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all_attacks", "protocol_attack","embed_only",'attack_original_only'], default="all_attacks",
                        help="all_attacks：每个样本遍历所有攻击; protocol_attack：按 protocol 指定单一攻击")
    parser.add_argument("--threads", type=int, default=16, help="并行线程数（加载+处理）")
    parser.add_argument("--limit", type=int, default=-1, help="最多处理多少条（-1 表示全部）")
    

    # 路径（提供你给的默认值）
    
    parser.add_argument("--new_protocol_path", type=str,
                        default="/DATA1/zhangzs/SOTA_paper/The-A-Files-master/protocols/tagged/ITW/protocol_wm_75_only_attacked.txt")
    parser.add_argument("--audio_dir", type=str,
                        default="/DATA1/zhangzs/SOTA_paper/The-A-Files-master/ITW_ALL_watermark")
    parser.add_argument("--embedded_path", type=str,
                        default="/DATA1/zhangzs/SOTA_paper/The-A-Files-master/ITW_ALL_attacked")
    

    # 输出
    parser.add_argument("--output", type=str, help="聚合后的 MultiIndex CSV")
    parser.add_argument("--output_raw", type=str, default="ITW_ALL_raw.csv", help="明细 CSV（每条记录一行）")

    # 协议生成


    # 其它
    parser.add_argument("--save_embedded", action="store_true",
                        help="在 all_attacks 模式下也保存 encode 后的音频（默认只在 protocol_attack 保存攻击后的音频）")
    parser.add_argument("--num_shards", type=int, default=1, help="将 protocol 顺序切成多少份")
    parser.add_argument("--shard_id", type=int, default=0, help="选择第几个分片 (0-based)")

    args = parser.parse_args()

    
    print(args.mode)
    print(args)
    # 攻击列表（与现有评估保持一致）
    attack_list = [
    'original',      # ✅ 原始对照
    'lp',            # ✅ 低通滤波（常见失真/带宽限制）
    'resample',      # ✅ 重采样（传输/压缩常见）
    'amp',           # ✅ 幅度缩放（音量变化）
    'pink',          # ✅ 粉噪声（背景噪声场景）
    # 'hp',           # 高通（和lp功能重叠，可注释掉）
    # 'smooth',       # 平滑（轻微失真，弱代表性）
    'boost',         # ✅ 增益（和amp相对，过驱/过饱和场景）
    # 'duck',         # 压缩（场景较特殊，可以先不用）
    'band',          # ✅ 带通/带阻（电话/语音常见）
    'stretch',       # ✅ 时域拉伸（时间失真）
    'speed',         # ✅ 变速（保持调子）
    # 'speed_pitch',  # 变速变调（和 speed 接近，可不用）
    # 'specaug',      # 谱增强（偏向训练 trick，不是真攻击）
    "lossy",         # ✅ MP3 压缩（最典型的有损压缩）
]


    # 加载
    all_files = load_files_by_protocol(args.audio_dir, args.new_protocol_path, num_threads=args.threads)
    if args.limit > 0:
        all_files = all_files[:args.limit]
    if not all_files:
        print("没有可处理的文件，请检查路径/协议。")
        return

    # 方法缓存（按采样率）
    method_names = [m.type() for m in SteganographyMethodFactory.get_all(all_files[0][0].samplerate)]
    method_cache = {sr: SteganographyMethodFactory.get_all(sr) for sr in set(f[0].samplerate for f in all_files)}
    # for m in SteganographyMethodFactory.get_all(sr):
    #     print("METHOD:", m.type(), "ENCODER_ID:", id(getattr(m, "encoder", None)))
    for sr, methods in method_cache.items():
        for m in methods:
            print("SR:", sr, "METHOD:", m.type(), "ENCODER_ID:", id(getattr(m, "encoder", None)))

    embed_count = defaultdict(int)
    all_results = []

    # 多线程处理每个文件
    def task(triple):
        file, label= triple
        return process_file(
            file=file,
            label=label,
            mode=args.mode,
            method_cache=method_cache,
            attack_list=attack_list,
            embed_count=embed_count,
            embedded_path=args.embedded_path,
            save_embedded=args.save_embedded
        )

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = [ex.submit(task, triple) for triple in all_files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Embedding+Attacking"):
            if args.mode == "all_attacks":
                all_results.extend(fut.result())
            else:
                all_results.append(fut.result())  # ✅ 直接加单个值
    if args.output:
        df_raw = pd.DataFrame(all_results)
        raw_path = args.output.replace(".csv", "_raw.csv")
        df_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

        # 聚合成多层表格
        if not df_raw.empty:
            df_pivot = df_raw.groupby(["Attack", "Method"]).mean(numeric_only=True).unstack("Method")
            # 统一列名为 MultiIndex(Method, Metric)
            df_pivot.columns = pd.MultiIndex.from_tuples(df_pivot.columns, names=["Method", "Metric"])
            df_pivot.to_csv(args.output, encoding="utf-8-sig")
        else:
            print("Warning: df_raw is empty. Skip pivot saving.")

        print("\n📊 各水印方法嵌入文件数统计：")
        for method, count in embed_count.items():
            print(f"  - {method:20s}: {count} files")
        print(f"\n✅ 原始结果已保存到: {raw_path}")
        print(f"✅ 聚合表格已保存到: {args.output}")

    
    

if __name__ == "__main__":
    main()
