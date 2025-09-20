import os
import random
from collections import defaultdict, Counter
from typing import List, Dict

def parse_label(parts: List[str]) -> str:
    # 兼容列数：优先第3列，其次第2列，否则给默认标签
    if len(parts) >= 3:
        return parts[2]
    elif len(parts) >= 2:
        return parts[1]
    else:
        return "any"

def split_protocol_stratified(protocol_path: str, num_parts: int, output_dir: str, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    # 读取并按标签分层
    groups: Dict[str, List[str]] = defaultdict(list)
    with open(protocol_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            label = parse_label(parts)
            groups[label].append(line)  # 保留换行符

    if not groups:
        raise ValueError(f"[Protocol] 解析到 0 行：{protocol_path}")

    # 初始化每份的容器
    parts_lines: List[List[str]] = [[] for _ in range(num_parts)]

    # 固定随机种子，组内打乱+均匀切分
    rng = random.Random(seed)
    for label, lines in groups.items():
        rng.shuffle(lines)
        base, rem = divmod(len(lines), num_parts)
        start = 0
        for i in range(num_parts):
            end = start + base + (1 if i < rem else 0)
            if start < end:
                parts_lines[i].extend(lines[start:end])
            start = end

    # 写出文件
    for i, lines in enumerate(parts_lines, 1):
        out_path = os.path.join(output_dir, f"protocol_part_{i}.txt")
        with open(out_path, "w") as out_f:
            out_f.writelines(lines)
        print(f"写入 {out_path}，共 {len(lines)} 行")

    # 打印每份的标签分布（诊断用）
    print("\n各份标签分布：")
    for i, lines in enumerate(parts_lines, 1):
        counter = Counter(parse_label(l.strip().split()) for l in lines)
        top = ", ".join(f"{k}:{v}" for k, v in sorted(counter.items()))
        print(f"  part {i:>2}: {top}")

if __name__ == "__main__":
    # protocol_path = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged/DF21/protocol_wm_75_only.txt"
    # num_parts = 12
    # output_dir = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged/DF21/split_stratified"

    protocol_path = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/split/protocol_wm_75_only.txt"   # 原 protocol 文件
    num_parts =  8                         # 要切分的份数
    output_dir = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/ori_split"

    # protocol_path = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/protocol_wm_75_only.txt"   # 原 protocol 文件
    # num_parts =  5                           # 要切分的份数
    # output_dir = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/split"


    # protocol_path = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/protocol_wm_75_only.txt"   # 原 protocol 文件
    # num_parts =  5                           # 要切分的份数
    # output_dir = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/split"

    split_protocol_stratified(protocol_path, num_parts, output_dir, seed=42)

    
