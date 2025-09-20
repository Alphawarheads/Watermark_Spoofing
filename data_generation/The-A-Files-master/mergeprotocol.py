import os
import re
from collections import defaultdict
from typing import Dict, List

PATTERN = re.compile(r"^(?P<prefix>.+?)_part_(?P<part>\d+)\.txt$")  # 前缀_part数字.txt

def ensure_unique_path(path: str) -> str:
    print("""若文件已存在，生成不重名的新路径：xxx_v001.txt / v002 …""")

    base, ext = os.path.splitext(path)
    cand = path
    i = 1
    while os.path.exists(cand):
        cand = f"{base}_v{i:03d}{ext}"
        i += 1
    return cand

def merge_parts_by_prefix(
    in_dir: str,
    out_dir: str,
    encoding: str = "utf-8",
    newline_between: bool = False,
    overwrite: bool = False,
) -> Dict[str, str]:
    """
    将 in_dir 下形如 前缀_partN.txt 的分片按前缀合并到 out_dir/前缀.txt
    返回: {前缀: 输出文件路径}
    """
    os.makedirs(out_dir, exist_ok=True)

    groups: Dict[str, List[tuple]] = defaultdict(list)  # prefix -> list[(part_num, filepath)]
    for name in os.listdir(in_dir):
        if not name.endswith(".txt"):
            continue
        m = PATTERN.match(name)
        if not m:
            continue
        prefix = m.group("prefix")
        part = int(m.group("part"))
        groups[prefix].append((part, os.path.join(in_dir, name)))

    outputs: Dict[str, str] = {}
    for prefix, parts in groups.items():
        parts.sort(key=lambda x: x[0])  # 按 part 数字升序
        out_path = os.path.join(out_dir, f"{prefix}.txt")
        if not overwrite:
            out_path = ensure_unique_path(out_path)

        with open(out_path, "w", encoding=encoding, newline="") as w:
            for i, (_, p) in enumerate(parts):
                with open(p, "r", encoding=encoding, errors="ignore") as r:
                    data = r.read()
                if data and not data.endswith("\n"):
                    data += "\n"
                w.write(data)
                if newline_between and i != len(parts) - 1:
                    w.write("\n")  # 组间额外空一行（可选）
        outputs[prefix] = out_path

    return outputs

# ====== 入口 ======
if __name__ == "__main__":
    # 直接在这里改参数即可
    IN_DIR = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/LAscores"    # 含 *_partN.txt 的文件夹
    OUT_DIR = "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/test2"  # 合并后的输出目录
    print(1)
    # 可选开关
    ENCODING = "utf-8"
    NEWLINE_BETWEEN = False  # 各 part 之间是否加一空行
    OVERWRITE = False        # 是否允许覆盖已存在输出；False=自动生成 _v001 递增

    results = merge_parts_by_prefix(
        in_dir=IN_DIR,
        out_dir=OUT_DIR,
        encoding=ENCODING,
        newline_between=NEWLINE_BETWEEN,
        overwrite=OVERWRITE,
    )
    # 简单回显
    for k, v in results.items():
        print(f"{k} -> {v}")
