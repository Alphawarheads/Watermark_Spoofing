#!/usr/bin/env python3
# copy_by_protocol.py
import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

def resolve_src_path(root: str, utt_id: str, exts: List[str]) -> Optional[str]:
    """
    在 root 下为给定 utt_id 尝试多种扩展名，返回存在的第一个路径。
    若 utt_id 已自带扩展名，也会直接检查。
    """
    base, ext = os.path.splitext(utt_id)
    if ext:
        p = os.path.join(root, utt_id)
        if os.path.exists(p):
            return p
        utt_id = base  # 去掉后缀继续尝试

    for e in exts:
        p = os.path.join(root, f"{utt_id}{e}")
        if os.path.exists(p):
            return p
    return None

def copy_one(row, A: str, B: str, C: str, exts: List[str], overwrite: bool, prefix: str) -> Tuple[str, str, str, str, str, str]:
    """
    执行一次复制：
    返回 (status, utt_id, src_path, dest_name, label, method)
      status in {"ok", "skip-exists", "missing-src", "error"}
    """
    utt_id, label, method = row["utt_id"], row["label"], row["method"]
    src_root = B if method == "original" else A

    src_path = resolve_src_path(src_root, str(utt_id), exts)
    if src_path is None:
        return ("missing-src", str(utt_id), "", "", label, method)

    # 目标文件名：加前缀（仅文件名，包含原扩展名）
    dest_name = os.path.basename(src_path)
    if prefix:
        dest_name = f"{prefix}{dest_name}"

    dest_path = os.path.join(C, dest_name)

    if os.path.exists(dest_path) and not overwrite:
        return ("skip-exists", str(utt_id), src_path, dest_name, label, method)

    try:
        os.makedirs(C, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        return ("ok", str(utt_id), src_path, dest_name, label, method)
    except Exception:
        return ("error", str(utt_id), src_path, dest_name, label, method)

def main():
    ap = argparse.ArgumentParser(description="按 protocol 第三列(original/非original) 从 A/B 拷到 C，重命名加前缀，并生成新 protocol C")
    ap.add_argument("--protocol", required=True, help="protocol 路径（3列：utt_id label method）")
    ap.add_argument("--src_nonoriginal", required=True, help="A 文件夹：非 original 音频所在目录")
    ap.add_argument("--src_original", required=True, help="B 文件夹：original 音频所在目录")
    ap.add_argument("--dst", required=True, help="C 目标目录")
    ap.add_argument("--exts", nargs="+", default=[".flac", ".wav"], help="尝试的音频后缀顺序")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="并行拷贝线程数")
    ap.add_argument("--overwrite", action="store_true", help="若目标已存在则覆盖（默认跳过）")
    ap.add_argument("--prefix", type=str, default="", help="为复制到 C 的所有文件名添加的前缀，如 wm75_")
    ap.add_argument("--out_protocol", type=str, default="", help="输出的新 protocol C 路径（默认写到 C/protocol_c.txt）")
    ap.add_argument("--protocol_no_ext", action="store_true", help="写 protocol C 时第一列不带扩展名")
    args = ap.parse_args()

    df = pd.read_csv(args.protocol, sep=r"\s+", header=None, names=["utt_id", "label", "method"])
    df = df.drop_duplicates(subset=["utt_id"], keep="first").reset_index(drop=True)

    out_proto_path = args.out_protocol or os.path.join(args.dst, "protocol_c.txt")
    proto_rows = []  # (new_utt_id_for_protocol, label, method)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(copy_one, row, args.src_nonoriginal, args.src_original, args.dst, args.exts, args.overwrite, args.prefix)
            for _, row in df.iterrows()
        ]
        ok = skip = miss = err = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Copying"):
            status, utt_id, src, dest_name, label, method = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip-exists":
                skip += 1
            elif status == "missing-src":
                miss += 1
            else:
                err += 1

            # 只在成功或跳过（文件已存在）时记录到新 protocol
            if status in {"ok", "skip-exists"} and dest_name:
                if args.protocol_no_ext:
                    dest_base, _ = os.path.splitext(dest_name)
                    new_utt = dest_base
                else:
                    new_utt = dest_name  # 含扩展名，和 C 中实际文件名一致
                proto_rows.append((new_utt, label, method))

    # 写出新的 protocol C
    if proto_rows:
        os.makedirs(os.path.dirname(out_proto_path), exist_ok=True)
        out_df = pd.DataFrame(proto_rows, columns=["utt_id", "label", "method"])
        out_df.to_csv(out_proto_path, sep=" ", index=False, header=False)
        print(f"\n📝 新 protocol C 已生成：{out_proto_path}（{len(out_df)} 行）")
    else:
        print("\n⚠️ 未生成 protocol C（没有成功/已存在的目标文件可记录）。")

    total = len(df)
    print("\n== Summary ==")
    print(f"Protocol rows : {total}")
    print(f"Copied        : {ok}")
    print(f"Skipped exist : {skip}")
    print(f"Missing src   : {miss}")
    print(f"Errors        : {err}")
    if miss:
        print("提示：检查 --exts 顺序是否与实际文件后缀匹配，或 A/B 路径是否正确。")

if __name__ == "__main__":
    main()
