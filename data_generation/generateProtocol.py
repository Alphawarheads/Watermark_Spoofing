import pandas as pd
import numpy as np
import os
import random
import os
from pathlib import Path
from typing import Iterable, List
# from evaluate_scripts.evaluat_AI4T import evaluate_all_files_in_folder as AI4Teval
from evaluate_scripts.evaluate_2021_DF import evaluate_all_files_in_folder as DFeval
from evaluate_scripts.evaluate_2021_LA import evaluate_all_files_in_folder as LAeval
from evaluate_scripts.evaluate_in_the_wild import evaluate_all_files_in_folder as ITWeval
print("generating protocol")
import shutil

def label_protocol(input_path, output_path, col_idx):
    """
    Read lines from `input_path` and append a label according to column `col_idx`.
    - If the column value is 'original' or empty => label 0
    - Otherwise => label 1
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    processed = []
    for parts in lines:
        # If column index is out of range, treat as empty
        if col_idx >= len(parts):
            label = 0
        else:
            value = parts[col_idx].strip()
            print(value)
            if value == "" or value.lower() == "original":
                label = 0
            else:
                label = 1
        # Append the new label at the end of the line
        parts.append(str(label))
        processed.append(" ".join(parts))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(processed))

def show_ratio(LA, label_col='label'):
    total = len(LA)
    bonafide = (LA[label_col] == 'bonafide').sum()
    spoof = total - bonafide
    ratio = f"{bonafide}:{spoof} ≈ {bonafide/total:.2f}:{spoof/total:.2f}"
    return bonafide, spoof, total, ratio

def generate_protocols(
    input_path,
    output_dir,
    label_col=1,
    uttid_col=0,
    handcrafted_methods=None,
    dnn_methods=None,
    seed=42,
    build_complement=True,                  # Generate complementary splits
    persist_order_path=None,                # Save/reuse shuffled order (npz)
    full_proto_name="protocol_wm_100.txt"   # 100% filename
):
    import pandas as pd
    import numpy as np
    import os

    if handcrafted_methods is None:
        # phase 1
        # handcrafted_methods = ['svd', 'phase', 'lsb', 'fsvc', 'patch', 'lwt']

        # phase 2
        handcrafted_methods = ['norm', 'dsss', 'dctb1', 'echo']
    if dnn_methods is None:
        dnn_methods = ['silent', 'dnn']

    os.makedirs(output_dir, exist_ok=True)

    # Read the original protocol and standardize columns
    LA = pd.read_csv(input_path, sep=None, header=None, engine="python")
    LA.columns = [f"col_{i}" for i in range(LA.shape[1])]
    LA = LA.rename(columns={f"col_{uttid_col}": "utt_id", f"col_{label_col}": "label"})
    LA = LA.reset_index(drop=True)
    all_idx = LA.index.to_numpy()

    # Shuffle once (separately for bonafide / spoof)
    rng = np.random.default_rng(seed)
    bon_idx = LA.index[LA["label"] == "bonafide"].to_numpy()
    spf_idx = LA.index[LA["label"] == "spoof"].to_numpy()

    if persist_order_path and os.path.exists(persist_order_path):
        z = np.load(persist_order_path, allow_pickle=True)
        bon_idx, spf_idx = z["bon_idx"], z["spf_idx"]
        print(f"[ORDER] Loaded fixed order: {persist_order_path}")
    else:
        rng.shuffle(bon_idx)
        rng.shuffle(spf_idx)
        if persist_order_path:
            np.savez(persist_order_path, bon_idx=bon_idx, spf_idx=spf_idx)
            print(f"[ORDER] Saved fixed order: {persist_order_path}")

    # Assign methods over a given subset of indices:
    # split into handcrafted/DNN halves and distribute methods evenly within each half
    def assign_methods_on_subset(idx_set):
        sel = LA.loc[list(idx_set)].copy()
        out = pd.Series(index=sel.index, dtype=object)

        bon = sel[sel["label"] == "bonafide"]
        spf = sel[sel["label"] == "spoof"]

        # Split by half: Hand / DNN
        bon_hand = bon.iloc[: len(bon) // 2]
        bon_dnn  = bon.iloc[len(bon) // 2 :]
        spf_hand = spf.iloc[: len(spf) // 2]
        spf_dnn  = spf.iloc[len(spf) // 2 :]

        def chunks(df, k):
            if len(df) == 0:
                return [df]*k
            return np.array_split(df, k)

        # Handcrafted distribution
        for m, cb, cs in zip(handcrafted_methods,
                             chunks(bon_hand, len(handcrafted_methods)),
                             chunks(spf_hand, len(handcrafted_methods))):
            out.loc[cb.index] = m
            out.loc[cs.index] = m

        # DNN distribution
        for m, cb, cs in zip(dnn_methods,
                             chunks(bon_dnn, len(dnn_methods)),
                             chunks(spf_dnn, len(dnn_methods))):
            out.loc[cb.index] = m
            out.loc[cs.index] = m

        return out  # index=subset indices, value=method

    # Baseline: 100% mapping (every sample has a method)
    full_methods = assign_methods_on_subset(set(all_idx))
    full_df = LA[["utt_id", "label"]].copy()
    full_df["method"] = full_methods.reindex(LA.index).values
    full_path = os.path.join(output_dir, full_proto_name)
    full_df.to_csv(full_path, sep=" ", index=False, header=False)
    print(f"[SAVE] 100% protocol saved: {full_path} (all samples are non-original)")

    # Utility: return indices to be watermarked under ratio r (keep class balance; take prefix from fixed order)
    def wm_indices_by_ratio(r):
        n_b = int(len(bon_idx) * r)
        n_s = int(len(spf_idx) * r)
        return set(bon_idx[:n_b]).union(set(spf_idx[:n_s]))

    # Materialize: derive a main or complementary protocol from the 100% mapping
    def materialize_from_full(ratio, filename, complement=False, save_only_tagged=False):
        wm_ids = wm_indices_by_ratio(ratio)
        if complement:
            wm_ids = set(all_idx) - wm_ids

        method_series = pd.Series("original", index=LA.index, dtype=object)
        # Copy methods from full_methods for selected ids to ensure consistency
        method_series.loc[list(wm_ids)] = full_methods.loc[list(wm_ids)].values

        out_df = LA[["utt_id", "label"]].copy()
        out_df["method"] = method_series.values

        out_path = os.path.join(output_dir, filename)
        out_df.to_csv(out_path, sep=" ", index=False, header=False)
        print(f"[SAVE] {out_path}")

        # Short report on the watermarked subset
        tagged = out_df[out_df["method"] != "original"]
        total = len(tagged)
        if total > 0:
            b = (tagged["label"] == "bonafide").sum()
            s = total - b
            print(f"  Watermarked subset: {total} items, bonafide:spoof = {b}:{s} ≈ {b/total:.2f}:{s/total:.2f}")

        if save_only_tagged:
            only_path = out_path.replace(".txt", "_only.txt")
            tagged[["utt_id", "label", "method"]].to_csv(only_path, sep=" ", index=False, header=False)
            print(f"  Saved watermarked-only file: {only_path}")

        return out_df

    # Generate 25/50/75 main protocols (derived from the 100% fixed mapping)
    p25 = materialize_from_full(0.25, "protocol_wm_25.txt")
    p50 = materialize_from_full(0.50, "protocol_wm_50.txt")
    p75 = materialize_from_full(0.75, "protocol_wm_75.txt", save_only_tagged=True)

    # Self-check: nesting (using index sets)
    def wm_ids_from_df(df):
        return set(df.index[df["method"] != "original"].to_list())

    assert wm_ids_from_df(p25).issubset(wm_ids_from_df(p50)) and \
           wm_ids_from_df(p50).issubset(wm_ids_from_df(p75)), "[ERROR] 25⊆50⊆75 does not hold!"

    # Generate complementary protocols (complements of the 100% mapping)
    if build_complement:
        c25 = materialize_from_full(0.25, "protocol_wm_25_complement.txt", complement=True)
        c50 = materialize_from_full(0.50, "protocol_wm_50_complement.txt", complement=True)
        c75 = materialize_from_full(0.75, "protocol_wm_75_complement.txt", complement=True)

        # Self-check: complement sets are disjoint and united they cover all; reverse nesting for complements
        all_set = set(all_idx)
        for (m, c, tag) in [(p25, c25, "25%"), (p50, c50, "50%"), (p75, c75, "75%")]:
            wm_m = wm_ids_from_df(m); wm_c = wm_ids_from_df(c)
            assert wm_m.isdisjoint(wm_c), f"[ERROR] {tag} main/complement have intersection!"
            assert (wm_m | wm_c) == all_set, f"[ERROR] {tag} main/complement union does not cover all!"

        wm25c = wm_ids_from_df(c25); wm50c = wm_ids_from_df(c50); wm75c = wm_ids_from_df(c75)
        assert wm75c.issubset(wm50c) and wm50c.issubset(wm25c), "[ERROR] Complement nesting 75c⊆50c⊆25c does not hold!"

    print("Derived 75/50/25 + complement from 100% fixed mapping: consistency and nesting verified.")


def analyze_watermark_protocol(file_paths, handcrafted=None, dnn=None):
    """
    Analyze one or more protocol_wm_*.txt files:
      - For each watermark method: bonafide/spoof counts, ratios, and totals
      - Each method's overall proportion
      - Overall hand:DNN:original proportions

    Args:
        file_paths: str or List[str]
        handcrafted: List[str], handcrafted watermark method names
        dnn: List[str], DNN watermark method names

    Returns:
        dict: analysis results for each file
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    if handcrafted is None:
        # handcrafted = ['svd', 'phase', 'lsb', 'fsvc', 'patch', 'lwt']
        handcrafted = ['norm', 'dsss', 'dctb1', 'echo']
    if dnn is None:
        # dnn = ['wavmark', 'audioseal', 'timbre']
        dnn = ['silent', 'dnn']

    result = {}

    for path in file_paths:
        print(f"\n[ANALYZE] file: {path}")
        LA = pd.read_csv(path, sep=' ', header=None, names=["utt_id", "label", "method"])
        total_all = len(LA)

        stats = {}
        for method in sorted(LA['method'].unique()):
            subset = LA[LA['method'] == method]
            total = len(subset)
            n_bonafide = (subset['label'] == 'bonafide').sum()
            n_spoof = total - n_bonafide
            bonafide_ratio = n_bonafide / total if total > 0 else 0
            spoof_ratio = n_spoof / total if total > 0 else 0
            method_ratio = total / total_all if total_all > 0 else 0
            stats[method] = {
                'total': total,
                'bonafide': n_bonafide,
                'spoof': n_spoof,
                'ratio': f"{bonafide_ratio:.2f}:{spoof_ratio:.2f}",
                'overall_pct': f"{method_ratio:.2%}"
            }
            print(f"  {method:12s} | total: {total:5d} | bonafide: {n_bonafide:5d} | spoof: {n_spoof:5d} | ratio: {bonafide_ratio:.2f}:{spoof_ratio:.2f} | overall: {method_ratio:.2%}")

        # Overall category stats
        count_hand = sum(stats[m]["total"] for m in handcrafted if m in stats)
        count_dnn = sum(stats[m]["total"] for m in dnn if m in stats)
        count_original = stats["original"]["total"] if "original" in stats else total_all - count_hand - count_dnn

        ratio_hand = count_hand / total_all if total_all > 0 else 0
        ratio_dnn = count_dnn / total_all if total_all > 0 else 0
        ratio_original = count_original / total_all if total_all > 0 else 0

        print(f"\n[SUMMARY] overall proportions (total {total_all}):")
        print(f"  Handcrafted : {count_hand} ({ratio_hand:.2%})")
        print(f"  DNN         : {count_dnn} ({ratio_dnn:.2%})")
        print(f"  Original    : {count_original} ({ratio_original:.2%})")
        print(f"  Ratio       : {ratio_hand:.2f}:{ratio_dnn:.2f}:{ratio_original:.2f}")

        result[path] = stats

    return result


def splitfiles(protocol_path, audio_src_dir, output_dir,
               dnn_methods=None, suffix=".flac",
               save_protocol=True, protocol_filename_fmt="protocol_{method}.txt",
               save_merged_protocol=True, merged_protocol_name="protocol_subset.txt"):
    """
    Split DNN-watermarked files by method into subfolders, and optionally create
    per-method protocol files.

    Args:
        protocol_path (str): original protocol path, three columns: utt_id label method
        audio_src_dir (str): root directory of the original audio
        output_dir (str): output root directory (creates subdirectories {output_dir}/{method}/)
        dnn_methods (list[str]): DNN methods to keep; default ['wavmark','audioseal','timbre']
        suffix (str): audio suffix (e.g., ".flac" / ".wav")
        save_protocol (bool): whether to save a protocol for each method subfolder
        protocol_filename_fmt (str): filename format for each method protocol (use {method} placeholder)
        save_merged_protocol (bool): whether to save the merged subset protocol (all selected rows)
        merged_protocol_name (str): filename for the merged protocol
    """
    if dnn_methods is None:
        dnn_methods = ['wavmark', 'audioseal', 'timbre']

    os.makedirs(output_dir, exist_ok=True)

    # Read the original protocol (strictly 3 columns: utt_id, label, method)
    LA = pd.read_csv(protocol_path, sep=r"\s+", header=None, names=["utt_id", "label", "method"])

    # Keep only DNN rows
    dnn_LA = LA[LA["method"].isin(dnn_methods)].copy()
    total = len(dnn_LA)
    print(f"[INFO] Found {total} DNN-watermarked samples (methods: {dnn_methods})")

    # Optionally save the merged subset protocol
    if save_merged_protocol and total > 0:
        merged_path = os.path.join(output_dir, merged_protocol_name)
        dnn_LA.to_csv(merged_path, sep=" ", header=False, index=False)
        print(f"[SAVE] Merged subset protocol: {merged_path} ({total} rows)")

    for method in dnn_methods:
        method_LA = dnn_LA[dnn_LA["method"] == method].copy()
        if method_LA.empty:
            print(f"  [WARN] {method}: 0 rows, skipped.")
            continue

        # Subdirectory for the target method
        target_dir = os.path.join(output_dir, method)
        os.makedirs(target_dir, exist_ok=True)
        print(f"  [COPY] {method}: {len(method_LA)} samples -> {target_dir}")

        # Copy audio files
        missing = 0
        for utt_id in method_LA["utt_id"]:
            src_path = os.path.join(audio_src_dir, f"{utt_id}{suffix}")
            tgt_path = os.path.join(target_dir,   f"{utt_id}{suffix}")
            if os.path.exists(src_path):
                shutil.copy(src_path, tgt_path)
            else:
                missing += 1
                print(f"    [WARN] audio file not found: {src_path}")

        if missing:
            print(f"    [INFO] {method} has {missing} missing audio files.")

        # Save the protocol for this method (preserve original 3-column format)
        if save_protocol:
            proto_name = protocol_filename_fmt.format(method=method)
            proto_path = os.path.join(target_dir, proto_name)
            method_LA.to_csv(proto_path, sep=" ", header=False, index=False)
            print(f"    [SAVE] protocol for {method}: {proto_path} ({len(method_LA)} rows)")

# Merge scores

def load_score_file(score_path):
    LA = pd.read_csv(score_path, sep=None, header=None, engine='python')
    LA.columns = ['utt_id', 'score']
    return LA.set_index('utt_id')

import os

def merge_scores(protocol_path, score_wm_path, score_ori_path, output_path):
    def normalize_utt_id(utt):
        return os.path.splitext(str(utt))[0]  # strip suffix (e.g., .flac)

    # Read protocol
    LA = pd.read_csv(protocol_path, sep=" ", header=None, names=["utt_id", "label", "method"])
    LA["utt_id"] = LA["utt_id"].apply(normalize_utt_id)

    # Load score files and normalize ids
    score_wm = load_score_file(score_wm_path)
    score_wm.index = score_wm.index.map(normalize_utt_id)

    score_ori = load_score_file(score_ori_path)
    score_ori.index = score_ori.index.map(normalize_utt_id)

    merged_scores = []

    for _, row in LA.iterrows():
        utt_id = row["utt_id"]
        method = row["method"]

        if method == "original":
            score = score_ori.loc[utt_id, "score"] if utt_id in score_ori.index else None
        else:
            score = score_wm.loc[utt_id, "score"] if utt_id in score_wm.index else None

        if score is not None:
            merged_scores.append((utt_id, score))
        else:
            print(f"[WARN] score not found: {utt_id}")

    # Output
    out_LA = pd.DataFrame(merged_scores, columns=["utt_id", "score"])
    out_LA.to_csv(output_path, sep=" ", index=False, header=False)
    print(f"[SAVE] merged scores -> {output_path}, total {len(out_LA)}")

# from your_module import merge_scores  # ensure merge_scores is imported if used elsewhere

def auto_merge_scores(
    track: str,
    score_wm_path: str,
    score_ori_path: str,
    *,
    protocols_root: str = "/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/protocols/tagged",
    percents: Iterable[int] = (75, 50, 25),
    out_subdir: str = "percentage",
) -> List[str]:
    """
    Automatically call `merge_scores` and generate files named like:
    watermark_timit_SLS_75_ITW.txt.
    """
    track = track.strip()
    allowed = {"LA21", "DF21", "ITW"}
    if track not in allowed:
        raise ValueError(f"`track` must be one of {allowed}, but got: {track}")

    wm_path = Path(score_wm_path).expanduser().resolve()
    ori_path = Path(score_ori_path).expanduser().resolve()
    if not wm_path.exists():
        raise FileNotFoundError(f"score_wm_path does not exist: {wm_path}")
    if not ori_path.exists():
        raise FileNotFoundError(f"score_ori_path does not exist: {ori_path}")

    # Output directory: sibling "percentage" folder next to the wm score file
    out_dir = wm_path.parent / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    ori_copy_path = out_dir / ori_path.name
    shutil.copy(ori_path, ori_copy_path)
    print(f"[COPY] Original file copied to {ori_copy_path}")

    # Protocol directory: {protocols_root}/{track}
    proto_dir = Path(protocols_root).expanduser().resolve() / track

    # Base name without extension
    base_name = wm_path.stem  # e.g., "watermark_timit_SLS_ITW"
    suffix = wm_path.suffix if wm_path.suffix else ".txt"

    outputs: List[str] = []
    for p in percents:
        proto = proto_dir / f"protocol_wm_{int(p)}.txt"

        # Output naming rule:
        # original: watermark_timit_SLS_ITW.txt
        # now:     watermark_timit_SLS_75_ITW.txt
        # logic: drop trailing "_{track}", insert percentage before track
        if base_name.endswith(f"_{track}"):
            core = base_name[: -len(track) - 1]  # drop "_ITW"
        else:
            core = base_name
        out_file = out_dir / f"{core}_{int(p)}_{track}{suffix}"

        merge_scores(
            protocol_path=str(proto),
            score_wm_path=str(wm_path),
            score_ori_path=str(ori_path),
            output_path=str(out_file),
        )
        if track == "LA21":
            LAeval(out_dir)
        elif track == "ITW":
            ITWeval(out_dir)
        elif track == "DF21":
            DFeval(out_dir)
        print(f"[OK] {track} {p}% -> {out_file}")
        outputs.append(str(out_file))
    outputs.insert(0, str(ori_copy_path))

    return outputs

def process_txt(input_path, output_path, col_idx):
    """
    Read `input_path` line by line and append a label based on column `col_idx`.
    - If that column is 'original' or empty => label 0
    - Otherwise => label 1
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    processed = []
    for parts in lines:
        # If column index is out of range, treat it as empty
        if col_idx >= len(parts):
            label = 0
        else:
            value = parts[col_idx].strip()
            print(value)
            if value == "" or value.lower() == "original":
                label = 0
            else:
                label = 1
        # Append the new label at the end of the line
        parts.append(str(label))
        processed.append(" ".join(parts))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(processed))

# === Example usage ===
if __name__ == "__main__":

    # generate_protocols(
    #     input_path="/public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/DF-keys-full/keys/DF/CM/trial_metadata.txt",  # original protocol
    #     output_dir="/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/split",       # output directory
    #     uttid_col=1,
    #     label_col=5,
    #     seed=42,
    #     build_complement=True,
    #     persist_order_path="/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/_order.npz",
    #     full_proto_name="protocol_wm_100.txt"
    # )

    # Check if the splits were generated correctly
    # analyze_watermark_protocol([
    #     "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/ITW/split/protocol_wm_75.txt",
    #     "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/ITW/split/protocol_wm_50.txt",
    #     "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/ITW/split/protocol_wm_25.txt",
    #     "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/ITW/split/protocol_wm_100.txt"
    # ])

    # Check complementary files
    # analyze_watermark_protocol([
    #     "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/split/protocol_wm_75_complement.txt",
    #     "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/split/protocol_wm_50_complement.txt",
    #     "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/split/protocol_wm_25_complement.txt",
    #     "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/split/protocol_wm_100_complement.txt"
    # ])

    # Split out DNN subsets
    # splitfiles(
    #     protocol_path="/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/DF21/split/protocol_wm_75.txt",
    #     audio_src_dir="/public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/ASVspoof2021_DF_eval/flac",  # original audio root
    #     output_dir="/public/home/qinxy/AudioData/Antispoofing/DF_DNN",                                     # output root
    #     dnn_methods=["dnn"],
    #     suffix=".flac",                                                                                    # your audio suffix
    #     save_protocol=True,                                                                                # write protocol_{method}.txt in each subdir
    #     save_merged_protocol=True
    # )

    # TRACK = "ITW"
    # comment = "ori"
    # outs = auto_merge_scores(
    #     track=TRACK,
    #     score_wm_path="/public/.../test2/{}_train_eval_wm_SLS_{}.txt".format(comment, TRACK),
    #     score_ori_path="/public/.../test2/{}_train_eval_ori_SLS_{}.txt".format(comment, TRACK),
    # )

    # print("Output Results:")
    # for p in outs:
    #     print(" -", p)

    # DFeval("/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test/train_eval")
    # LAeval("/public/home/qinxy/zhangzs/SSL_Anti-spoofing-main/test/baseline_eval")
    # ITWeval("/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test")
    
    #tagging if watermarked. Generating train protocols
    # input= "/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged/LA19/50A_LA19.txt"
    # output="/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/trainData/asvspoof2019_50_watermarked_train.txt"
    # col= 2
    # process_txt(input, output, col)