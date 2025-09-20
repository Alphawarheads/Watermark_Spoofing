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


# === Example usage ===
if __name__ == "__main__":

    
    TRACK = "ITW"
    comment = "ori"
    outs = auto_merge_scores(
        track=TRACK,
        score_wm_path="/public/.../test2/{}_train_eval_wm_SLS_{}.txt".format(comment, TRACK),
        score_ori_path="/public/.../test2/{}_train_eval_ori_SLS_{}.txt".format(comment, TRACK),
    )

    print("Output Results:")
    for p in outs:
        print(" -", p)

    # DFeval("/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test/train_eval")
    # LAeval("/public/home/qinxy/zhangzs/SSL_Anti-spoofing-main/test/baseline_eval")
    # ITWeval("/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/test")
