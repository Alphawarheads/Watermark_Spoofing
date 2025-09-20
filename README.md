# THE IMPACT OF AUDIO WATERMARKING ON AUDIO ANTI-SPOOFING COUNTERMEASURES

This repository provides the official implementation for our paper: **[THE IMPACT OF AUDIO WATERMARKING ON AUDIO ANTI-SPOOFING COUNTERMEASURES](https://someonepaper.com)** .

## What’s inside

1. **Watermark-Spoofing Dataset (WSD)** — scripts and label files to generate our watermarked evaluation and training sets.
2. **Knowledge-Preserving Watermark Learning (KPWL)** — scripts to reproduce the KPWL framework and experiments.

---

## Watermark-Spoofing Dataset (WSD)
We derive the WSD from public corpora and benchmark it against the original datasets:

* **ASVspoof 2019 (LA)** — training split used for model training. Download: [https://datashare.is.ed.ac.uk/handle/10283/3336](https://datashare.is.ed.ac.uk/handle/10283/3336)
* **ASVspoof 2021 (LA & DF)** — used for evaluation.

  -LA: [https://zenodo.org/record/4837263](https://zenodo.org/record/4837263)
  -DF: [https://zenodo.org/record/4835108](https://zenodo.org/record/4835108)
* **In-the-Wild** — used for evaluation. Download: [https://deepfake-total.com/in\_the\_wild](https://deepfake-total.com/in_the_wild)

**Typical setup:** train on **ASVspoof 2019 LA (train)**; evaluate on **ASVspoof 2021 LA/DF (eval)** and **In-the-Wild**.

The data generation utilities live in `data_generation/`. Follow that folder’s README to reproduce WSD and its variants (seen/unseen watermark splits).

---

## Installation (KPWL)

```bash
# 1) Clone
git clone https://github.com/Alphawarheads/Watermark_Spoofing.git
cd Watermark_Spoofing

# 2) (Optional) Use the bundled Fairseq snapshot
unzip fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1.zip

# 3) Conda env
conda create -n kpwl python=3.7 -y
conda activate kpwl

# 4) PyTorch (adjust CUDA/ROCm wheels as needed)
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1

# 5) Fairseq (editable)
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
cd ..

# 6) Project requirements
pip install -r requirements.txt
```

> Notes: our main experiments were trained on **4 GPUs/DCUs**.

---

## Pre-trained wav2vec 2.0 XLS-R (300M)

Download XLS-R weights from: [https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

---

## Evaluation

The commands below produce three `score.txt` files (one per evaluation set). These scores are later used to compute **EER (%)**.

> **Tip:** Always use the full flags `--eval_track` and `--eval_output` (do not abbreviate to `--eval`).

### LA 2021

**Original (non-watermarked):**

```bash
TRACK=LA
PROTOCOLS=/public/home/qinxy/.../ASVspoof2021.LA.cm.eval.trl.txt
DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/ASVspoof2021_LA_eval/flac/
EVAL_OUTPUT=/public/home/qinxy/.../test2/${COMMENT}_eval_ori_LA21.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
    main_multinodes.py --track="$TRACK" --is_eval --eval \
  --model_path models/model_4dcu_kpwl.pth \
  --protocols_path="$PROTOCOLS" \
  --database_path="$DATABASE_PATH" \
  --eval_output="$EVAL_OUTPUT"
```

**Watermark-Spoof (Seen):**

```bash
TRACK=LA
PROTOCOLS=/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged/LA21/protocol_wm_75_only.txt
DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/LA21_phase1/
EVAL_OUTPUT=/public/home/qinxy/.../test2/${COMMENT}_eval_wm_seen_LA21.txt
```

**Watermark-Spoof (Unseen):**

```bash
TRACK=LA
PROTOCOLS=/public/home/qinxy/zhangzs/watermarking/The-A-Files-master/protocols/tagged_phase2/LA21/split/protocol_wm_75_only.txt
DATABASE_PATH=/public/home/qinxy/AudioData/Antispoofing/LA21_phase2/
EVAL_OUTPUT=/public/home/qinxy/.../test2/${COMMENT}_eval_wm_unseen_LA21.txt
```

> See `eval.sh` for the full set of evaluation scripts.

---
**Test with our pretrained models:**
- [Download Trained KPWL](https://drive.google.com/file/d/13y5iuSVkCtkUh5Udyk0lBSS7uBMIAgM7/view?usp=drive_link)
- [Download Trained Baseline](https://drive.google.com/file/d/1OXhM-9KYpEgZXGd05Pslpy9yaZq1izmu/view?usp=drive_link)

---

### Merge scores and evaluate

* `_wm_` in a filename = scores produced on watermarked audio.
* `_ori_` in a filename = scores produced on original (non-watermarked) audio.

```python autoEval
# If auto_merge_scores is not in the same file, import it:
# from your_module import auto_merge_scores

TRACK = "ITW"     # one of {"LA21", "DF21", "ITW"}
comment = "ori"   # label used in your score filenames

outs = auto_merge_scores(
    track=TRACK,
    score_wm_path=f"/public/.../test2/{comment}_train_eval_wm_SLS_{TRACK}.txt",
    score_ori_path=f"/public/.../test2/{comment}_train_eval_ori_SLS_{TRACK}.txt",
)

print("Output Results:")
for p in outs:
    print(" -", p)
```

See the README in generate_data/ for details on how the protocols are constructed.

---

## Training

### Baseline pretraining (original)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  main_multinodes.py \
  --track=DF --lr=0.000001 --batch_size=5 --loss=WCE \
  --num_epochs=50 --comment 4dcu_baseline
```

### KPWL fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
    main_kpwl.py \
  --eval_track=DF --lr=5e-7 --batch_size=5 --num_epochs=2 \
  --model_path best_model.pth \
  --comment model_4dcu_kpwl
```

---
Acknowledgments & Third-Party Code

This repository builds upon and incorporates modified components from the following open-source projects:

SLSforASVspoof-2021-DF — https://github.com/QiShanZhang/SLSforASVspoof-2021-DF.git

The-A-Files — https://github.com/pawel-kaczmarek/The-A-Files.git

Portions of the code are adapted from these repositories. All third-party code remains under its original licenses; please refer to the upstream LICENSE files and retain all applicable copyright notices. If any attribution or licensing detail is incomplete, let us know and we will correct it promptly.

**Contact / Citation**
If you use this repository or WSD/KPWL in your research, please cite the paper and link back to this repo.
