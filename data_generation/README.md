Here’s a cleaned-up, consistent, and polished version:

---

## Watermark-Spoofing Dataset (WSD)

We derive **WSD** from public corpora and benchmark it against the original datasets:

* **ASVspoof 2019 (LA)** — training split used for model training.
  Download: [https://datashare.is.ed.ac.uk/handle/10283/3336](https://datashare.is.ed.ac.uk/handle/10283/3336)
* **ASVspoof 2021 (LA & DF)** — used for evaluation.
  LA: [https://zenodo.org/record/4837263](https://zenodo.org/record/4837263)
  DF: [https://zenodo.org/record/4835108](https://zenodo.org/record/4835108)
* **In-the-Wild** — used for evaluation.
  Download: [https://deepfake-total.com/in\_the\_wild](https://deepfake-total.com/in_the_wild)

**Typical setup:** train on **ASVspoof 2019 LA (train)**; evaluate on **ASVspoof 2021 LA/DF (eval)** and **In-the-Wild**.

The data-generation utilities live in `data_generation/`. See that folder’s README to reproduce WSD and its variants (seen/unseen watermark splits).

---

## Generating Protocols & Utilities

This script provides a small toolkit to:

1. build watermarked protocol files at various ratios,
2. inspect method/class distributions,
3. (optional) split DNN-watermarked audio into subfolders.

It targets ASVspoof 2019/2021 and In-the-Wild experiments.

### Prerequisites

* Python 3.7+
* `pandas`, `numpy`
* Your evaluation functions

### File Concepts

* **Original protocol:** your dataset metadata (column layout depends on source). You specify which column is `utt_id` and which column is the class `label` (`bonafide` or `spoof`).
* **Watermarked protocol:** derived files with a third column `method`

  * `original` → keep original audio
  * other values → watermarked by the named method (e.g., `norm`, `dsss`, `dctb1`, `echo`, `silent`, `dnn`)

The script produces:

* `protocol_wm_100.txt` — every item is assigned a non-`original` watermark method
* `protocol_wm_{75|50|25}.txt` — only a subset is watermarked; others are `original`
* `protocol_wm_{25|50|75}_complement.txt` — complementary sets
* `protocol_wm_75_only.txt` — only the watermarked rows (convenience view)

---

## Typical Workflow

### 1) Generate watermarked protocols

```python
from your_script import generate_protocols

generate_protocols(
    input_path="/path/to/original/protocol.txt",   # raw protocol/metadata
    output_dir="/path/to/out/protocols",           # where wm protocols are written
    uttid_col=1,                                   # column index for utt_id
    label_col=5,                                   # column index for 'bonafide'/'spoof'
    seed=42,
    build_complement=True,
    persist_order_path="/path/to/out/_order.npz",  # save shuffling order for reproducibility
    full_proto_name="protocol_wm_100.txt"
)
```

**What it does:**

* normalizes to `utt_id` and `label`,
* shuffles class-balanced indices once (optionally persisted),
* assigns watermark methods (half handcrafted, half DNN by default),
* materializes 100%, 75%, 50%, 25% (and complements if requested).

### 2) Inspect distributions (optional)

```python
from your_script import analyze_watermark_protocol

analyze_watermark_protocol([
    "/path/to/out/protocols/protocol_wm_75.txt",
    "/path/to/out/protocols/protocol_wm_50.txt",
    "/path/to/out/protocols/protocol_wm_25.txt",
    "/path/to/out/protocols/protocol_wm_100.txt",
])
```

You’ll see per-method counts and overall **Handcrafted : DNN : Original** proportions.

### 3) Label which items are watermarked

```python
# Example: mark a file where column 2 indicates the watermark method
from your_script import process_txt

input_path  = "/public/home/qinxy/.../LA19/50A_LA19.txt"  # a file listing 50% watermarked items
output_path = "/public/home/qinxy/.../database/trainData/asvspoof2019_50_watermarked_train.txt"
col_index   = 2

process_txt(input_path, output_path, col_index)
```

### 4) (Optional) Split DNN-watermarked audio by method

```python
from your_script import splitfiles

splitfiles(
    protocol_path="/path/to/out/protocols/protocol_wm_75.txt",
    audio_src_dir="/path/to/audio/root",           # where {utt_id}.flac lives
    output_dir="/path/to/out/DF_DNN",              # creates subfolders per method
    dnn_methods=["dnn"],                           # or ["wavmark","audioseal","timbre"]
    suffix=".flac",
    save_protocol=True,
    save_merged_protocol=True
)
```

This copies matching audio into subfolders and writes per-method protocol files.

---

## Function Reference

* `label_protocol(input_path, output_path, col_idx)`
  Appends a numeric label to each line based on `col_idx` (`original`/empty → 0; otherwise → 1).

* `generate_protocols(...)`
  Builds 100/75/50/25 protocols (and complements) with consistent method assignment.

* `analyze_watermark_protocol(file_paths, handcrafted=None, dnn=None)`
  Prints per-method/class counts and overall proportions.

* `splitfiles(protocol_path, audio_src_dir, output_dir, ...)`
  Copies DNN-watermarked audio into per-method subfolders and writes per-method protocols.

* `process_txt(input_path, output_path, col_idx)`
  Tags each row with `1` if watermarked (column indicates method), else `0`.

---

## Notes & Troubleshooting

* Ensure score files and protocol `utt_id`s match (the merge step strips file extensions).
* Keep protocol ratios class-balanced by reusing the persisted shuffle order.
* Large I/O? Place protocols and scores on fast local storage to avoid bottlenecks.

---

## Installation for Watermarking Scripts

* **Handcrafted watermarking:** see the **TAF** folder.
* **DNN watermarking:** it’s recommended to implement your own script.

Create an environment:

```bash
conda create -n watermark python=3.10.13 -y
conda activate watermark
pip install -r requirements.txt
# or:
# conda env create -f environment.yml
```

### Example: full-attack mode (research evaluation)

```bash
python main.py \
  --mode all_attacks \
  --threads 32 \
  --new_protocol_path /public/home/qinxy/.../protocols/tagged/ITW/protocol_wm_75.txt \
  --audio_dir /public/home/qinxy/AudioData/Antispoofing/release_in_the_wild \
  --embedded_path /public/home/qinxy/AudioData/Antispoofing/The-A-Files-master/test1 \
  --output Test.csv \
  --limit -1
```

### Example: embed-only mode (no attacks; follow a protocol)

```bash
python main.py \
  --mode embed_only \
  --threads 32 \
  --new_protocol_path /public/home/qinxy/.../DF21/ori_split/protocol_part_8.txt \
  --audio_dir /public/home/qinxy/AudioData/Antispoofing/ASVspoof2021/ASVspoof2021_DF_eval/flac/ \
  --embedded_path /public/home/qinxy/AudioData/Antispoofing/DF21_phase2 \
  --save_embedded \
  --output watermarked_file \
  --limit -1
```

> Writing your own watermarking script is encouraged if you have custom embedding or attack pipelines.


Acknowledgments & Third-Party Code

This repository builds upon and incorporates modified components from the following open-source projects:

SLSforASVspoof-2021-DF — https://github.com/QiShanZhang/SLSforASVspoof-2021-DF.git

The-A-Files — https://github.com/pawel-kaczmarek/The-A-Files.git

Portions of the code are adapted from these repositories. All third-party code remains under its original licenses; please refer to the upstream LICENSE files and retain all applicable copyright notices. If any attribution or licensing detail is incomplete, let us know and we will correct it promptly.