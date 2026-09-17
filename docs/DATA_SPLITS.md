# Deduplication and data splits

SonoCorpus combines 53 public ultrasound datasets. Some of them contain pixel-identical images stored
under different names. If two copies of the same image landed in different splits, the test set would
be partly in-sample. The split lists used for SonoBase are therefore built in two steps: duplicates are
found first, and the deduplicated sample pool is split afterwards. Training and evaluation read images
from the intact converted datasets and take the file lists from a separate split directory.

| Step | Tool |
|---|---|
| Duplicate scan | `src/data/utils/check_overlap.py` (Slurm wrapper: `src/scripts/data/dedup_overlap.sbatch`) |
| Split generation | `src/data/utils/generate_splits.py` |
| Shared helpers | `src/data/utils/dataset_split.py` |

`DATASET_DIR` points at the converted datasets and `ANNOTATION_DIR` at the split directory (see the
repository README).

## 1. Find duplicates

```bash
cd src
python -m data.utils.check_overlap \
  --path "$DATASET_DIR" --hash-algo md5 --workers 32 \
  --output overlap_report.json
```

`--hash-algo md5` hashes decoded pixels and finds exact duplicates; this is the setting used for the
paper. `file_md5` hashes file bytes instead, and `phash` finds perceptual near-duplicates. The report
lists every duplicate group within and across datasets. For the copy of SonoCorpus used in the paper,
all duplicates were within single datasets.

## 2. How duplicates are handled

| Type | Example | Handling |
|---|---|---|
| Duplicate images in an image dataset | two files with identical pixels | all but one member of each group are dropped from the pool before splitting |
| Duplicate videos | two video IDs sharing identical frames, such as CAMUS `patient0033` and `patient0068` | videos sharing any identical frame are clustered and one representative is kept |
| Repeated frames within one video | adjacent identical frames | kept; they always stay in the same split |

Because the pool is deduplicated before it is split, no duplicate can end up in two splits, whatever the
random seed.

## 3. Tiers

Tier membership is derived from the two data configs, so there is no separate list to keep in sync.

| Tier | Definition | Datasets |
|---|---|---|
| Pretrain | only in `src/configs/pretrain/data/38_pt_8_bm.yaml` | 38 |
| Benchmark | in both configs | 8 |
| External | only in `src/configs/pretrain/data/8_bm_7_ext.yaml` | 7 |

## 4. Split generation

| Tier | train | val | test |
|---|---|---|---|
| Pretrain | 0.95 | 0.05 | none |
| Benchmark | 0.70 | 0.10 | 0.20 |
| External | 0.10, used as the few-shot adaptation pool | none | 0.90 |

The seed is 42. Image datasets are split by image and video datasets by video. Two datasets use their
own splitters, registered in `SPLIT_OVERRIDES`:

- **CAMUS** is split 0.70 / 0.10 / 0.20 by patient, so both views of a patient stay in one split.
- **RegPro** keeps its official validation cases as the test set and splits its official training cases
  0.9 / 0.1 into train and val.

```bash
python -m data.utils.generate_splits \
  --saus-root "$DATASET_DIR" \
  --overlap-report overlap_report.json \
  --output-dir "$ANNOTATION_DIR"
```

Always pass `--overlap-report`; without it the pool is not deduplicated. `--dry-run` runs without writing
files, and `--pretrain-config` / `--eval-config` select other tier definitions.

## 5. How training and evaluation use the lists

Every data config takes images and annotations from `${DATASET_DIR}/<dataset>/` and the sample lists
from `${ANNOTATION_DIR}/<dataset>/{train,val,test}_list.txt`, both through environment variables. Using a
different split set needs no config change.

A split set is clean when:

1. no list contains a sample twice;
2. no sample appears in more than one of train, val and test;
3. for every duplicate group in the overlap report, all kept members sit in one split, and each cluster of
   duplicate videos keeps at most one video;
4. every listed sample exists under `DATASET_DIR`.

`generate_splits.py` satisfies the first three conditions by construction.

## 6. Using the split lists from the Zenodo manifest

The SonoCorpus manifest (https://doi.org/10.5281/zenodo.22770825) stores the lists as
`splits/<dataset>_<split>.txt` and the few-shot pools as
`splits/fewshot/<dataset>_fewshot_k<k>_seed<seed>.txt`. Dataset names contain no underscores, so the
layout the code expects can be rebuilt from the unpacked record:

```bash
# split lists -> $ANNOTATION_DIR/<dataset>/<split>_list.txt
for f in splits/*.txt; do
  b=$(basename "$f" .txt); ds=${b%%_*}; name=${b#*_}
  [ "$name" = README_PSEUDONYMISED ] && continue
  mkdir -p "$ANNOTATION_DIR/$ds" && cp "$f" "$ANNOTATION_DIR/$ds/${name}_list.txt"
done

# few-shot pools -> src/experiments/few_shot/few_shot_splits/<dataset>_N<k>_seed<seed>.txt
FS=/path/to/sonobase/src/experiments/few_shot/few_shot_splits
mkdir -p "$FS"
for f in splits/fewshot/*.txt; do
  b=$(basename "$f" .txt); ds=${b%%_fewshot_*}; rest=${b#*_fewshot_k}
  cp "$f" "$FS/${ds}_N${rest%%_seed*}_seed${rest#*_seed}.txt"
done
```

CDNet is the one exception. Its source file names contain person-name-like identifiers, so its lists hold
`sha256(<file stem>)[:16]` instead of the stems. Hash the stems of your own converted CDNet files the
same way to match them.
