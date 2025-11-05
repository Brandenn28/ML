# Cross-Domain Plant ID — Pre-Processing Summary (No Code)

This note explains **what I did in data pre-processing**, **what files were produced**, and **why** such that we can jump straight to training.

---

## 1) Objective

Prepare a **clean, reproducible, leak-free** dataset with:

* Stable label indices for the model.
* Clear domain tags (herbarium vs field/photo).
* Robust splits (train/validation) that respect tiny classes and the field domain.
* Sanity checks for broken/corrupt images.
* Practical stats and weights to handle class imbalance and guide augmentations.

---

## 2) Inputs we consumed (from `list/`)

* `species_list.txt` — master species table: `class_id` + scientific name.
* `train.txt` — training image manifest (paths + class IDs).
* `test.txt` — test image manifest (paths, no labels).
* `groundtruth.txt` — **true test labels** (used **only** for evaluation after training).
* `class_with_pairs.txt` / `class_without_pairs.txt` — species IDs that do / don’t have herbarium–field pairs (analysis only).

> Note: Some field photos are under a `photo/` folder. We treat **`photo` as `field`**.

---

## 3) Artifacts we produced (in `processed/`)

**Core (training depends on these):**

* `species_mapping.csv`

  * Columns: `class_id, species_name, label_idx`
  * A frozen, contiguous mapping `class_id → label_idx (0..N-1)` used everywhere.
* `train.csv` / `val.csv` / `test.csv`

  * Columns: `rel_path, abs_path, class_id, label_idx, species_name, domain`
  * `domain ∈ {herbarium, field}` (with `photo → field`).
  * `test.csv` contains no labels.
* `test_with_groundtruth.csv`

  * Same columns as above **with** labels for test.
  * **For evaluation only** (never used during training).

**Quality & analysis:**


* `class_counts.csv` — per-class counts in `train`.
* `class_weights.csv` — inverse-frequency weights per class.
* `train_sample_weights.csv` — per-sample weights for `WeightedRandomSampler`.
* `image_sizes.csv` — width, height, aspect ratio per image (all splits).
* `per_class_domain_stats.csv` — per species: `n_herbarium, n_field, total, has_pair`.

### Each CSV file at a glance:

### `species_mapping.csv`

*  Canonical class dictionary: `class_id, species_name, label_idx (0…N-1)`.
*  Fixes a single, reproducible label order for the model head and metrics.
*  Load in all training/eval scripts to convert dataset `class_id` ↔ `label_idx`.



### `train.csv`

*  Original training split with per-image metadata (`abs_path, label_idx, domain`).
*  Clean, leak-free list of training samples with domains normalized (`photo → field`).
*  DataLoader for training if you’re not using the improved `train_v2.csv`.


### `val.csv`

*  Original validation split matching `train.csv`.
*  Honest hold-out to tune hyperparameters and monitor overfitting.
*  DataLoader for validation if you’re not using `val_v2.csv`.



### `train_v2.csv`

*  **Improved** training split (tiny-class protection + ensure ≥1 field sample in val when possible).
*  Prevents starving rare classes and makes validation reflective of the test domain.
*  **Preferred** training DataLoader split.



### `val_v2.csv`

*  **Improved** validation split paired with `train_v2.csv`.
*  Field-aware validation for more stable early stopping and selection.
*  **Preferred** validation DataLoader split.



### `test.csv`

*  Test image manifest **without labels** (paths + `domain=field`).
*  Defines the evaluation inputs while keeping labels hidden from training.
*  Inference over the test set before computing metrics.



### `test_with_groundtruth.csv`

*  Test manifest **with true labels** (`class_id/label_idx` joined).
*  Compute Top-1/Top-5 accuracy after training; no leakage into training.
*  **Evaluation only**—compare predictions vs. these labels.



### `class_counts.csv`

*  Per-class sample counts for the chosen training split, plus inverse-frequency numbers.
*  Quantifies imbalance; baseline for weighting strategies.
*  Reporting/EDA; derive weights for samplers or loss (can be read directly).



### `class_weights.csv`

*  Minimal table of `label_idx → weight_inv` (inverse frequency).
*  Plug-and-play weights for re-weighted losses (e.g., CrossEntropy).
*  Construct criterion with `weight=` tensor.



### `train_sample_weights.csv`

*  Per-sample weights aligned to training rows (`abs_path, label_idx, sample_weight`).
*  Enable **WeightedRandomSampler** to balance batches without altering loss.
*  Training DataLoader via `sampler=WeightedRandomSampler(...)`.



### `image_sizes.csv`

*  Width, height, aspect ratio (and any decode errors) for all images.
*  Choose sensible input size/crop policy; spot outliers/corruption.
*  EDA/report figures; justify transform choices; optionally filter extremes.



### `per_class_domain_stats.csv`

*  For each species: `n_herbarium`, `n_field`, `total`, and `has_pair` flag.
*  Shows domain coverage; identifies herbarium-only vs paired species.
*  Reporting/plots; guide domain-specific augmentation or sampling.



### `species_with_pairs.csv`

*  Human-readable list of class IDs **with** herbarium–field pairs (joined with names).
*  Quick reference for species that teach cross-domain correspondence.
*  EDA/report; optionally prioritize/stratify experiments.



### `species_without_pairs.csv`

*  Human-readable list of herbarium-only species (joined with names).
*  Highlights few-/zero-field classes where generalization is hardest.
*  EDA/report; consider extra augmentation/weighting for these classes.



### Notes

* Files with `abs_path` are environment-specific; if you change machines (e.g., Windows ↔ Colab), regenerate processed CSVs to refresh paths.
* Always rely on `species_mapping.csv` for label order; don’t rebuild label indices inside training.

---

## 4) What the pipeline did 

1. **Built a stable species mapping**

   * Parsed `species_list.txt` and assigned a **contiguous `label_idx`** (0..N-1).
   * Saved as `species_mapping.csv`—this index is the single source of truth for the classifier head and metrics.

2. **Parsed manifests & attached metadata**

   * Read `train.txt` and reconciled `(path, class_id)` pairs; merged with `species_mapping.csv`.
   * **Inferred domain** from path:

     * `herbarium → herbarium`, `field/photo → field`.
     * `test` defaults to `field`.
   * Resolved `abs_path` and flagged any missing files.

3. **Sanity check: image decode & EXIF**

   * Opened each image once, normalized EXIF orientation, and forced RGB.
   * Logged failures to `bad_images.csv` and **dropped** them from splits to avoid runtime crashes.

4. **Created a reproducible train/validation split**

   * Started from the full training pool.
   * **Stratified by `label_idx`** with a fixed seed.
   * **Tiny-class rule:** if a class has **<5 images**, we **don’t** hold any out for validation (prevents starving training).
   * **Domain tweak:** when a class has field images and we do hold out val, we **ensure at least one field image** lands in validation (keeps validation meaningful for the target domain).
   * Saved to `train.csv` and `val.csv`.

5. **Kept test labels separate**

   * Merged `groundtruth.txt` with the mapping to produce `test_with_groundtruth.csv`.
   * **Never** used in training—only for evaluation (Top-1 / Top-5).

6. **Prepared imbalance artifacts**

   * Counted images per class (from `train.csv`).
   * Wrote per-class inverse-frequency weights (`class_weights.csv`) and per-sample weights (`train_sample_weights.csv`) so you can use a weighted sampler or re-weighted loss immediately.

7. **Computed practical stats**

   * `image_sizes.csv` — min/mean/median/max width & height; aspect ratio distribution (helps justify resize policy and detect outliers).
   * `per_class_domain_stats.csv` — per species counts for herbarium vs field, and a `has_pair` flag (useful for reporting and domain adaptation choices).

---

## 5) Design choices 

* **Contiguous label indices** → ensures the classifier head and evaluation use the same order consistently.
* **Domain tagging** (`herbarium` vs `field`) → unlocks domain-specific augmentations in training.
* **Tiny-class protection** → prevents validation from consuming scarce examples.
* **Field-aware validation** → reflects the real test domain and stabilizes early stopping.
* **Decode/EXIF pass** → avoids training crashes and “mystery rotations.”
* **Imbalance weights** → quick path to fair sampling or weighted loss without recomputing.
* **Dimension stats** → evidence-based choice of input size/augmentations.
* **Separate test labels** → no leakage; honest Top-1/Top-5.

---

## 6) How to proceed to training 

* **Loaders:** read `train.csv` / `val.csv` (image from `abs_path`, label from `label_idx`).
* **Normalization:** use ImageNet mean/std for pretrained backbones.
* **Augmentations:**

  * *Herbarium:* mild (small rotation, light color jitter).
  * *Field:* stronger (random resized crop, flip, stronger color jitter, small rotation).
* **Imbalance:**

  * Use `train_sample_weights.csv` with `WeightedRandomSampler`, **or**
  * Use `class_weights.csv` in the loss (`CrossEntropyLoss(weight=...)`).
* **Evaluation:** run on `test.csv`, compare against `test_with_groundtruth.csv` to compute Top-1 / Top-5.


