# Predicting lipophilicity (logD) from molecular structure

This primarily a machine-learning project on a chemistry dataset. A model reads a molecule's structure and predicts how lipophilic it is (its **logD**), is tested on **new chemical families** it has never seen, says **"I don't know"** when a molecule looks too unlike its training data, and has a **light explainability (XAI) summary** (`permutation_importance`, `PartialDependenceDisplay`) that an LLM puts into plain English.

I am an ML student, not a chemist. The README therefore also explains the chemistry vocabulary and tools (SMILES, ChEMBL, RDKit, fingerprints, scaffolds) as I understood them while building this.

**Attention!** This is not ful XAI, here there is No SHAP, LIME or counterfactuals; nothing says why *this* molecule got *this* value. It is my first attempt for **global, light** explanation of individual predictions and highlighting Permutation importance by how much a column is used, not whether higher values raise or lower logD.

---
## 1. The problem in plain terms

| Term                      | Meaning                                                                                                                                                                                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **logD** (my target, `y`) | A measured number for how a compound distributes between an oily solvent (octanol) and water at a given acidity (pH 7.4 in this dataset). Higher = more oil-loving. Drug chemists care because it influences how a compound is absorbed and behaves in the body. |
| **Input `X`**             | A molecule, written as text (a SMILES string), turned into numbers (descriptors + fingerprint bits).                                                                                                                                                             |
| **Task**                  | Regression: SMILES → logD.                                                                                                                                                                                                                                       |
| **Why scaffold splits**   | In drug discovery the next project explores *new* chemistry. A random split puts close relatives of a test molecule into training and flatters the model. Splitting by chemical family (scaffold) measures what we actually want.                                |

%%Dataset: the *Lipophilicity* set from MoleculeNet / DeepChem, about 4,200 compounds measured by AstraZeneca and deposited in ChEMBL. Raw columns: `CMPD_CHEMBLID` (ID), `smiles` (input), `exp` (label, renamed `logd`).%%

## 2. Headline results

All numbers are RMSE in logD units, mean over 3 seeds × 5 folds, on the **scaffold split** unless stated.

* **No-chemistry floor:** predicting the mean gives **1.201**, which is the SD of logD.
* **Best model: `HGB_both`** (gradient-boosted trees on descriptors + fingerprint) scores **0.685** on new families, versus **0.624** on a random split. New families cost about **+0.06 RMSE** for every model, so the penalty is real but modest.
* **Trees beat linear and nearest-neighbour** by clear margins (95% CIs exclude 0). A neural net (`MLP_both`) is **statistically tied** with the trees.
* **"I don't know" works.** Refusing molecules whose nearest training neighbour has similarity below 0.5 keeps 61% of molecules at RMSE **0.580**; the refused 39% have RMSE **0.824** (all molecules together: 0.685).
* **Light explainability:** two descriptor columns hold about 73% of the model's total permutation importance; collapsing near-copy columns barely changes that.

## 3. Quick start
### 3.1 Environment

RDKit must come from **conda**. A pip install can fail with confusing errors on some systems.

```bash

conda create -n logd python=3.12 -y # any recent Python (3.11+) should do: pandas 3 / numpy 2 need it

conda activate logd

conda install -c conda-forge rdkit=2026.09.1 -y # conda writes the month zero-padded

# everything else via pip, without the rdkit line (conda already installed it):

grep -v -E "^(rdkit|appnope|torch)" requirements.txt > requirements_pip.txt # macOS/Linux; on Windows delete those lines by hand

pip install -r requirements_pip.txt python-dotenv google-genai

```


%%Pinned in `requirements.txt` (key ones): `rdkit==2026.9.1`, `scikit-learn==1.9.1`, `pandas==3.1.1`, `numpy==2.5.5`, `scipy==1.18.1`, `pyarrow==26.0.1` (parquet files), `PyYAML==6.0.3`.%%

### 3.2 The `.env` file (only for the LLM step)

Write your own .env file with your api key for llm part.

```ini

GEMINI_API_KEY=your-key-from-Google-AI-Studio

GEMINI_MODEL=gemini-3.5-flash-lite # optional; this is the default in the script

```

%%Without a key everything still runs: step 05 writes `xai_facts.json`, skips the LLM, and prints the instruction text so you can paste it plus the JSON into any chat.%%

### 3.3 Run

```bash

python scripts/run_all.py # runs only the steps whose output is missing

python scripts/run_all.py --force 03_cv_ladder # rerun one step

python scripts/run_all.py --force all # rerun everything

```
## 4. Repository layout

```

.

├── config.yaml # every setting in one place

├── .env # API key (not committed)

├── requirements.txt

├── data/

│ ├── raw/Lipophilicity.csv

│ └── processed/ # molecules.parquet, features.parquet, oof.parquet

├── reports/ # permutation_importance*.csv, descriptor_clusters.csv, xai_facts.json, xai_summary.txt

├── scripts/ # numbered steps + run_all.py

└── src/ # config, data, features, scaffold, models, validate

```

## 5. Pipeline, step by step


| Step | Script                  | What it does                                                                                                                  | Output                                  |
| ---- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| 0    | `00_download.py`        | Downloads the CSV, checks its MD5 fingerprint                                                                                 | `data/raw/Lipophilicity.csv`            |
| 1    | `01_build_data.py`      | Cleans molecules, merges duplicates, adds scaffolds, builds features, prints a data report                                    | `molecules.parquet`, `features.parquet` |
| 3    | `02_cv_ladder.py`       | Cross-validates the model ladder on random and scaffold folds, bootstraps pairwise differences, tests the "I don't know" flag | `oof.parquet` + printed tables          |
| 4    | `03_analysis_interp.py` | Permutation importance, near-copy grouping, five hypothesis checks                                                            | `reports/*.csv`                         |
| 5    | `04_llm_summary.py`     | Packs the checked facts into JSON and has an LLM word them                                                                    | `xai_facts.json`, `xai_summary.txt`     |

## 6. Working with chemistry data: tools and concepts

### What each tool is, and where it appears


| Tool / concept                  | In one line                                                                                                                                               | Where in the code                                                 |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **SMILES**                      | A molecule written as text, e.g. `CC(=O)O`                                                                                                                | the `smiles` column; parsed in `src/data.py`                      |
| **ChEMBL**                      | Public database of bioactive molecules run by EMBL-EBI; the source of this dataset and its IDs                                                            | `CMPD_CHEMBLID`                                                   |
| **RDKit**                       | Open-source cheminformatics library: parses SMILES, standardises structures, computes descriptors, fingerprints and scaffolds                             | `src/data.py`, `src/features.py`                                  |
| **Descriptors** (`d_*`)         | About 200 numbers RDKit computes per molecule (weight, polar surface area, counts of groups, a calculated logP, ...)                                      | `Descriptors.CalcMolDescriptors`                                  |
| **Morgan fingerprint** (`fp_*`) | A 1,024-bit vector; each bit flags whether some small circular substructure (radius 2) occurs. Different substructures can share a bit (hash collisions). | `rdFingerprintGenerator`, `morgan_radius: 2`, `morgan_bits: 1024` |
| **Tanimoto / Jaccard**          | Overlap of two bit vectors: shared bits ÷ bits in either. Same thing for binary fingerprints.                                                             | kNN metric; the "I don't know" similarity                         |
| **Murcko scaffold**             | A molecule stripped down to its ring systems and linkers: its "family" label                                                                              | `add_scaffold`, the grouping for scaffold CV                      |
| **Stereoisomers**               | Same atoms and bonds, different 3D arrangement. Descriptors and fingerprints mostly can't tell them apart.                                                | merged in `merge_duplicates`                                      

  
### The pipeline on a toy molecule

Illustrative only; it mirrors what `src/data.py` and `src/features.py` do.


```python

from rdkit import Chem

from rdkit.Chem import Descriptors, rdFingerprintGenerator

from rdkit.Chem.MolStandardize import rdMolStandardize

from rdkit.Chem.Scaffolds import MurckoScaffold

  

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O.[Na]") # text -> molecule object (None if invalid)

parent = rdMolStandardize.LargestFragmentChooser().choose(mol) # drop the counter-ion

smiles = Chem.MolToSmiles(parent) # canonical form: one string per structure

scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent) # family label ("c1ccccc1" for this one)

desc = Descriptors.CalcMolDescriptors(parent) # dict of ~200 numbers -> d_* columns

fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprintAsNumPy(parent) # 1,024 bits -> fp_* columns

```


### Data decisions

| Decision            | Choice                                                                                      | Why                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Sampling            | Keep every valid molecule                                                                   | 4,200 rows is small; subsampling only costs accuracy                                                |
| Salts / mixtures    | Keep the largest fragment                                                                   | A counter-ion is not the drug and would add fake features                                           |
| Duplicates          | Merge on structure, take mean logD, keep `n_dup` and `logd_spread`                          | The same molecule must not land in both train and test                                              |
| Stereoisomers       | Merge them too (key without stereo)                                                         | Features see them as identical, so splitting them would leak                                        |
| Descriptors         | All RDKit descriptors; absurd values (\|x\| > 1e6) → NaN; drop constant columns and `d_Ipc` | `d_Ipc` is huge-ranged and NaN-heavy and breaks scaling; trees handle NaN, Ridge/MLP get an imputer |
| Fingerprint         | Morgan radius 2, 1,024 bits; drop constant bits                                             | Standard substructure size; enough bits for ~4,200 molecules                                        |
| Final feature table | **1,231 columns = 207 descriptors + 1,024 bits**                                            | printed by step 01                                                                                  |


## 7. Models: "the ladder"

%%Each rung answers one question; comparing neighbouring rungs changes one thing at a time.%%
Only fixed hyperparameters are used (see `config.yaml`); there is no per-model tuning apart from Ridge's alpha. That is a deliberate simplification and a limitation.

| Model                              | Features                         | Question it answers                                                                                                           |
| ---------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `B0_mean`                          | none                             | What error with no chemistry at all? (the floor)                                                                              |
| `B1_ridge_desc`                    | descriptors                      | Is logD roughly linear in the descriptors? Ridge's alpha is chosen by scaffold CV *inside* each training fold.                |
| `B2_knn_fp`                        | fingerprint                      | Can "copy the most similar known molecules" work? (k=5, Jaccard distance)                                                     |
| `HGB_desc` / `HGB_fp` / `HGB_both` | descriptors / fingerprint / both | Which representation carries the signal? Identical settings for all three, so differences come from the features, not tuning. |
| `MLP_both`                         | both                             | Does a neural net beat trees on ~4,200 molecules? Descriptors are scaled; fingerprint bits pass through untouched.            |

## 8. Evaluation design


* **Scaffold-grouped CV:** each chemical family goes entirely to one fold (`src/scaffold.py`). A **random split** is run alongside to measure the gap.
* **Out-of-fold (OOF) table:** every molecule gets one honest prediction per model and seed, saved to `oof.parquet`. All tables and tests are computed from it.
* **Paired family bootstrap** (`src/validate.py`): to ask "is A really better than B?", whole scaffold families are resampled 2,000 times and the RMSE difference is recomputed. Model A is better only if the whole 95% CI is below 0.
* **"Knows its limits":** for each test molecule, the similarity to its nearest **training** molecule (computed per fold) is the confidence signal. Molecules below a threshold *t* are refused. Coverage is always reported next to the error, because refusing most molecules makes the rest look good without being useful. Thresholds were fixed in advance, not tuned on test errors.

## 9. Results

Run: 3 seeds × 5 folds; numbers are from my run of `02_cv_ladder.py`.

### 9.1 RMSE by split

| Model         |    Random |  Scaffold | Gap (scaffold − random) |
| ------------- | --------: | --------: | ----------------------: |
| B0_mean       |     1.201 |     1.201 |                   0.001 |
| B1_ridge_desc |     0.817 |     0.865 |                   0.048 |
| B2_knn_fp     |     0.924 |     1.006 |                   0.082 |
| HGB_desc      |     0.645 |     0.707 |                   0.062 |
| HGB_fp        |     0.800 |     0.866 |                   0.066 |
| **HGB_both**  | **0.624** | **0.685** |                   0.060 |
| MLP_both      |     0.639 |     0.699 |                   0.059 |

### 9.2 Scaffold split, other metrics

| Model         |      RMSE | SD over seeds |   MAE |     R² | Spearman |
| ------------- | --------: | ------------: | ----: | -----: | -------: |
| B0_mean       |     1.201 |         0.001 | 0.971 | −0.002 |   −0.055 |
| B1_ridge_desc |     0.865 |         0.009 | 0.658 |  0.480 |    0.714 |
| B2_knn_fp     |     1.006 |         0.009 | 0.773 |  0.298 |    0.538 |
| HGB_desc      |     0.707 |         0.006 | 0.538 |  0.653 |    0.789 |
| HGB_fp        |     0.866 |         0.006 | 0.671 |  0.479 |    0.676 |
| **HGB_both**  | **0.685** |         0.004 | 0.519 |  0.674 |    0.805 |
| MLP_both      |     0.699 |         0.006 | 0.521 |  0.661 |    0.810 |

### 9.3 Is model A really better than B? (scaffold split, RMSE(A) − RMSE(B), 95% CI)

| A vs B                    | What it isolates                   | Difference | 95% CI           | Verdict                 |
| ------------------------- | ---------------------------------- | ---------: | ---------------- | ----------------------- |
| HGB_both vs HGB_desc      | does the fingerprint add anything? |     −0.022 | [−0.029, −0.015] | A better (small)        |
| HGB_fp vs HGB_desc        | fingerprint vs descriptors         |     +0.159 | [0.136, 0.182]   | descriptors better      |
| HGB_desc vs B1_ridge_desc | trees vs linear                    |     −0.158 | [−0.206, −0.116] | trees better            |
| HGB_fp vs B2_knn_fp       | learned rules vs copy neighbours   |     −0.140 | [−0.161, −0.117] | HGB better              |
| MLP_both vs HGB_both      | neural net vs trees                |     +0.014 | [−0.004, 0.033]  | **no clear difference** |

### 9.4 The "I don't know" flag (HGB_both, scaffold split)


Overall RMSE of all molecules: 0.685.


| Refuse if similarity < t |  Coverage | RMSE (predicted) | RMSE (refused) |
| -----------------------: | --------: | ---------------: | -------------: |
|                      0.3 |     93.6% |            0.661 |          0.963 |
|                      0.4 |     76.8% |            0.613 |          0.882 |
|                  **0.5** | **61.4%** |        **0.580** |      **0.824** |
|                      0.6 |     46.3% |            0.545 |          0.785 |
|                      0.7 |     26.7% |            0.527 |          0.734 |

Median nearest-neighbour similarity to the training set is **0.69** on random folds and **0.58** on scaffold folds (10th percentile 0.34 vs 0.32). That gap is a plausible reason why the scaffold penalty is small: even a scaffold split leaves some similar molecules in training. I have not tested this explanation.
## 10. Light XAI with an LLM that communicates, not creates

### 10.1 What the pipeline does

1. **Importance (`04`):** for each scaffold fold, train `HGB_both`, shuffle one column at a time on the **test** molecules, and record how much RMSE grows (permutation importance, 5 repeats, seed 0).
2. **Near-copies:** 207 descriptors are grouped into **115 clusters** (Ward linkage cut at distance 0.5 on 1 − |Spearman correlation|; 74 singletons, largest cluster 21). The model is retrained with one representative per cluster plus all fingerprint bits (1,139 columns) to check whether correlated columns were splitting each other's credit.
3. **Five ML hypotheses, each with a rule fixed in advance** (`04`/`05`): the code computes the numbers and the verdict.
4. **LLM wording (`05`):** the checked facts go into `xai_facts.json`; Gemini (via `google-genai`, `temperature=0`) turns them into a short summary. The prompt forbids new facts, chemistry explanations and causal claims. A **number guard** warns if the reply contains a number that is not in the facts.

The LLM never sees molecules, data or raw importances, only the JSON the code built.

### 10.2 Results

Top columns by permutation importance (RMSE increase when shuffled; 1,231 columns in total):

| Rank | Column | Importance | After collapsing near-copies |
|---:|---|---:|---:|
| 1 | `d_MolLogP` | 0.452 | 0.443 |
| 2 | `d_fr_COO` | 0.322 | 0.324 |
| 3 | `d_VSA_EState5` | 0.017 | 0.018 |
| 4 | `d_PEOE_VSA13` | 0.013 | 0.013 |
| 5 | `fp_284` | 0.011 | 0.012 |


Hypotheses (rules are conventions I set before seeing the numbers; they are not statistical tests):


| ID  | Statement                              | Rule                                                                 | Numbers                                             | Verdict               | My reading                                                                                             |
| --- | -------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------ |
| H1  | Model relies on very few columns       | top-2 hold ≥ 50% of total importance                                 | top-1 42%, top-2 73%, top-10 80%                    | supported             | Strongly true.                                                                                         |
| H2  | Most columns are not used              | ≥ 50% of columns have importance ≤ 0                                 | 614 of 1,231 (49.9%); only 75 above 0.001           | not supported         | Borderline: it prints as "50%" but misses the rule by one column. Read it as "about half are unused".  |
| H3  | Top columns are mainly descriptors     | ≥ 6 of top-10 are descriptors                                        | 9 of 10                                             | supported             | Matches the ladder (descriptors beat fingerprint).                                                     |
| H4  | Near-copies were hiding importance     | a top column has near-copies and its importance rises when collapsed | `d_MolLogP` 0.452 → 0.443; `d_fr_COO` 0.322 → 0.324 | supported by the rule | **The rule is too lenient.** The changes are −2% and +0.8%, so near-copies were *not* hiding anything. |
| H5  | Top columns stay on top after grouping | top-2 (or representatives) still top-2                               | same two columns before and after                   | supported             | Stable.                                                                                                |

  

The LLM's output (`reports/xai_summary.txt`, unedited):
  

```

Using 1231 columns and 5 scaffold folds, the HGB_both model was evaluated by measuring the increase in RMSE

when one column is shuffled on test molecules only. Across 5 repeats per column with seed 0, five hypotheses

were tested to examine column importance.

  

H1: supported, with top-1 42%, top-2 73%, top-10 80%; top column d_MolLogP 0.452 (fold SD not saved).

H2: not supported, with 50% of 1231 columns are <= 0; 75 are above 0.001.

H3: supported, with 9 of the top-10 are descriptors, 1 are fingerprint bits.

H4: supported, with d_MolLogP: 1 near-copies, 0.452 alone -> 0.443 collapsed; d_fr_COO: 2 near-copies, 0.322 alone -> 0.324 collapsed.

H5: supported, with before ['d_MolLogP', 'd_fr_COO'], after ['d_MolLogP', 'd_fr_COO'].

  

Limits include that importance shows what the model uses, not what causes the target, and that verdict rules

such as 50% or 6 of 10 were fixed by the author in advance and are not statistical tests.

```

  
The LLM reproduces the facts faithfully, including the lenient H4 verdict. It does not and cannot catch a weak rule, which is why the rules are written and checked in code.

*Background for non-chemists, not claimed as a finding:* `d_MolLogP` is RDKit's calculated estimate of lipophilicity (Crippen logP) and `d_fr_COO` counts carboxylic-acid groups. The analysis treats both purely as column names.


## 10. Configuration (`config.yaml`)

| Key                                         | Value                                                            | Notes                                                                   |
| ------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `data.url`, `data.md5`                      | Deepchem S3 link, checksum                                       | MD5 is a fingerprint of the file's contents                             |
| `data.smiles_col` / `target_col` / `id_col` | `smiles` / `exp` / `CMPD_CHEMBLID`                               | raw column names                                                        |
| `features.morgan_radius` / `morgan_bits`    | 2 / 1024                                                         | fingerprint size                                                        |
| `features.desc_abs_max`                     | 1,000,000                                                        | larger \|value\| → NaN                                                  |
| `cv.n_splits`, `cv.seeds`                   | 5, `[0, 1, 2]`                                                   | use `[0]` for a quick run                                               |
| `models.ridge_alphas`                       | `[-3, 3, 12]`                                                    | arguments of `np.logspace`: 12 values from 0.001 to 1000                |
| `models.knn_k`                              | 5                                                                |                                                                         |
| `models.hgb`                                | 500 iterations, lr 0.05, L2 = 1.0, no early stopping             | same for all three HGB rows                                             |
| `models.mlp`                                | (256, 128), alpha 0.01, batch 64, 200 iterations, early stopping | early stopping uses a random 10% hold-out, so it is slightly optimistic |
| `bootstrap.n_boot`, `bootstrap.seed`        | 2000, 0                                                          |                                                                         |

## 11. Data source, credit and licence
* Dataset: *Lipophilicity*, MoleculeNet (Wu et al., 2018, "MoleculeNet: a benchmark for molecular machine learning", *Chemical Science*), distributed through DeepChem; original measurements by AstraZeneca, deposited in ChEMBL (CHEMBL3301361).

* Tools: RDKit, scikit-learn, pandas, scipy, Google Gemini API (`google-genai`).

* Code licence: add your choice (e.g. MIT). Check the dataset's own terms before redistributing the data.
