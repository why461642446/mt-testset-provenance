# Supplementary Material

**What Does a Machine-Translated Test Set Measure? Provenance Effects in Multilingual Intent Classification**

This file carries the material moved out of the main text during revision. Section and table numbers refer to the main text unless stated otherwise.

---

## S1. Efficiency measurements (formerly Section 4.5)

### 4.5 Efficiency, and what these numbers can be compared across

**Table 8a. Measured training time per run (seconds).** Non-pretrained rows: mean over seeds 42–46 (n = 5). Pretrained rows: mean over seeds 42–44 (n = 3). **Read the two blocks separately.** The non-pretrained models were trained on CPU for 15 epochs; the pretrained models were trained on a Tesla T4 for 3 epochs. Rows within a block are directly comparable; rows across blocks differ in both hardware and training budget and must not be compared as a ratio. The epoch counts differ because the pretrained models converge within 3 epochs and the non-pretrained ones do not.

 | Model | en-US | ko-KR | Hardware | Epochs |
| --- | --- | --- | --- | --- |
| Non-pretrained block | | | CPU | 15 |
| TextCNN | 156 | 226 | CPU | 15 |
| BiLSTM | 270 | 308 | CPU | 15 |
| Pretrained block | | | Tesla T4 | 3 |
| DistilBERT | 417 | 469 | T4 | 3 |
| BERT | 806 | 861 | T4 | 3 |
| ALBERT (English-only) | 834 | — | T4 | 3 |

Two comparisons are supported by this table, and two that are often made are not.

**Within the pretrained block — same hardware, same budget — DistilBERT is the better deployment choice.** It scores 0.6 points lower in English and 1.6 lower in Korean while being markedly cheaper to run. We state the direction and the accuracy cost, and **we do not convert the training-time difference into a ratio**, for the reason given in the next paragraph.

**Within the non-pretrained block — likewise matched — TextCNN and the padding-masked BiLSTM are close, and neither dominates.** On the three locales TextCNN leads by **0.58 points in English** (80.55 against 79.97 at whitespace, sd 0.64 and 0.70) and by **0.47 in Korean** (73.83 against 73.36, sd 0.76 and 1.07), and **trails by 0.32 in Chinese** (10.31 against 10.63, sd 0.22 and 0.22); all three differences are small relative to the between-seed standard deviations, and the Chinese comparison is in any case invalid because under whitespace the models are not reading the text (Section 4.2.4).

**We therefore do not claim that TextCNN is the more accurate of the two.** The claim we do make is about deployability: a non-pretrained model is two orders of magnitude smaller, and English and Korean accuracy is close enough that the size difference dominates for a memory-constrained device.

**Training time is not a reproducible measurement in this study, so we report no ratios from it.** Re-running an identical configuration — same model, same locale, same hardware, same epoch budget — produced wall-clock training times differing by up to **3.1×** between runs (a Korean BiLSTM: 306.5 s in one session against 98.4 s in another), because the machine was shared with other work. Any ratio computed from such measurements reflects machine load rather than the models compared. We therefore report per-run timings only for runs executed in the same session, and draw no efficiency conclusion from training time at all.

**The cross-block ratio is not a measurement and we do not report one.** The non-pretrained models ran on CPU for 15 epochs and the pretrained models on a T4 for 3; those differ in processor, epoch count, and optimiser. A comparison of the non-pretrained models against DistilBERT on training time would be invalid for exactly that reason, and we do not draw it.

 TextCNN is orders of magnitude smaller than the pretrained checkpoints examined here while retaining competitive accuracy in English and Korean; because checkpoint size varies by locale and some English and Korean parameter counts are nominal rather than instrumented measurements, we do not express this trade-off as a single cross-locale parameter-to-accuracy ratio. That the smaller model remains competitive is the fact that matters for a memory-constrained device. *ALBERT also inverts the parameter-count intuition.* It has roughly a tenth of BERT's parameters, yet trains **more** slowly (834 s against 806 s) on the same hardware and the same budget, and scores 3.4 points lower on English. Parameter count is a poor proxy for training cost at fixed sequence length, where depth and per-layer attention cost dominate.

 **Table 8b.** Parameters, model size, inference latency, and peak GPU memory, measured per run. Latency is the mean over the full 2,974-utterance test set; peak GPU memory is recorded during evaluation.

 **Parameter count for the non-pretrained models depends on the tokenizer vocabulary, so those rows are given per (locale, tokenizer) rather than pooled.**

| Model (checkpoint) | Locale | Tokenizer | Vocab | Params (M) | Size (MB) | Latency (ms/sample) | Peak GPU (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TextCNN | en | whitespace | 5,304 | 0.67 | 2.55 | 0.16 | n/a (CPU) |
| TextCNN | en | character | 32 | **0.14** | 0.55 | 0.08 | n/a (CPU) |
| TextCNN | ko | whitespace | 9,923 | **1.13** | 4.31 | 0.18 | n/a (CPU) |
| TextCNN | ko | character | 1,037 | **0.24** | 0.92 | 0.09 | n/a (CPU) |
| TextCNN | ko | morphological | 6,425 | **0.78** | 2.98 | 0.14 | n/a (CPU) |
| TextCNN | zh | whitespace | 10,000 (capped) | **1.14** | 4.34 | 0.17 | n/a (CPU) |
| TextCNN | zh | character | 2,194 | **0.36** | 1.37 | 0.10 | n/a (CPU) |
| TextCNN | zh | morphological | 7,281 | **0.87** | 3.31 | 0.15 | n/a (CPU) |
| BiLSTM (padding-masked) | en | whitespace | 5,304 | 1.42 | 5.42 | 0.23 | n/a (CPU) |
| BiLSTM (padding-masked) | ko | character | 1,037 | **1.98** | 7.55 | 0.31 | n/a (CPU) |
| BiLSTM (padding-masked) | zh | whitespace | 10,000 (capped) | **2.60** | 9.92 | 0.36 | n/a (CPU) |
| ALBERT (albert-base-v2, English-only) | ko | subword | — | 11.7 | 45 | 7.97 | 218 |
| ALBERT (albert-base-v2, English-only) | zh | subword | — | 11.7 | 45 | 7.90 | 219 |
| BERT (bert-base-chinese) | zh | subword | — | 102.3 | 390 | 7.21 | 873 |
| DistilBERT (multilingual) | zh | subword | — | 135.4 | 516 | 3.62 | 1120 |
| mBERT (bert-base-multilingual-cased) | ko, zh | subword | — | 177.9 | 679 | 7.21 | 1448 |
| XLM-R (xlm-roberta-base) | ko, zh | subword | — | 278.1 | 1061 | 6.35 | 2212 |

*The non-pretrained models were trained and evaluated on CPU and therefore have no GPU peak figure. Their parameter count follows `100 × vocabulary + 138,360`, so **the same architecture spans a factor of eight across locales and tokenizers (0.14 M to 1.14 M)**, and the whitespace rows are the largest in every locale. **A single pooled value of 0.67 M for TextCNN "en, ko, zh" would be the English whitespace figure**; it understates the Korean and Chinese whitespace conditions by roughly 70% and overstates the character conditions by nearly fivefold.*

*The vocabulary sizes in this table are the sizes the **model** uses and therefore exceed the training-corpus counts in Table 6b by exactly two, for the padding and unknown symbols. The discrepancy affects only the whitespace and morphological rows, where the count exceeds 10,000 and is capped; the character rows are identical in both tables because they are far below the cap. Table 6b reports corpus counts, this table reports embedding dimensions.*

**Latency figures are also not comparable across the two blocks.** TextCNN's 0.16 ms and BiLSTM's 0.23 ms were measured on CPU; every other row was measured on a Tesla T4. A ratio between them would quantify the hardware, not the models, and we do not report one. Within the GPU block the latencies are directly comparable, and there the ordering is clean: **DistilBERT is the fastest pretrained model at 3.62 ms/sample — 1.99× faster than BERT (7.21 ms) and 1.75× faster than the next-fastest, XLM-R (6.35 ms)** — despite having more parameters than BERT, because the multilingual vocabulary inflates its size but not its per-token compute.

 **The comparison that is valid and that matters for a device is parameter count and memory, not latency across hardware.** The smallest non-pretrained configuration in this study is character-level TextCNN on English at **0.14 M parameters and 0.55 MB**; the largest non-pretrained configuration is whitespace TextCNN on Chinese at 1.14 M and 4.34 MB; BERT requires 102.3 M and 390 MB; XLM-R requires 278.1 M and 1,061 MB with a peak of 2,212 MB during evaluation. That is a range of **three** orders of magnitude in memory, and memory is the binding constraint on an embedded smart-home device. Character-level TextCNN on Korean — 0.24 M parameters, 0.92 MB — reaches 80.80 % against BERT's 86.11 % and mBERT's 51.66 % on the same locale, giving up about five points for a footprint roughly three orders of magnitude smaller. **We therefore frame the recommendation as a memory trade rather than as a speed ratio**, and we flag the CPU/GPU split explicitly so that no reader computes a cross-hardware speed-up from this table.

 **The character-level conditions are also the cheapest, which strengthens the tokenization recommendation rather than trading against it.** Choosing character segmentation reduces TextCNN's parameter count from 1.13 M to 0.24 M in Korean and from 1.14 M to 0.36 M in Chinese — a reduction of 68–79% — while *improving* accuracy. On this evidence there is no accuracy-for-memory trade in the tokenization decision at all: the script-appropriate scheme is both more accurate and substantially smaller. **What the table does settle is that parameter count is not a proxy for cost.** Taking the Chinese condition, where the relevant inference measurements are available, DistilBERT carries *more* parameters than BERT (135.4M against 102.3M) yet has substantially lower inference latency (3.62 against 7.21 ms per sample), because its parameters are concentrated in a large embedding matrix rather than in layers. ALBERT has an order of magnitude fewer parameters than either (11.7M, 45 MB, 218 MB peak) and is nevertheless **slower than BERT** at inference on the same condition (7.90 against 7.21 ms per sample) — the same inversion seen for training time in Table 8a. A table sorted by parameter count would rank all three of these models wrongly.

 Under matched training conditions DistilBERT also trains substantially faster than BERT, in English (417 against 806 s) and in Korean (469 against 861 s). The ordering is checkpoint-dependent: in English, DistilBERT uses a 66.4M-parameter monolingual checkpoint against BERT's 110.1M, while the parameter-count inversion described above occurs only for the Chinese checkpoint pairing used here. **A reporting caveat this table makes visible.** Parameter count is a property of the *checkpoint*, not of the model family: the multilingual DistilBERT vocabulary (119,547 subword types) inflates its embedding matrix far beyond the 66M of its English counterpart. A table aggregating by architecture name across languages can therefore state the opposite of the truth, which is why we report checkpoint, size, and locale together.



---

## S2. Zero-shot cross-lingual transfer and the audit of `bert-base-chinese` (formerly Section 4.4)

### 4.4 Zero-shot cross-lingual transfer

### 4.4 Zero-shot cross-lingual transfer

**Table 7.** Zero-shot cross-lingual transfer: trained on `en-US`, evaluated on `ko-KR` / `zh-CN` without any target-language fine-tuning. Accuracy (%), 60 classes, chance = 1.67%, mean ± sample standard deviation over seeds 42/43/44 (n = 3). **The checkpoint is part of the result**, because two rows that look like different models are not.

| Arm | Checkpoint | ko-KR | zh-CN |
| --- | --- | --- | --- |
| mBERT | `bert-base-multilingual-cased` | **29.67 ± 3.81** | 51.66 ± 4.05 |
| BERT (Korean) | `bert-base-multilingual-cased` | **29.67 ± 3.81** | — |
| BERT (Chinese) | `bert-base-chinese` | — | **65.78 ± 1.70** |
| DistilBERT | `distilbert-base-multilingual-cased` | 16.58 ± 2.25 | 46.58 ± 0.84 |
| XLM-R | `xlm-roberta-base` | **70.47 ± 0.94** | **77.05 ± 2.02** |

**The Korean `BERT` and `mBERT` rows are the same experiment, not two.** Our configuration has no widely used monolingual Korean BERT, so both were instantiated from `bert-base-multilingual-cased`; the three per-seed accuracies are identical to four decimal places (30.3631, 25.5548, 33.0868). **We report them as one arm.** A reader comparing "BERT" against "mBERT" on Korean would otherwise be reading a duplicate as confirmation. **The Chinese `BERT` row is independent and is the most surprising result in this section.** `bert-base-chinese` — a monolingual Chinese checkpoint, fine-tuned on English and evaluated on Chinese without any Chinese fine-tuning — reaches **65.78 ± 1.70**, against mBERT's 51.66 ± 4.05 on the same condition. This is a 14.1-point margin from a model that has never been trained on multilingual data at all. **The audit in Table 7b verifies this row against the four standard failure modes** — see Section 4.4 — including a randomly initialised control that rules out data leakage. **A result this surprising requires an audit rather than a caveat.** The concern is that a monolingual Chinese vocabulary cannot plausibly transfer English supervision to Chinese characters, and that 14 points over a multilingual model is therefore more likely to indicate a pipeline error — a mis-specified language pair, a training file resolved to the wrong locale, or an evaluation set that is not what it is labelled. **We ran the four checks that distinguish these, and the result survives all of them.**

**Table 7b. Audit of the zero-shot arms** (three seeds, sample standard deviation). Each row is one training run evaluated three times.

| Checkpoint | en-US test | ko-KR test | zh-CN test |
| --- | --- | --- | --- |
| `bert-base-chinese` | 82.97 ± 0.42 | 6.32 ± 0.38 | **65.78 ± 1.70** |
| `bert-base-multilingual-cased` | 83.49 ± 0.37 | 29.67 ± 3.81 | 51.66 ± 4.05 |
| **`bert-base-chinese`, randomly initialised** | **77.74 ± 0.91** | **5.14 ± 2.10** | **4.34 ± 0.20** |
| *chance* | 1.67 | 1.67 | 1.67 |
| *majority class* | 7.67 | 7.70 | 7.67 |

**Check 1 — which file the loader actually opened.** The audit recorded, per locale, the absolute path each `load_data` call resolved to, an MD5 of that file, and the first training sentence it contained. The English run opened `/kaggle/working/massive/1.1/data/en-US.jsonl`, MD5 `135c741954f86f23d37784dba247c78a` — the same file and the same checksum recorded in Appendix A — and its first training sentence is **"wake me up at nine am on friday"**. **0 of its 403,436 characters are CJK**, against 100% of training sentences in the Korean and Chinese files. **A mis-resolved path is excluded.**

**Check 2 — the random-initialisation control.** This is the decisive one, because **a pipeline that leaked labels or test text would raise every model that reads it, pretrained or not.** We therefore trained the *same architecture with no pretrained weights at all* through an identical pipeline. It reaches **77.74 ± 0.91 on English**, confirming that the training loop works. On Chinese it reaches **4.34 ± 0.20** — **below the 7.67% majority-class rate and 2.7 points above chance**, and on Korean 5.14 ± 2.10. **The fine-tuned model reaches 65.78.** A 61-point separation between a randomly initialised encoder and a pretrained one, on the same data through the same code, cannot be produced by leakage: whatever the fine-tuned model is using, it is something that pretraining put there.

**Check 3 — where the 65.78% comes from.** Accuracy is broad rather than concentrated: **37 of the 59 intents exceed 50% and 18 exceed 80%**, and the strong intents span the domains — `qa_currency` 100%, `iot_coffee` 97%, `iot_wemo_off` 94%, `iot_cleaning` 92%, `email_sendemail` 91%, `weather_query` 90%, `transport_taxi` 88%, `cooking_recipe` 88%. **A leakage artefact would concentrate accuracy on the few intents whose test items happen to be in the training set**; this pattern does not. The model predicts **54 of the 60 classes**, and its single most frequent prediction accounts for only **7.1%** of test items — again the opposite of the collapsed readout a leak produces.

**Check 4 — digits and Latin characters.** Shared non-Chinese symbols are an obvious alternative explanation: a model that cannot read Chinese might still key on numerals and Latin tokens. We therefore re-evaluated the Chinese test set with every digit and Latin character removed. Accuracy is **66.48 / 63.69 / 66.71** across the three seeds, against **66.75 / 63.82 / 66.78** on the unmodified text — **a change of at most 0.3 points.** Only 2.1% of test sentences contain such characters, and accuracy on those is 65.08% against 65.80% on the rest. **The effect does not run through shared symbols.**

**What the audit does not explain, and we say so plainly.** It establishes that the result is **real, reproducible and not a data-handling artefact**; it does not establish **why** a model fine-tuned only on English represents Chinese so much better than mBERT does. Three candidate accounts remain, and our design separates none of them: (i) `bert-base-chinese` has a Chinese-specialised vocabulary, though we measured the two tokenizers to give essentially identical token counts on Chinese test text (12.31 tokens per utterance for both; `[UNK]` 0.00% against 0.08%), which weakens this account; (ii) English fine-tuning degrades mBERT's cross-lingual alignment more than it degrades the Chinese representations of a monolingual model; (iii) some property of the monolingual checkpoint's pretraining corpus that our controls do not isolate. **We report the measurement and flag the mechanism as open** (Section 7.2) rather than presenting it as understood.

**The audit also supplies the checksums this study previously lacked.** The three MASSIVE files used were `en-US.jsonl` (MD5 `135c741954f86f23d37784dba247c78a`), `ko-KR.jsonl` (`18f2b556dde5b8ac77ace58d0f2a6c1d`) and `zh-CN.jsonl` (`2d2a3fb725b3a37793e2dcd16442b1e0`), each with 11,514 training, 2,033 development and 2,974 test entries over the same 60 intents. The original runs did not record these, and one consequence of that omission is that the audit had to be run at all; **Appendix A states the gap and what we did about it.**

**XLM-R dominates mBERT by a wide and consistent margin.** The gap is 40.8 points on Korean (70.47 against 29.67) and 25.4 points on Chinese (77.05 against 51.66), and it is not a matter of seed noise: XLM-R's standard deviation is 0.94 and 2.02, against mBERT's 3.81 and 4.05. mBERT is both worse and less stable when transferred.

 **DistilBERT is the weakest arm on both locales** (16.58 Korean, 46.58 Chinese), below mBERT despite sharing its tokenizer family. This is worth recording because DistilBERT is the efficiency recommendation in Section 4.5: **the model we recommend for cost is the one we would not recommend for zero-shot transfer**, and a practitioner choosing on footprint alone would get this wrong. **The size of the XLM-R margin is the practically relevant finding.** For Korean, zero-shot XLM-R (70.47) lands within 3.4 points of in-language supervised TextCNN (73.83) and within 11 of in-language BERT (81.73). That is an unusually favourable position for a model that has never seen a Korean training example, and it means that for a new locale the first question is not whether to collect labelled data but whether XLM-R with no target-language training already clears the deployment bar.

 **The Chinese zero-shot numbers carry a caveat that must be stated.** XLM-R reaches 77.05 on Chinese, which is well *above* its Korean figure; yet for Chinese we showed in Section 4.1 that the same class of model reaches 86.11 in-language. The ordering (in-language > zero-shot > chance) holds, but we caution against reading the Chinese zero-shot figure as evidence that Chinese transfers more easily than Korean, because the two locales differ in subword coverage and in how much Chinese text the checkpoints saw during pretraining — an effect we do not isolate here and cannot attribute from these results alone.



---

## S3. The 2x2 on the full test split (Table 5f)

**Table 5f. The 2x2 on the full test split (2,974 entries), sample standard deviation.** The non-pretrained rows use **five seeds (42–46)**; the mBERT and XLM-R rows, shown for comparison, use **three seeds (42–44)** and **their own native subword tokenizers** — they are not character-level models and the "character" label applies only to the non-pretrained rows.

| Locale | Model | Tokenizer | L→L | L→MT | MT→MT | MT→L | L→MT cost | Deployment overstatement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ko-KR | TextCNN | character | 80.80 ± 0.59 | 58.20 ± 0.94 | 76.84 ± 0.45 | 65.27 ± 1.14 | −22.60 | +11.57 |
| ko-KR | BiLSTM | character | 77.59 ± 0.43 | 53.35 ± 1.16 | 72.84 ± 0.32 | 60.41 ± 0.96 | −24.24 | +12.43 |
| zh-CN | TextCNN | character | 80.62 ± 0.46 | 66.88 ± 0.42 | 78.36 ± 0.44 | 70.60 ± 0.91 | −13.74 | +7.76 |
| zh-CN | BiLSTM | character | 79.69 ± 0.30 | 63.50 ± 0.85 | 76.72 ± 0.47 | 68.36 ± 0.65 | −16.19 | +8.37 |
| ko-KR | TextCNN | whitespace | 73.83 ± 0.76 | 33.37 ± 0.93 | 62.95 ± 0.36 | 34.02 ± 0.87 | −40.46 | +28.93 |
| ko-KR | BiLSTM | whitespace | 73.36 ± 1.07 | 31.56 ± 2.13 | 62.38 ± 0.92 | 33.70 ± 1.12 | −41.80 | +28.68 |
| zh-CN | TextCNN | whitespace | 10.31 ± 0.22 | 5.36 ± 0.17 | 10.45 ± 0.17 | 5.36 ± 0.20 | −4.95 | +5.09 |
| zh-CN | BiLSTM | whitespace | 10.63 ± 0.22 | 5.41 ± 0.14 | 10.64 ± 0.12 | 5.39 ± 0.14 | −5.22 | +5.25 |
| **ko-KR** | **mBERT** | **native subword** | **81.73 ± 0.65** | **63.72 ± 1.63** | **77.40 ± 1.23** | **70.89 ± 1.58** | **−18.01** | **+6.51** |
| **zh-CN** | **mBERT** | **native subword** | **83.79 ± 0.64** | **74.31 ± 0.30** | **81.14 ± 0.09** | **78.50 ± 0.44** | **−9.48** | **+2.63** |
| **ko-KR** | **XLM-R** | **native subword** | **83.58 ± 0.40** | **72.10 ± 1.40** | **79.72 ± 1.02** | **78.08 ± 1.32** | **−11.48** | **+1.65** |
| **zh-CN** | **XLM-R** | **native subword** | **84.75 ± 0.66** | **77.08 ± 1.13** | **81.72 ± 0.61** | **81.73 ± 0.35** | **−7.67** | **−0.01** |

---
## S4. Deployment overstatement under a matched versus a mismatched translation system (Table 5h)

**Table 5h. Deployment overstatement under a matched versus a mismatched translation system** (character-level segmentation, fixed evaluation subset, three seeds, 2,000-resample bootstrap intervals over test sentences).

| Locale | Model | MT→MT (same system) | MT→MT2 (cross system) | MT→L | Deployment overstatement, same system | Deployment overstatement, cross system | Reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ko-KR | TextCNN | 75.37 | 73.90 | 63.80 | **+11.57** [+10.13, +13.09] | **+10.11** [+8.67, +11.56] | **−1.47** [−2.57, −0.39] |
| ko-KR | BiLSTM | 70.26 | 68.90 | 58.94 | **+11.32** [+9.73, +12.87] | **+9.95** [+8.45, +11.48] | **−1.36** [−2.49, −0.17] |
| zh-CN | TextCNN | 77.20 | 76.53 | 69.87 | **+7.32** [+5.92, +8.83] | **+6.66** [+5.34, +7.99] | **−0.66** [−1.65, +0.32] |
| zh-CN | BiLSTM | 74.76 | 73.61 | 67.59 | **+7.17** [+5.78, +8.59] | **+6.02** [+4.67, +7.39] | **−1.15** [−2.21, −0.09] |

---
## S5. Pretrained vocabulary coverage (Table 6c)

**Table 6c. Pretrained vocabulary coverage of the target scripts.** Unknown-token rate and mean token count when each pretrained tokenizer is applied directly to the test split; no fine-tuning involved. Measured over the first 1,500 test utterances per locale.

 | Locale | Tokenizer (vocabulary size) | [UNK] rate | Mean tokens |
| --- | --- | --- | --- |
| en-US | albert-base-v2 (30,000) | 0.00% | 9.24 |
| en-US | bert-base-multilingual-cased (119,547) | 0.00% | 10.27 |
| ko-KR | `albert-base-v2` (30,000) | 40.65% | 11.31 |
| ko-KR | bert-base-multilingual-cased (119,547) | 0.27% | 11.62 |
| zh-CN | `albert-base-v2` (30,000) | 24.75% | 4.06 |
| zh-CN | bert-base-multilingual-cased (119,547) | 0.09% | 11.70 |

---
## S6. Train–test overlap and duplicated-entry accuracy (Tables 6b-ii and 6b-iii)

**Table 6b-ii. Exact overlap between the training and test splits, and what it is worth.** A test entry is a duplicate if its text occurs verbatim in the training split of the same locale. Both denominators are given because they differ: repeated entries share a single unique string.

| Locale | Test entries | Unique test strings | Duplicate entries | Rate (entries) | Duplicate strings | Rate (unique) |
| --- | --- | --- | --- | --- | --- | --- |
| en-US | 2,974 | 2,970 | 21 | 0.71% | 21 | 0.71% |
| ko-KR | 2,974 | 2,919 | **187** | 6.29% | 155 | 5.31% |
| zh-CN | 2,974 | 2,921 | **223** | 7.50% | 187 | 6.40% |

**Table 6b-iii. Accuracy on duplicated versus non-duplicated Chinese test entries** (whitespace TextCNN, localized condition, seeds 42–44, n = 3 per entry; entries as in the preceding table).

| Subset | Entries × seeds | Correct | Accuracy |
| --- | --- | --- | --- |
| All test entries | 8,922 | 926 | 10.31% |
| **Duplicated entries** | 669 | **556** | **83.11%** |
| Non-duplicated entries | 8,253 | 370 | 4.48% |

**7.5% of the Chinese test entries can be answered by lookup, and they supply 60% of every correct prediction the whitespace model makes.** Train–test overlap of this kind is a known source of inflated benchmark scores [Lewis et al., 2021; Elangovan et al., 2021]. Remove them and the whitespace Chinese accuracy falls from 10.31% to 4.48%, below the majority-class rate of 7.67% (Section 4.2.4). The corresponding figures for Korean are a 6.29% duplicate rate worth 96.26% accuracy on those entries, which shifts the Korean baseline by about 1.5 points — real, but an order of magnitude less consequential than the Chinese case. This is the mechanism that makes the Chinese whitespace L→MT gap read −0.2 points while the character-level gap reads −13.2: the translated test set contains almost no strings that also occur in the localized training split (31 of 2,909, 1.07%), so it forfeits the lookup advantage entirely. *The vocabulary-coverage table is in Supplementary Section S5.*


---

## S7. Reproducibility manifest (formerly Appendix A)





**Table A1. Pretrained checkpoint assignment and model size.** Checkpoint identifiers are the HuggingFace names actually used for that locale. Parameter and size figures are **measured** from the released `results_v2.csv` for the Korean and Chinese runs; entries marked *(nominal)* are published checkpoint specifications, given where the corresponding run predates the efficiency instrumentation, and are labelled as such rather than presented as measurements.

 | Model | en-US | ko-KR | zh-CN | Params (M) | Size (MB) |
| --- | --- | --- | --- | --- | --- |
| BERT | bert-base-uncased | bert-base-multilingual-cased | bert-base-chinese | 102.3 (zh, measured) / 110.1 (en, nominal) / 177.9 (ko, measured) | 390 / 418 / 679 |
| DistilBERT | distilbert-base-uncased | distilbert-base-multilingual-cased | distilbert-base-multilingual-cased | 66.4 (en, nominal) / 135.4 (ko, zh, measured) | 255 / 516 |
| ALBERT | albert-base-v2 | albert-base-v2 | albert-base-v2 | 11.7 (ko, zh, measured) | 45 |
| mBERT | bert-base-multilingual-cased | bert-base-multilingual-cased | bert-base-multilingual-cased | 177.9 (measured) | 679 |
| XLM-R | xlm-roberta-base | xlm-roberta-base | xlm-roberta-base | 278.1 (measured) | 1,061 |

**The pattern the table exposes** is why we report size per checkpoint and per locale rather than per architecture: **the same architecture name can differ in size by a factor of two depending on which locale's weights are used**, and DistilBERT is the clearest case — 66.4M in English against 135.4M for the multilingual checkpoint.

 ALBERT's uniform 11.7M makes it the smallest model in every locale, which rules out capacity as an explanation for its performance profile: it is small everywhere, and yet it scores 85.12 in English against 8.80 in Korean and 7.50 in Chinese (Table 4). **What that profile does not establish is *why*.** Section 4.1 shows the collapse is confounded between monolingual pretraining and English-only vocabulary coverage — `albert-base-v2` leaves 40.65% of Korean and 24.75% of Chinese test tokens as `[UNK]` (Table 6c) — and our design does not separate the two. We therefore report the ALBERT row as a combined demonstration and draw no conclusion about which cause dominates.

 Each run emits a manifest recording: start and finish timestamps, the full command-line arguments, device and GPU model, library versions (Python / PyTorch / Transformers / NumPy / pandas), and the per-run rows (seed, sample counts, sorted class list, epochs, accuracy, F1, parameter count, size, latency, peak GPU memory) in the accompanying `results_v2.csv`. **One gap, stated plainly: the manifests for the original runs do not record a checksum of the data files they read.** They contain the argument list but not the identity of the corpus those arguments resolved to, so the training file used by any given run cannot be verified after the fact. This mattered for exactly one result — the Chinese zero-shot `bert-base-chinese` arm of Section 4.4, whose score we could not explain and therefore could not initially rule out as a pipeline artefact. **We closed the gap for that result by re-running it.** The audit of Section 4.4 recorded, per locale:

| File | MD5 | train / dev / test | Intents |
| --- | --- | --- | --- |
| `en-US.jsonl` | `135c741954f86f23d37784dba247c78a` | 11,514 / 2,033 / 2,974 | 60 |
| `ko-KR.jsonl` | `18f2b556dde5b8ac77ace58d0f2a6c1d` | 11,514 / 2,033 / 2,974 | 60 |
| `zh-CN.jsonl` | `2d2a3fb725b3a37793e2dcd16442b1e0` | 11,514 / 2,033 / 2,974 | 60 |

**All other runs in this study remain without recorded checksums**, and a reader who wishes to verify them must rely on the argument lists and on the reproduction evidence reported in Section 3.6. **That evidence consists of fifteen configurations re-run from scratch on a second machine, comparing accuracies to five decimal places; the largest absolute spread across those fifteen was 0.000050 (i.e. 0.005 percentage points, or five parts in 100,000).** **Any re-run should record an MD5 of every input file alongside the arguments**, and we would not repeat the omission. Code and result CSVs are released with the paper.

## S8. Full tokenization ablation (formerly Section 4.3, unabridged)


### 4.3 Tokenization ablation, and what the Chinese collapse is not

Figure 5. The double dissociation. Masking the padding position repairs English and Korean but not Chinese, isolating tokenization from the padding defect. Figure 6. Effect of tokenization on 60-class intent classification. Panel (a) TextCNN, panel (b) BiLSTM. **Table 6.** Effect of segmentation on non-pretrained models. Accuracy (%), 60 classes, chance = 1.67%. Mean ± sample standard deviation; **every cell is at n = 5 (seeds 42–46)**, and the per-cell seed count is given in the table itself.

| Locale | Model | whitespace | whitespace + padding mask | morphological (jieba / Okt) | character |
| --- | --- | --- | --- | --- | --- |
| en-US | TextCNN | 80.55 ± 0.64 (5) | n/a | n/a | 76.42 ± 0.56 (5) |
| en-US | BiLSTM | 39.06 ± 20.37 (5) | 79.97 ± 0.70 (5) | n/a | 67.69 ± 0.26 (5) |
| ko-KR | TextCNN | 73.83 ± 0.76 (5) | n/a | 78.66 ± 0.69 (5) | 80.80 ± 0.59 (5) |
| ko-KR | BiLSTM | 44.48 ± 17.31 (5) | **73.36 ± 1.07 (5)** | **77.48 ± 0.44 (5)** | **77.59 ± 0.43 (5)** |
| zh-CN | TextCNN | 10.31 ± 0.22 (5) | n/a | 76.69 ± 0.57 (5) | 80.62 ± 0.46 (5) |
| zh-CN | BiLSTM | 7.03 ± 0.00 (5) | **10.63 ± 0.22 (5)** | **75.66 ± 0.58 (5)** | **79.69 ± 0.30 (5)** |

*Numbers in parentheses are the seed count for that cell. **Every cell in this table is at n = 5 (seeds 42–46).** Bold cells use the padding-masked readout and are the ones on which conclusions rest; the **unmasked** column is shown only to document what the defective configuration produces, and no conclusion rests on it.*

*Values in this table are computed on the **full test split** (2,974 utterances), unlike Tables 5a–5c, which use the fixed evaluation subset. **Table 5f gives the full-split 2x2 alongside the subset version, and the paragraph below it states why both are reported.***

*`n/a`: the padding mask applies only to the recurrent model; TextCNN pools globally and has no timestep selection to correct.

 English has no morphological condition — there is no meaningful morphological segmentation of a space-delimited language that would be comparable to `jieba` or Okt.*

**Reading the recurrent row correctly required a correction, and the correction changes the conclusion.** The uncorrected unmasked readout produces a three-number ladder for Korean BiLSTM — 49.38 → 57.80 → 67.61 — that appears to show a large and orderly progression. **The first figure is the five-seed unmasked mean; the other two are three-seed values, so the ladder is not internally consistent in seed count — one more reason it is not used as evidence.** **The masked readout reverses the picture**: the masked whitespace baseline alone (73.36) exceeds every unmasked morphological and character value. Fine-grained segmentation shortens the padding, which partially repairs the readout defect by accident, so the uncorrected ladder measures two things at once and separates neither. **With the corrected and completed values the pattern is: whitespace is worse than either script-appropriate scheme by a wide margin in all four language–model combinations (gains of 4.1 to 66.4 points), and morphology is worse than characters in three of the four.** The single exception is the Korean recurrent model, where 77.48 against 77.59 is a difference of 0.11 against a standard deviation of 0.44 and is not distinguishable. **Elsewhere the morphology-to-character margin is large relative to seed noise** — 2.14 points for Korean TextCNN (sd 0.59–0.69), 4.03 for Chinese BiLSTM (sd 0.30–0.58) and 3.93 for Chinese TextCNN — and should be described as a clear difference, not a marginal one.**

**Table 6b. Test-set out-of-vocabulary rate, for both the full vocabulary and the vocabulary the model actually uses.** The vocabulary is built from the training split only; a token unseen in training is out-of-vocabulary at test time. `Full` counts every training token; `model` applies the 10,000-entry cap imposed by the implementation, which binds only where the full vocabulary exceeds it. The final two columns are the rate of the translated test set against a localized-trained vocabulary.

| Locale | Train vocab from | Test set | Tokenizer | Full vocab | OOV (full) | Model vocab | **OOV (model)** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| en-US | localized | localized | whitespace | 5,302 | 3.51% | 5,302 | 3.51% |
| en-US | localized | localized | character | 32 | 0.00% | 32 | 0.00% |
| ko-KR | localized | localized | whitespace | 9,921 | 11.35% | 9,921 | 11.35% |
| ko-KR | localized | localized | character | 1,037 | 0.14% | 1,037 | 0.14% |
| **ko-KR** | **localized** | **translated** | whitespace | 9,921 | **37.99%** | 9,921 | **37.99%** |
| **ko-KR** | **localized** | **translated** | character | 1,037 | **3.16%** | 1,037 | **3.16%** |
| ko-KR | translated | translated | whitespace | 11,740 | 13.90% | 9,998 | 15.50% |
| zh-CN | localized | localized | whitespace | 11,142 | 89.20% | 9,998 | **90.07%** |
| zh-CN | localized | localized | character | 2,194 | 0.40% | 2,194 | 0.40% |
| **zh-CN** | **localized** | **translated** | whitespace | 11,142 | 98.09% | 9,998 | **98.26%** |
| **zh-CN** | **localized** | **translated** | character | 2,194 | **4.75%** | 2,194 | **4.75%** |

**The character-level rates are 3.16% (Korean) and 4.75% (Chinese), not zero.** We state the figures rather than describing character segmentation as eliminating vocabulary mismatch: it reduces it by an order of magnitude, which is what makes the two corpora comparable, but it does not remove it. **The Chinese whitespace column is the mechanism behind the broken condition.** Under whitespace tokenization, **89.2% of Chinese test tokens are unseen during training** — and this is a *localized* test set against a *localized* training split, so the defect exists before any machine translation enters the picture. Against the translated test set it rises to 98.3%. **This table gives the mechanism behind the Chinese result.** Under whitespace tokenization, **89.2% of Chinese test tokens are unseen during training and 90.25% of test utterances consist *entirely* of unseen tokens.** For nine utterances in ten the model is asked to classify a sequence it has effectively never encountered, and it can do no better than emit a constant prediction — which is what the 10.31% and 7.03% figures record, and why the Chinese BiLSTM returns the identical value of 7.03% under both the localized and the machine-translated test sets (Section 4.2).

 Character-level segmentation reduces that to 0.40% and 0.00% respectively. **The comparison is not "Chinese is hard for these models"; it is "the models were tested on strings they had never seen."** We measured the test-set OOV rates directly in order to test that mechanism instead of asserting it, because our earlier phrasing — that there was "no textual signal" — stated the conclusion without demonstrating it.

 The other two locales behave as expected under whitespace and provide the control: 3.51% and 11.35% OOV token rates with under 1% of utterances fully out-of-vocabulary. Those are ordinary levels of lexical novelty, and they are consistent with the 80.55% and 73.83% accuracies those conditions achieve. Two things follow. First, this is why the tokenizer-coverage and pretraining-language explanations for the ALBERT result cannot be separated by our design (Section 4.1). Second, it shows that **subword tokenization is not automatically script-appropriate**: a subword tokenizer trained on one language can be as unsuitable for another script as whitespace splitting is for Chinese. The relevant property is not "subword versus whitespace" but "was the vocabulary induced from data in this script".

 **Korean turns the ablation into a graded ladder — but only once the readout is corrected.** With the padding mask in place, both architectures improve as segmentation is made finer, and neither morphology nor characters are a marginal tweak:

| Model | whitespace (masked) | morphological (Okt) | character | steps |
| --- | --- | --- | --- | --- |
| TextCNN | 73.83 ± 0.76 | 78.66 ± 0.69 | 80.80 ± 0.59 | +4.83, +2.14 |
| BiLSTM | **73.36 ± 1.07** | **77.48 ± 0.45** | **77.59 ± 0.43** | **+4.12, +0.11** |

**The defensible claim is `whitespace < {morphological, character}`, not a three-step ladder.** The whitespace-to-morphological step is large under both architectures (+4.83 and +4.12) and **far exceeds the seed standard deviations** (0.69–1.07).

**The morphological-to-character step is not the same kind of result**: for TextCNN it is +2.14, roughly three times the standard deviation, but **for the recurrent model it is +0.11 against a standard deviation of 0.43** — indistinguishable. We therefore claim a two-level ordering, not monotonic improvement, and we do not use the word "monotonic".

**This is a correction to our own analysis, and it changes the conclusion.** The uncorrected ladder for Korean BiLSTM is **44.48 → 57.80 → 67.61** under whitespace, morphological and character segmentation, and **its three values mix seed counts** (the first is the five-seed unmasked mean completed for this paper, the other two are three-seed values from the original ablation), so it is reported only as a description of the defective configuration and never as a result.

**The corrected readout reverses the picture**: the masked whitespace baseline alone (73.36) exceeds every unmasked morphological and character value. Fine-grained segmentation shortens the padding, which partially repairs the readout defect by accident, so the uncorrected ladder measured two things at once and separated neither. The values in Table 6 are the corrected ones, and no conclusion in this paper rests on the uncorrected column. The ordering survives for both architectures, which rules out the possibility that the Chinese result is a two-point comparison that happened to land favourably. Korean morphological segmentation differs from whitespace splitting on **66% of utterances** (measured over an 800-utterance sample; mean tokens 4.91 → 6.15, an increase of 25%), and that finer granularity buys a consistent and measurable gain.

**Segmentation quality is not a binary property of a pipeline; where it matters, it is a graded one — but the grading stops below character level for this architecture.**

We also measured the agreement directly in order to check the opposite possibility — that the Okt condition would silently reproduce the whitespace condition and contribute nothing. It does not: the two disagree on two-thirds of utterances, and the accuracy difference is in the direction and of the magnitude the finer segmentation predicts. Had the agreement been near-total, we would have reported the column as a no-op, not as a result. **Two causes, not one.** Our initial framing attributed the weakness of the non-pretrained models on Chinese to tokenization. A pilot on the full splits showed that this was only half the story, and the completed runs separate the two causes cleanly.

 *Factor 1 — tokenization (isolated by TextCNN).* TextCNN reaches **80.55 ± 0.64** on English and **73.83 ± 0.76** on Korean, both of which delimit words with spaces, but only **10.31** on Chinese, where whitespace splitting collapses 97.6% of utterances to a single token (Table 2). Because TextCNN pools globally over positions it is insensitive to how the remainder of the sequence is padded, so this gap is attributable to segmentation rather than to the model's capacity to represent position.

 *Factor 2 — a padding artefact in the recurrent baseline (isolated by BiLSTM ± mask).* The unmasked BiLSTM, which takes the final timestep as the sentence representation, does not merely underperform: it is **unstable across seeds** on both English (39.06 ± 20.37, individual seeds spanning 7.03 to 57.70) and Korean (44.48 ± 17.31). Selecting the last non-padding timestep instead raises English to **79.97 ± 0.70** and Korean to **73.36 ± 1.07**, reducing the standard deviation by a factor of **29.1 and 16.2** respectively. **Both columns are at n = 5, so the comparison is like for like.**

**The double dissociation is the result.** The two factors do not merely coexist; they bind on different languages with very different strength, and the pattern is diagnostic:

 | Locale | Unmasked | Masked | Gain from correcting padding |
| --- | --- | --- | --- |
| en-US | 39.06 | 79.97 | +40.9 |
| ko-KR | 44.48 | 73.36 | +28.9 |
| zh-CN | 7.03 | 10.63 | +3.6 |

Correcting the padding artefact recovers English and Korean almost completely — the masked BiLSTM lands within 0.6 points of TextCNN in both (79.97 vs 80.55; 73.36 vs 73.83), so for these two languages the padding defect was the binding constraint. For Chinese the same correction moves accuracy by under four points, leaving the model near chance. **The constraint that binds on Chinese is therefore segmentation, not padding** — which is what makes the `character` and `morphological` columns of Table 6 the test of the segmentation account.

 **The shape of the failure differs, and this is itself informative.** Chinese is *stably* broken: all three seeds return exactly 7.03%, the majority-class rate, with zero variance. English is *unstably* broken: it ranges from 7.03 to 57.70 across seeds. A model receiving one token per sentence has, literally, no signal to condition on and deterministically emits the prior. A model receiving adequate tokens but a corrupted readout retains signal that initialisation-dependent training may or may not exploit. The variance therefore carries mechanistic information rather than being noise to be averaged away. The dissociation is shown in Figure 5: masking the padding position repairs English and Korean but leaves Chinese where it was.

 **Consequence for our earlier work — stated explicitly.** Our previous study reported the Chinese result alone and attributed it to tokenization. The present analysis shows that this conflated two independent defects. We therefore (i) report both BiLSTM conditions rather than the more favourable one, (ii) treat TextCNN as the tokenization-controlled comparison and BiLSTM ± mask as the padding-controlled comparison, and (iii) do not merge them into a single claim that "Chinese fails".

 **The prediction we specified in advance is confirmed, and the effect is large.** Before running the ablation we stated that if character-level segmentation raised Chinese TextCNN substantially above its whitespace value of 10.31, the segmentation account would be supported. Chinese TextCNN reaches **80.62 ± 0.46** under character-level segmentation — a gain of **+70.3 points** over the whitespace value of 10.31, from near chance to a range comparable with English (76.42 ± 0.56) and Korean (80.80 ± 0.59) under the same scheme, and stable across all five seeds. **The recurrent model shows the same effect once its readout is corrected**: Chinese BiLSTM rises from 10.63 (masked whitespace) to **79.69 ± 0.30** under character-level segmentation, a gain of 69.1 points over the masked whitespace baseline of 10.63. We report the masked pair here; the unmasked values appear in Table 6 and are not used as evidence (Section 4.3). We do not claim the three locales are statistically equivalent, and the spread argues against it. Under character-level segmentation the maximum cross-locale spread is **4.38 points**, from English (76.42 ± 0.56) to Korean (80.80 ± 0.59), compared with **70.24 points** under whitespace segmentation. What the data support is a substantially narrower cross-locale range, not statistical equivalence. We flag this explicitly because "the three languages are equivalent" is the kind of claim a reader may infer from a bar chart, and it is not what we measured.

 This result carries two consequences. *The Chinese failure is a preprocessing choice, not a property of the language or the architecture.* The architecture, the training data, the optimiser, the number of epochs, and the seeds are unchanged; one line of tokenization is. A model that cannot exceed 10.31% is not one that has been shown to lack the capacity to model Chinese — it is one that was never given Chinese to model. We therefore recommend that reports of non-pretrained models performing near chance on Chinese be read, by default, as a tokenization defect until shown otherwise. *Appropriate tokenization makes the three languages comparable.* Under character-level segmentation the three locales land at 76.42, 80.80, and 80.62 — a spread of 4.4 points — whereas under whitespace they span 10.31 to 80.55. The cross-lingual gap that motivated this study was, to a first approximation, an artefact of applying an English-appropriate convention to a language for which it does not hold.

 **Tokenisation is not uniformly beneficial — the effect is language-specific.** If character-level segmentation simply helped everywhere, it would be a trivial recommendation. It does not:

 | Locale | TextCNN, whitespace | TextCNN, character | Change |
| --- | --- | --- | --- |
| en-US | 80.55 | 76.42 | −4.1 |
| ko-KR | 73.83 | 80.80 | +7.0 |
| zh-CN | 10.31 | 80.62 | +70.3 |

Character-level segmentation *reduces* English TextCNN accuracy, where it discards word units that carry the signal; it improves Korean modestly; and it is responsible for nearly the whole Chinese recovery, where it restores units that whitespace splitting had destroyed. This three-way interaction is why the recommendation must be language-conditional, and it also argues against the simplest explanation of the Chinese result — that character tokenization is merely an easier task in general. If it were, English would improve too. **That English reduction must be read with a caveat we can quantify.** The non-pretrained models share a fixed maximum sequence length of 30 positions, which is ample for whitespace tokens (English utterances average 6.9 words, 95th percentile 13) but not for characters: English utterances average 34.9 characters with a 95th percentile of 65, so **55.2% of English test utterances are truncated at character level**, against 5.1% at a length of 64 and 4.0% for Korean and 0.5% for Chinese at length 30. The English character-level condition therefore conflates segmentation with truncation, and we do not use the magnitude of the English reduction as a clean estimate of a tokenization effect. The *direction* is unambiguous — truncation cannot help — but the size is not interpretable from this design, and separating the two would require re-running the non-pretrained models with a locale-specific maximum length. We report the reduction because it is what the pipeline produced, and flag the confound rather than the number. The three segmentation schemes available to the non-pretrained models are plotted in Figure 6, where the language × tokenization interaction is visible as the reversal in the English panel.

 **An incidental third instance of the padding signature.** Character-level English also removes the padding defect, without our intending it: English utterances average 34.9 characters, so character-level sequences exceed the fixed length of 30 and are truncated rather than padded, with the result that the final timestep is a real token. English BiLSTM therefore moves from 39.06 ± 20.37 under whitespace to **67.69 ± 0.26** under character-level segmentation — again a substantial accuracy gain accompanied by a collapse in variance. Together with the English masking ablation (variance 20.37 → 0.70) and the Korean masking ablation (17.31 → 1.07), this is the third instance of the same signature. We regard the repetition as supportive of the mechanism proposed in Section 3.2; we do not claim it as proof, since all three instances share the same dataset and the same two architectures.

 **Any segmentation that respects Chinese boundaries works; the specific scheme matters less than the choice to stop splitting on spaces.** The full Chinese TextCNN ladder:

| Tokenizer | Chinese TextCNN | Change vs whitespace |
| --- | --- | --- |
| whitespace | 10.31 ± 0.22 | — |
| jieba (morphological) | 76.69 ± 0.57 | +66.4 |
| character | 80.62 ± 0.46 | +70.3 |

Both linguistically motivated schemes recover the task almost completely, and they differ from each other by only 3.7 points. This strengthens the interpretation: the failing condition is not "insufficiently sophisticated segmentation" but "no segmentation at all". A practitioner who cannot adopt a Chinese word segmenter can still use character splitting and recover most of the performance.

 The recurrent model shows the same ordering, and **the corrected values are 75.66 ± 0.58 (`jieba`) against 79.69 ± 0.30 (characters)** — a **4.03-point** gap, or roughly seven times the standard deviation of the character condition, and therefore a clear difference rather than a marginal one. `jieba` produces about 5–6 tokens per Chinese utterance against roughly 10.5 characters, so the `jieba` condition remains the more coarsely segmented of the two, and the direction of the gap is what that granularity difference predicts. **The uncorrected unmasked comparison is 42.24 against 69.96 with a per-seed standard deviation of 18.77 for `jieba`** — the widest spread anywhere in this study, and the signature of a partly corrupted readout rather than of a segmentation effect. 
---

*The train–test overlap table and the accuracy on duplicated versus non-duplicated entries (Tables 6b-ii and 6b-iii) are in Supplementary Section S6. In brief: **7.5% of the Chinese test entries occur verbatim in the training split**, they are classified correctly **83.1%** of the time, they supply **60%** of every correct prediction the Chinese whitespace model makes, and removing them drops accuracy from 10.31% to **4.48%** — below the majority-class rate of 7.67%.*


---
