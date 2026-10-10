# Supplementary Material

**What Does a Machine-Translated Test Set Measure? Provenance Effects in Multilingual Intent Classification**

This file carries the material moved out of the main text during revision. Section and table numbers refer to the main text unless stated otherwise.

---

## S1. Efficiency measurements (formerly Section 4.5)

### What these numbers can be compared across

**This section carries the measurements. The limitations that attach to them — differing hardware, differing training budgets, and non-reproducible wall-clock time — are in Section S10.**

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
| BiLSTM (padding-masked) | en | whitespace | 5,304 | 0.78 | 2.98 | 0.23 | n/a (CPU) |
| BiLSTM (padding-masked) | ko | character | 1,037 | **0.35** | 1.35 | 0.31 | n/a (CPU) |
| BiLSTM (padding-masked) | zh | whitespace | 10,000 (capped) | **1.25** | 4.77 | 0.36 | n/a (CPU) |
| ALBERT (albert-base-v2, English-only) | ko | subword | — | 11.7 | 45 | 7.97 | 218 |
| ALBERT (albert-base-v2, English-only) | zh | subword | — | 11.7 | 45 | 7.90 | 219 |
| BERT (bert-base-chinese) | zh | subword | — | 102.3 | 390 | 7.21 | 873 |
| DistilBERT (multilingual) | zh | subword | — | 135.4 | 516 | 3.62 | 1120 |
| mBERT (bert-base-multilingual-cased) | ko, zh | subword | — | 177.9 | 679 | 7.21 | 1448 |
| XLM-R (xlm-roberta-base) | ko, zh | subword | — | 278.1 | 1061 | 6.35 | 2212 |

*The non-pretrained models were trained and evaluated on CPU and therefore have no GPU peak figure. Their parameter count follows `100 × vocabulary + 138,360`, so **the same architecture spans a factor of eight across locales and tokenizers (0.14 M to 1.14 M)**, and the whitespace rows are the largest in every locale. **A single pooled value of 0.67 M for TextCNN "en, ko, zh" would be the English whitespace figure**; it understates the Korean and Chinese whitespace conditions by roughly 70% and overstates the character conditions by nearly fivefold.*

*The vocabulary sizes in this table are the sizes the **model** uses, and they exceed the training-corpus counts in Table 6b **by two** wherever the corpus vocabulary is below the cap: the embedding matrix carries one slot for a padding symbol and one for an unknown-word symbol in addition to the induced types. **The cap is a separate effect and does not explain the +2.** It binds on three rows only — `zh` whitespace in TextCNN (11,142 corpus types, capped at 10,000) and `ko`/`zh` translated-training vocabularies in Table 6b — and where it binds it *reduces* the corpus count rather than adding to it. The character rows carry the same +2 as the whitespace rows (English 32 → 34, Korean 1,037 → 1,039, Chinese 2,194 → 2,196) because they are below the cap, and the two tables agree on those rows after the two special symbols are accounted for. Table 6b reports corpus counts, this table reports embedding dimensions.*

*Correction, recorded here rather than made silently. The three `BiLSTM (padding-masked)` rows above were previously printed as 1.42 M / 1.98 M / 2.60 M parameters and 5.42 / 7.55 / 9.92 MB. **Those figures are not reproduced by any released run record.** All 40 released per-run `results_v2.csv` files (the individual run-directory records, merged into `canonical_runs.csv`) report **781,340** parameters for English whitespace, **354,840** for Korean character and **1,250,940** for Chinese whitespace, and the block follows `100 × vocabulary + 250,940` exactly, with the vocabulary column above. We print the measured values; the earlier figures were also non-monotone in vocabulary size, which the measured ones are not. The `TextCNN` block is unaffected and was verified the same way.*

**Latency figures are also not comparable across the two blocks.** TextCNN's 0.16 ms and BiLSTM's 0.23 ms were measured on CPU; every other row was measured on a Tesla T4. A ratio between them would quantify the hardware, not the models, and we do not report one. Within the GPU block the latencies are directly comparable, and there the ordering is clean: **DistilBERT is the fastest pretrained model at 3.62 ms/sample — 1.99× faster than BERT (7.21 ms) and 1.75× faster than the next-fastest, XLM-R (6.35 ms)** — despite having more parameters than BERT, because the multilingual vocabulary inflates its size but not its per-token compute.

 **The comparison that is valid and that matters for a device is parameter count and memory, not latency across hardware.** The smallest non-pretrained configuration in this study is character-level TextCNN on English at **0.14 M parameters and 0.55 MB**; the largest non-pretrained configuration is whitespace TextCNN on Chinese at 1.14 M and 4.34 MB; BERT requires 102.3 M and 390 MB; XLM-R requires 278.1 M and 1,061 MB with a peak of 2,212 MB during evaluation. That is a range of **three** orders of magnitude in memory, and memory is the binding constraint on an embedded smart-home device. Character-level TextCNN on Korean — 0.24 M parameters, 0.92 MB — reaches 80.80 % against 84.55 % for Korean BERT in Table 4, giving up 3.75 points for a footprint roughly three orders of magnitude smaller. **We therefore frame the recommendation as a memory trade rather than as a speed ratio**, and we flag the CPU/GPU split explicitly so that no reader computes a cross-hardware speed-up from this table.

 **The character-level conditions are also the cheapest, which strengthens the tokenization recommendation rather than trading against it.** Choosing character segmentation reduces TextCNN's parameter count from 1.13 M to 0.24 M in Korean and from 1.14 M to 0.36 M in Chinese — a reduction of 68–79% — while *improving* accuracy. On this evidence there is no accuracy-for-memory trade in the tokenization decision at all: the script-appropriate scheme is both more accurate and substantially smaller. **What the table does settle is that parameter count is not a proxy for cost.** Taking the Chinese condition, where the relevant inference measurements are available, DistilBERT carries *more* parameters than BERT (135.4M against 102.3M) yet has substantially lower inference latency (3.62 against 7.21 ms per sample), because its parameters are concentrated in a large embedding matrix rather than in layers. ALBERT has an order of magnitude fewer parameters than either (11.7M, 45 MB, 218 MB peak) and is nevertheless **slower than BERT** at inference on the same condition (7.90 against 7.21 ms per sample) — the same inversion seen for training time in Table 8a. A table sorted by parameter count would rank all three of these models wrongly.

 Under matched training conditions DistilBERT also trains substantially faster than BERT, in English (417 against 806 s) and in Korean (469 against 861 s). The ordering is checkpoint-dependent: in English, DistilBERT uses a 66.4M-parameter monolingual checkpoint against BERT's 110.1M, while the parameter-count inversion described above occurs only for the Chinese checkpoint pairing used here. **A reporting caveat this table makes visible.** Parameter count is a property of the *checkpoint*, not of the model family: the multilingual DistilBERT vocabulary (119,547 subword types) inflates its embedding matrix far beyond the 66M of its English counterpart. A table aggregating by architecture name across languages can therefore state the opposite of the truth, which is why we report checkpoint, size, and locale together.



---

## S2. Zero-shot cross-lingual transfer and the audit of `bert-base-chinese` (formerly Section 4.4)

**This section carries the full protocol, both tables and the four audit checks. A one-paragraph summary is in Section S11.**


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
| *majority class* (`calendar_set`) | 7.03 | 7.03 | 7.03 |

*Both reference rows describe the **full test split**, which is the basis on which every row of this table is evaluated. The most frequent gold class is `calendar_set`, 209 of 2,974 entries; because the three test sets are parallel renderings of the same utterances the labels align exactly and the share is identical in all three locales. An earlier version printed per-subset values (7.67 / 7.70 / 7.67) in this row.*

**Check 1 — which file the loader actually opened.** The audit recorded, per locale, the absolute path each `load_data` call resolved to, an MD5 of that file, and the first training sentence it contained. The English run opened `/kaggle/working/massive/1.1/data/en-US.jsonl`, MD5 `135c741954f86f23d37784dba247c78a` — the same file and the same checksum recorded in Section S7 — and its first training sentence is **"wake me up at nine am on friday"**. **0 of its 403,436 characters are CJK**, against 100% of training sentences in the Korean and Chinese files. **A mis-resolved path is excluded.**

**Check 2 — the random-initialisation control.** This is the decisive one, because **a pipeline that leaked labels or test text would raise every model that reads it, pretrained or not.** We therefore trained the *same architecture with no pretrained weights at all* through an identical pipeline. It reaches **77.74 ± 0.91 on English**, confirming that the training loop works. On Chinese it reaches **4.34 ± 0.20** — **below the 7.67% majority-class rate and 2.7 points above chance**, and on Korean 5.14 ± 2.10. **The fine-tuned model reaches 65.78.** A 61-point separation between a randomly initialised encoder and a pretrained one, on the same data through the same code, cannot be produced by leakage: whatever the fine-tuned model is using, it is something that pretraining put there.

**Check 3 — where the 65.78% comes from.** Accuracy is broad rather than concentrated: **37 of the 59 intents exceed 50% and 18 exceed 80%**, and the strong intents span the domains — `qa_currency` 100%, `iot_coffee` 97%, `iot_wemo_off` 94%, `iot_cleaning` 92%, `email_sendemail` 91%, `weather_query` 90%, `transport_taxi` 88%, `cooking_recipe` 88%. **A leakage artefact would concentrate accuracy on the few intents whose test items happen to be in the training set**; this pattern does not. The model predicts **54 of the 60 classes**, and its single most frequent prediction accounts for only **7.1%** of test items — again the opposite of the collapsed readout a leak produces.

**Check 4 — digits and Latin characters.** Shared non-Chinese symbols are an obvious alternative explanation: a model that cannot read Chinese might still key on numerals and Latin tokens. We therefore re-evaluated the Chinese test set with every digit and Latin character removed. Accuracy is **66.48 / 63.69 / 66.71** across the three seeds, against **66.75 / 63.82 / 66.78** on the unmodified text — **a change of at most 0.3 points.** Only 2.1% of test sentences contain such characters, and accuracy on those is 65.08% against 65.80% on the rest. **The effect does not run through shared symbols.**

**What the audit does not explain, and we say so plainly.** It establishes that the result is **real, reproducible and not a data-handling artefact**; it does not establish **why** a model fine-tuned only on English represents Chinese so much better than mBERT does. Three candidate accounts remain, and our design separates none of them: (i) `bert-base-chinese` has a Chinese-specialised vocabulary, though we measured the two tokenizers to give essentially identical token counts on Chinese test text (12.31 tokens per utterance for both; `[UNK]` 0.00% against 0.08%), which weakens this account; (ii) English fine-tuning degrades mBERT's cross-lingual alignment more than it degrades the Chinese representations of a monolingual model; (iii) some property of the monolingual checkpoint's pretraining corpus that our controls do not isolate. **We report the measurement and flag the mechanism as open** (Section 7.2) rather than presenting it as understood.

**The audit also supplies the checksums this study previously lacked.** The three MASSIVE files used were `en-US.jsonl` (MD5 `135c741954f86f23d37784dba247c78a`), `ko-KR.jsonl` (`18f2b556dde5b8ac77ace58d0f2a6c1d`) and `zh-CN.jsonl` (`2d2a3fb725b3a37793e2dcd16442b1e0`), each with 11,514 training, 2,033 development and 2,974 test entries over the same 60 intents. The original runs did not record these, and one consequence of that omission is that the audit had to be run at all; **Section S7 states the gap and what we did about it.**

**XLM-R dominates mBERT by a wide and consistent margin.** The gap is 40.8 points on Korean (70.47 against 29.67) and 25.4 points on Chinese (77.05 against 51.66), and it is not a matter of seed noise: XLM-R's standard deviation is 0.94 and 2.02, against mBERT's 3.81 and 4.05. mBERT is both worse and less stable when transferred.

 **DistilBERT is the weakest arm on both locales** (16.58 Korean, 46.58 Chinese), below mBERT despite sharing its tokenizer family. This is worth recording because DistilBERT is the efficiency recommendation in Section S10: **the model we recommend for cost is the one we would not recommend for zero-shot transfer**, and a practitioner choosing on footprint alone would get this wrong. **The size of the XLM-R margin is the practically relevant finding.** For Korean, zero-shot XLM-R (70.47) lands within 3.4 points of in-language supervised TextCNN (73.83) and within 11.3 of in-language mBERT (81.73). That is an unusually favourable position for a model that has never seen a Korean training example, and it means that for a new locale the first question is not whether to collect labelled data but whether XLM-R with no target-language training already clears the deployment bar.

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

**Table 5h. Deployment overstatement under a matched versus a mismatched translation system** (character-level segmentation, fixed evaluation subset, **three seeds — a separate run from Tables 5a–5c, which use five**, 2,000-resample bootstrap intervals over test sentences).

| Locale | Model | MT→MT (same system) | MT→MT2 (cross system) | MT→L | Deployment overstatement, same system | Deployment overstatement, cross system | Reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ko-KR | TextCNN | 75.37 | 73.90 | 63.80 | **+11.57** [+10.13, +13.09] | **+10.11** [+8.67, +11.56] | **−1.47** [−2.57, −0.39] |
| ko-KR | BiLSTM | 70.26 | 68.90 | 58.94 | **+11.32** [+9.73, +12.87] | **+9.95** [+8.45, +11.48] | **−1.36** [−2.49, −0.17] |
| zh-CN | TextCNN | 77.20 | 76.53 | 69.87 | **+7.32** [+5.92, +8.83] | **+6.66** [+5.34, +7.99] | **−0.66** [−1.65, +0.32] |
| zh-CN | BiLSTM | 74.76 | 73.61 | 67.59 | **+7.17** [+5.78, +8.59] | **+6.02** [+4.67, +7.39] | **−1.15** [−2.21, −0.09] |

*Seed count. This table uses **three** seeds because the cross-system arm requires the second translation system, which we ran for three seeds; Tables 5a, 5b and 5m use **five**. The two seed sets agree closely on the same-system quantity — Korean TextCNN is +11.57 here and +11.48 in Table 5a, Chinese TextCNN +7.32 here and +7.52 there — but because the two values are computed from different seed sets they are not interchangeable, and every citation of a same-system overstatement figure in the main text states which table it comes from.*

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





**Table A1. Pretrained checkpoint assignment and model size.** Checkpoint identifiers are the HuggingFace names actually used for that locale. Parameter and size figures are **measured** from the released per-run `results_v2.csv` records (one file per run directory, merged into `canonical_runs.csv`) except where marked *(nominal)*. The efficiency instrumentation was added after the English and Korean pretrained runs of Table 4 had completed (Section 3.6), so the BERT entries for those two locales are published checkpoint specifications rather than measurements taken by their own runs: `bert-base-uncased` is 110.1 M (en, nominal) and `bert-base-multilingual-cased` at 177.9 M is (ko, nominal) here. The same checkpoint *is* instrumented elsewhere in this study — the `mBERT` row below and the Korean mBERT evaluations of Table 8b — so 177.9 M is a real measurement of that checkpoint, but not of the Korean pretrained run of Table 4, and we label the cell according to the run it describes.

 | Model | en-US | ko-KR | zh-CN | Params (M) | Size (MB) |
| --- | --- | --- | --- | --- | --- |
| BERT | bert-base-uncased | bert-base-multilingual-cased | bert-base-chinese | 102.3 (zh, measured) / 110.1 (en, nominal) / 177.9 (ko, nominal) | 390 / 418 / 679 |
| DistilBERT | distilbert-base-uncased | distilbert-base-multilingual-cased | distilbert-base-multilingual-cased | 66.4 (en, nominal) / 135.4 (ko, zh, measured) | 255 / 516 |
| ALBERT | albert-base-v2 | albert-base-v2 | albert-base-v2 | 11.7 (ko, zh, measured) | 45 |
| mBERT | bert-base-multilingual-cased | bert-base-multilingual-cased | bert-base-multilingual-cased | 177.9 (measured) | 679 |
| XLM-R | xlm-roberta-base | xlm-roberta-base | xlm-roberta-base | 278.1 (measured) | 1,061 |

**The pattern the table exposes** is why we report size per checkpoint and per locale rather than per architecture: **the same architecture name can differ in size by a factor of two depending on which locale's weights are used**, and DistilBERT is the clearest case — 66.4M in English against 135.4M for the multilingual checkpoint.

 ALBERT's uniform 11.7M makes it the smallest model in every locale, which rules out capacity as an explanation for its performance profile: it is small everywhere, and yet it scores 85.12 in English against 8.80 in Korean and 7.50 in Chinese (Table 4). **What that profile does not establish is *why*.** Section 4.1 shows the collapse is confounded between monolingual pretraining and English-only vocabulary coverage — `albert-base-v2` leaves 40.65% of Korean and 24.75% of Chinese test tokens as `[UNK]` (Table 6c) — and our design does not separate the two. We therefore report the ALBERT row as a combined demonstration and draw no conclusion about which cause dominates.

 Each run emits a manifest recording: start and finish timestamps, the full command-line arguments, device and GPU model, library versions (Python / PyTorch / Transformers / NumPy / pandas), and the per-run rows (seed, sample counts, sorted class list, epochs, accuracy, F1, parameter count, size, latency, peak GPU memory) in the accompanying per-run `results_v2.csv` (one file per run directory; the set is merged into `canonical_runs.csv`). **One gap, stated plainly: the manifests for the original runs do not record a checksum of the data files they read.** They contain the argument list but not the identity of the corpus those arguments resolved to, so the training file used by any given run cannot be verified after the fact. This mattered for exactly one result — the Chinese zero-shot `bert-base-chinese` arm of Section 4.4, whose score we could not explain and therefore could not initially rule out as a pipeline artefact. **We closed the gap for that result by re-running it.** The audit of Section 4.4 recorded, per locale:

| File | MD5 | train / dev / test | Intents |
| --- | --- | --- | --- |
| `en-US.jsonl` | `135c741954f86f23d37784dba247c78a` | 11,514 / 2,033 / 2,974 | 60 |
| `ko-KR.jsonl` | `18f2b556dde5b8ac77ace58d0f2a6c1d` | 11,514 / 2,033 / 2,974 | 60 |
| `zh-CN.jsonl` | `2d2a3fb725b3a37793e2dcd16442b1e0` | 11,514 / 2,033 / 2,974 | 60 |

**All other runs in this study remain without recorded checksums**, and a reader who wishes to verify them must rely on the argument lists and on the reproduction evidence reported here and in Section 3.6. **That evidence consists of fifteen configurations re-run from scratch on a second machine, comparing accuracies to five decimal places; the largest absolute spread across those fifteen was 0.000050 (i.e. 0.005 percentage points, or five parts in 100,000).** **Any re-run should record an MD5 of every input file alongside the arguments**, and we would not repeat the omission. Code and result CSVs are released with the paper.

**The development split is never read.** MASSIVE ships 11,514 training, 2,033 development
and 2,974 test entries per locale (Table 1), and the corpus generator translated all three splits,
so a machine-translated development set exists (2,033 entries per locale under `massive_mt/`). No
experiment uses it. The loader in `experiment_v2.py` selects only the `train` and `test`
partitions, so there is no model selection, no early stopping and no epoch selection on a held-out
split, and **every accuracy reported in this paper is the test-split accuracy of the last training
epoch** at a fixed budget (15 epochs for the non-pretrained models, 3 for the pretrained encoders).
Section 6's "limited hyperparameter search" therefore refers to settings fixed outside the released
training loop, not to a search on this split.

**Configuration of the non-pretrained models.** Read from `experiment_v2.py` rather than restated
from prose, so that every value can be checked against the released code.

| Setting | TextCNN | BiLSTM (padding-masked) |
| --- | --- | --- |
| Embedding dimension | 100 | 100 |
| Vocabulary | induced from the training split of the condition, capped at 10,000 entries **including** `<PAD>` = 0 and `<UNK>` = 1 (so at most 9,998 induced types) | same |
| Unknown words | index 1 | index 1 |
| Maximum length | 30 tokens, truncated then right-padded with 0 | 30 tokens, truncated then right-padded with 0 |
| Positions | kernel sizes (3, 4, 5), 100 feature maps each | one bidirectional layer, hidden size 128 per direction (256 concatenated) |
| Readout | global max-pooling over positions, concatenated | last **non-padding** timestep of each direction, concatenated |
| Dropout | 0.5, on the pooled vector, before the classifier | 0.5, on the readout, before the classifier |
| Classifier | linear, 300 -> 60 | linear, 256 -> 60 |
| Optimiser, learning rate, batch | Adam, 1e-3, 32 | Adam, 1e-3, 32 |
| Epochs, loss | 15, unweighted cross-entropy | 15, unweighted cross-entropy |
| Scheduler, early stopping | none | none |
| Whitespace tokenization | lower-case, split on whitespace | same |
| Character tokenization | spaces removed, then split into characters | same |
| Morphological tokenization | `jieba` (Chinese) / Okt (Korean) | same |

*One exception, and it is stated in Section S12 as well: the English character-level cells of
Table 6 were re-run with the maximum length raised from 30 to 64, because 54.2% of English
character-level test utterances are truncated at 30 — lengths are counted as Unicode characters of
the MASSIVE `utt` field, spaces included. Every other cell in this paper uses the
settings above. The pretrained encoders use a maximum length of 128, batch 16, AdamW with 10%
linear warmup and gradient clipping at 1.0, 3 epochs, and a learning rate of 2e-5 (1e-5 for the
XLM-R family).*

**Translation: decoding settings, precision and hardware.** Read from `make_massive_mt.py`, the
script that generated every translated corpus.

| Setting | Value |
| --- | --- |
| Primary system | `facebook/nllb-200-distilled-600M`, all 16,521 `en-US` utterances (train, development and test) |
| Second system | `facebook/nllb-200-3.3B`, **test split only** (2,974 entries per locale) |
| Third lineage | the DeepL REST API, test split only (cross-lineage check of Section 4.2.2b) |
| Decoding | **greedy**: `generate(...)` is called with only `forced_bos_token_id` and `max_new_tokens`, so no beam search and no sampling parameter is set and the library default (greedy) applies |
| Maximum new tokens | 128 |
| Source-side truncation | 128 tokens, batch-padded |
| Precision | **fp16 on CUDA**, fp32 on CPU |
| Batch size | 16 |
| Target language | forced with the official NLLB code (`kor_Hang`, `zho_Hans`, `deu_Latn`, `vie_Latn`, `jpn_Jpan`) |
| Hardware, 3.3B run | a Kaggle GPU session with a **single Tesla T4 (16 GB)**, fp16 |
| Hardware, 600M run | `[not recorded]` - the script prints the device and the memory footprint, but no log of that run survives |
| Upstream checkpoint revision | `[not recorded]` - `from_pretrained` is called without a `revision` argument, and the corpora record only the model name in their `mt_source` field |

**What this means for the regeneration claim.** Section 3.1 says the corpora can be regenerated
"bit-for-bit". That holds for the pipeline - the script, its arguments and the greedy sampling
regime are all released and deterministic - but not for the upstream weights: because no checkpoint
revision is pinned, re-running the released script against a later revision of the same NLLB
checkpoints need not reproduce the same text. We state the unpinned input rather than claiming a
determinism we did not record.

**Table A2. Per-seed accuracies behind the fixed evaluation subset 2x2 (Tables 5a and 5b).**
Rendered from the released `canonical_numbers.json` by the released `make_perseed_table.py`; no
value here is transcribed by hand. `L→L` = localized train / localized test; `L→MT` = localized
train / translated test; `MT→MT` = translated train / translated test; `MT→L` = translated
train / localized test. "char + mask" is the padding-masked recurrent readout. The rows marked
`morph` are the two-seed morphological TextCNN cells of the fixed subset. The full-test-split
per-seed values behind Tables 4 and 6 are the `accuracy` column of the released per-run
`results_v2.csv` records (one such file per run directory, merged into `canonical_runs.csv`), one row per configuration and seed.


| Provenance cell | Locale | Model | Tokenizer | n | Per-seed accuracy (%) | Mean | SD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L→L | ko-KR | BiLSTM | char + mask | 5 | 76.37, 75.69, 76.03, 75.13, 75.66 | 75.78 | 0.46 |
| L→L | ko-KR | BiLSTM | morph + mask | 5 | 76, 75.28, 76.07, 75.43, 75.17 | 75.59 | 0.42 |
| L→L | ko-KR | BiLSTM | whitespace + mask | 5 | 70.74, 72.95, 70.66, 72.46, 73.22 | 72.01 | 1.22 |
| L→L | ko-KR | TextCNN | char | 5 | 80.35, 78.7, 78.96, 79.11, 78.78 | 79.18 | 0.67 |
| L→L | ko-KR | TextCNN | morph | 2 | 77.76, 75.81 | 76.78 | 1.38 |
| L→L | ko-KR | TextCNN | whitespace | 5 | 70.66, 71.75, 71.15, 71.86, 72.88 | 71.66 | 0.84 |
| L→L | zh-CN | BiLSTM | char + mask | 5 | 78.07, 78.03, 77.42, 77.88, 77.49 | 77.78 | 0.3 |
| L→L | zh-CN | BiLSTM | morph + mask | 5 | 73.31, 74.5, 73.58, 73.08, 74.42 | 73.78 | 0.65 |
| L→L | zh-CN | BiLSTM | whitespace + mask | 5 | 4.18, 4.26, 4.22, 4.26, 4.1 | 4.2 | 0.06 |
| L→L | zh-CN | TextCNN | char | 5 | 78.64, 78.45, 78.64, 79.6, 78.76 | 78.82 | 0.45 |
| L→L | zh-CN | TextCNN | morph | 2 | 74.42, 74.5 | 74.46 | 0.05 |
| L→L | zh-CN | TextCNN | whitespace | 5 | 4.26, 4.29, 4.26, 4.29, 4.18 | 4.26 | 0.05 |
| L→MT | ko-KR | BiLSTM | char + mask | 5 | 51.69, 51.8, 53.98, 52.78, 51.28 | 52.31 | 1.09 |
| L→MT | ko-KR | BiLSTM | whitespace + mask | 5 | 28.81, 32.68, 27.61, 32.23, 32.16 | 30.7 | 2.32 |
| L→MT | ko-KR | TextCNN | char | 5 | 59.5, 57.63, 57.02, 56.72, 58.38 | 57.85 | 1.12 |
| L→MT | ko-KR | TextCNN | whitespace | 5 | 32.83, 33.02, 31.03, 32.23, 33.7 | 32.56 | 1 |
| L→MT | zh-CN | BiLSTM | char + mask | 5 | 62.12, 61.66, 60.93, 63.15, 61.5 | 61.87 | 0.83 |
| L→MT | zh-CN | BiLSTM | whitespace + mask | 5 | 3.99, 4.03, 4.03, 4.14, 3.95 | 4.03 | 0.07 |
| L→MT | zh-CN | TextCNN | char | 5 | 66.1, 65.3, 65.57, 65.8, 65.45 | 65.64 | 0.32 |
| L→MT | zh-CN | TextCNN | whitespace | 5 | 4.06, 3.95, 4.06, 4.1, 3.99 | 4.03 | 0.06 |
| MT→MT | ko-KR | BiLSTM | char + mask | 5 | 71.19, 70.55, 70.81, 70.36, 71.37 | 70.86 | 0.42 |
| MT→MT | ko-KR | BiLSTM | whitespace + mask | 5 | 59.13, 59.84, 60.63, 58.11, 60.14 | 59.57 | 0.98 |
| MT→MT | ko-KR | TextCNN | char | 5 | 75.28, 75.69, 75.88, 74.79, 74.76 | 75.28 | 0.51 |
| MT→MT | ko-KR | TextCNN | whitespace | 5 | 60.82, 60.52, 60.07, 59.95, 60.63 | 60.4 | 0.37 |
| MT→MT | zh-CN | BiLSTM | char + mask | 5 | 75.19, 74.65, 74.08, 75.58, 74.81 | 74.86 | 0.56 |
| MT→MT | zh-CN | BiLSTM | whitespace + mask | 5 | 4.1, 3.95, 3.99, 4.1, 4.06 | 4.04 | 0.07 |
| MT→MT | zh-CN | TextCNN | char | 5 | 77.26, 77.19, 76.23, 76.57, 76.42 | 76.73 | 0.47 |
| MT→MT | zh-CN | TextCNN | whitespace | 5 | 4.1, 4.06, 4.03, 4.14, 4.18 | 4.1 | 0.06 |
| MT→L | ko-KR | BiLSTM | char + mask | 5 | 58.45, 58.26, 56.99, 59.84, 59.43 | 58.6 | 1.11 |
| MT→L | ko-KR | BiLSTM | whitespace + mask | 5 | 33.02, 33.7, 32.34, 30.77, 32.83 | 32.53 | 1.1 |
| MT→L | ko-KR | TextCNN | char | 5 | 63.97, 65.55, 63.11, 64.54, 61.83 | 63.8 | 1.41 |
| MT→L | ko-KR | TextCNN | whitespace | 5 | 33.7, 33.13, 31.78, 32.83, 32.68 | 32.82 | 0.7 |
| MT→L | zh-CN | BiLSTM | char + mask | 5 | 67.18, 67.37, 66.87, 66.72, 65.64 | 66.76 | 0.67 |
| MT→L | zh-CN | BiLSTM | whitespace + mask | 5 | 4.1, 4.1, 4.18, 4.18, 4.14 | 4.14 | 0.04 |
| MT→L | zh-CN | TextCNN | char | 5 | 69.33, 70.28, 68.1, 69.75, 68.6 | 69.21 | 0.88 |
| MT→L | zh-CN | TextCNN | whitespace | 5 | 4.29, 4.26, 4.1, 4.1, 4.1 | 4.17 | 0.1 |


## S8. Full tokenization ablation (formerly Section 4.3, unabridged)


### 4.3 Tokenization ablation, and what the Chinese collapse is not

Figure 5. The double dissociation. Masking the padding position repairs English and Korean but not Chinese, isolating tokenization from the padding defect. Figure 6. Effect of tokenization on 60-class intent classification. Panel (a) TextCNN, panel (b) BiLSTM. **Table 6.** Effect of segmentation on non-pretrained models. Accuracy (%), 60 classes, chance = 1.67%. Mean ± sample standard deviation; **every cell is at n = 5 (seeds 42–46)**, and the per-cell seed count is given in the table itself.

| Locale | Model | whitespace | whitespace + padding mask | morphological (jieba / Okt) | character |
| --- | --- | --- | --- | --- | --- |
| en-US | TextCNN | 80.55 ± 0.64 (5) | n/a | n/a | 79.00 ± 0.36 (5) |
| en-US | BiLSTM | 39.06 ± 20.37 (5) | 79.97 ± 0.70 (5) | n/a | 69.11 ± 0.62 (5) |
| ko-KR | TextCNN | 73.83 ± 0.76 (5) | n/a | 78.66 ± 0.69 (5) | 80.80 ± 0.59 (5) |
| ko-KR | BiLSTM | 44.48 ± 17.31 (5) | **73.36 ± 1.07 (5)** | **77.48 ± 0.44 (5)** | **77.59 ± 0.43 (5)** |
| zh-CN | TextCNN | 10.31 ± 0.22 (5) | n/a | 76.69 ± 0.57 (5) | 80.62 ± 0.46 (5) |
| zh-CN | BiLSTM | 7.03 ± 0.00 (5) | **10.63 ± 0.22 (5)** | **75.66 ± 0.58 (5)** | **79.69 ± 0.30 (5)** |

*Numbers in parentheses are the seed count for that cell. **Every cell in this table is at n = 5 (seeds 42–46).** Bold cells use the padding-masked readout and are the ones on which conclusions rest; the **unmasked** column is shown only to document what the defective configuration produces, and no conclusion rests on it. **One caveat attaches to the English character column.** It was first produced under the fixed maximum sequence length of 30, at which 54.2% of English character-level test utterances are truncated; the values printed here are the corrected ones re-run at a maximum length of 64, which is why the English character figures in this table, in Section 4.3 and in Section 5.1 are now identical. Section S12 gives the correction and the size of the truncation effect.*

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

---

## S9. Near-duplicate robustness: the threshold sweep and the similarity bins

**This section is the evidence behind two sentences of the main text**: Section 4.2.2a states that sweeping the near-duplicate threshold from 0.95 to 0.80 gives a monotone but small drift, and the corresponding sentence of the Discussion states that removing near-duplicates across that sweep changes the headline by only 6–8%. Both are computed by the released `similarity_analysis.py` and stored in `similarity_analysis.json`; the numbers are listed here so that neither sentence has to be taken on trust.

**Procedure.** For every test sentence, the script computes the character 1–3-gram TF-IDF cosine against every training sentence of each of the four provenance cells and keeps the maximum. A test sentence is then discarded if its nearest neighbour in *any* cell reaches the threshold. Threshold 1.01 means no test sentence has an exact match (the fixed evaluation subset already removes exact duplicates within a cell), so the 1.01 row reproduces the headline figures of Tables 5a and 5b.

**Table S9-i. Threshold sweep over near-duplicate removal** (character-level TextCNN, fixed evaluation subset, five seeds). Test-side cost is `L→MT − L→L`; overstatement is `MT→MT − MT→L`.

| Cosine threshold | ko-KR n | ko-KR test-side cost | ko-KR overstatement | zh-CN n | zh-CN test-side cost | zh-CN overstatement |
| --- | --- | --- | --- | --- | --- | --- |
| 1.01 (no removal) | 2,662 | −21.33 | +11.48 | 2,608 | −13.17 | +7.52 |
| 0.95 | 2,641 | −21.20 | +11.56 | 2,578 | −13.23 | +7.53 |
| 0.90 | 2,568 | −20.86 | +11.29 | 2,499 | −12.96 | +7.19 |
| 0.85 | 2,447 | −20.54 | +11.10 | 2,396 | −12.44 | +7.09 |
| 0.80 | 2,276 | −19.96 | +10.83 | 2,261 | −12.31 | +6.93 |

*Reading the sweep.* Every step is in the same direction and no step removes more than 4.5% of the sample between adjacent thresholds; pooled over the four thresholds and the two locales, the largest change in the overstatement is 0.65 points and in the test-side cost 1.37 points. The drift is monotone in the sense that more aggressive removal always shrinks the magnitude of both quantities, and small in the sense that the sign and the order of magnitude are unchanged at every threshold. That is what Section 4.2.2a reports as the near-duplicate robustness check.

**Table S9-ii. The same quantity by English-source similarity quartile** (identical procedure, the threshold replaced by a partition of the test set on the similarity of each test sentence's own source sentence to its cell's training sentences). Q1 is the least similar quartile.

| Quartile | ko-KR n | ko-KR test-side cost | ko-KR overstatement | zh-CN n | zh-CN test-side cost | zh-CN overstatement |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 (least similar) | 666 | −19.22 | +5.62 | 652 | −12.12 | +5.61 |
| Q2 | 665 | −21.29 | +11.01 | 652 | −11.50 | +3.96 |
| Q3 | 665 | −21.98 | +15.19 | 652 | −14.08 | +8.83 |
| Q4 (most similar) | 666 | −22.82 | +14.11 | 652 | −15.00 | +11.69 |

*Reading the bins.* The cost grows monotonically as the test sentence's source sentence becomes more similar to the training distribution, and the overstatement is far larger in the upper quartiles than in the lowest — Korean peaks in the third quartile and falls back slightly in the fourth, Chinese peaks in the fourth. Both are the direction a distribution-match account predicts and the direction the threshold sweep also shows from the other side. **The bins are reported for character-level TextCNN only in this table.** The same partition is extended to BiLSTM, mBERT and XLM-R base, and to the cosine threshold as a layer rather than a bin, in Section S9-iii below; the partition is then stratified by intent and by source length in S9-iv, and every threshold is recomputed under the per-comparison discard rule in S9-v. **Those extensions are exploratory rather than pre-registered**, and the character-level TextCNN panel above remains the one the main text cites.

 **The shape of the failure differs, and this is itself informative.** Chinese is *stably* broken: all three seeds return exactly 7.03%, the majority-class rate, with zero variance. English is *unstably* broken: it ranges from 7.03 to 57.70 across seeds. A model receiving one token per sentence has, literally, no signal to condition on and deterministically emits the prior. A model receiving adequate tokens but a corrupted readout retains signal that initialisation-dependent training may or may not exploit. The variance therefore carries mechanistic information rather than being noise to be averaged away. The dissociation is shown in Figure 5: masking the padding position repairs English and Korean but leaves Chinese where it was.

 **Consequence for an earlier analysis — stated explicitly.** An earlier analysis of the Chinese condition alone attributed it to tokenization; the present results show that this conflated two independent defects. We therefore (i) report both BiLSTM conditions rather than the more favourable one, (ii) treat TextCNN as the tokenization-controlled comparison and BiLSTM ± mask as the padding-controlled comparison, and (iii) do not merge them into a single claim that "Chinese fails".

 **The prediction we specified in advance is confirmed, and the effect is large.** Before running the ablation we stated that if character-level segmentation raised Chinese TextCNN substantially above its whitespace value of 10.31, the segmentation account would be supported. Chinese TextCNN reaches **80.62 ± 0.46** under character-level segmentation — a gain of **+70.3 points** over the whitespace value of 10.31, from near chance to a range comparable with English (79.00 ± 0.36) and Korean (80.80 ± 0.59) under the same scheme, and stable across all five seeds. **The recurrent model shows the same effect once its readout is corrected**: Chinese BiLSTM rises from 10.63 (masked whitespace) to **79.69 ± 0.30** under character-level segmentation, a gain of 69.1 points over the masked whitespace baseline of 10.63. We report the masked pair here; the unmasked values appear in Table 6 and are not used as evidence (Section 4.3). We do not claim the three locales are statistically equivalent, and the spread argues against it. Under character-level segmentation the maximum cross-locale spread is **1.80 points**, from Chinese (79.00 ± 0.36) to Korean (80.80 ± 0.59), compared with **70.24 points** under whitespace segmentation. What the data support is a substantially narrower cross-locale range, not statistical equivalence. We flag this explicitly because "the three languages are equivalent" is the kind of claim a reader may infer from a bar chart, and it is not what we measured.

 This result carries two consequences. *The Chinese failure is a preprocessing choice, not a property of the language or the architecture.* The architecture, the training data, the optimiser, the number of epochs, and the seeds are unchanged; one line of tokenization is. A model that cannot exceed 10.31% is not one that has been shown to lack the capacity to model Chinese — it is one that was never given Chinese to model. We therefore recommend that reports of non-pretrained models performing near chance on Chinese be read, by default, as a tokenization defect until shown otherwise. *Appropriate tokenization makes the three languages comparable.* Under character-level segmentation the three locales land at 79.00, 80.80, and 80.62 — a spread of 1.8 points — whereas under whitespace they span 10.31 to 80.55. The cross-lingual gap that motivated this study was, to a first approximation, an artefact of applying an English-appropriate convention to a language for which it does not hold.

 **Tokenisation is not uniformly beneficial — the effect is language-specific.** If character-level segmentation simply helped everywhere, it would be a trivial recommendation. It does not:

 | Locale | TextCNN, whitespace | TextCNN, character | Change |
| --- | --- | --- | --- |
| en-US | 80.55 | 79.00 | −1.55 |
| ko-KR | 73.83 | 80.80 | +7.0 |
| zh-CN | 10.31 | 80.62 | +70.3 |

Character-level segmentation *reduces* English TextCNN accuracy, where it discards word units that carry the signal; it improves Korean modestly; and it is responsible for nearly the whole Chinese recovery, where it restores units that whitespace splitting had destroyed. This three-way interaction is why the recommendation must be language-conditional, and it also argues against the simplest explanation of the Chinese result — that character tokenization is merely an easier task in general. If it were, English would improve too. **That English reduction must be read with a caveat we can quantify.** The non-pretrained models share a fixed maximum sequence length of 30 positions, which is ample for whitespace tokens (English utterances average 6.9 words, 95th percentile 13) but not for characters: English utterances average 34.9 characters with a 95th percentile of 65, so **54.2% of English test utterances are truncated at character level** (lengths are Unicode characters of the MASSIVE `utt` field, **spaces included**), against 5.1% at a length of 64 and **3.6%** for Korean and **0.7%** for Chinese at length 30. The English character-level condition therefore conflates segmentation with truncation, and we do not use the magnitude of the English reduction as a clean estimate of a tokenization effect. The *direction* is unambiguous — truncation cannot help — but the size is not interpretable from this design, and separating the two would require re-running the non-pretrained models with a locale-specific maximum length. We report the reduction because it is what the pipeline produced, and flag the confound rather than the number. The three segmentation schemes available to the non-pretrained models are plotted in Figure 6, where the language × tokenization interaction is visible as the reversal in the English panel.

 **An incidental third instance of the padding signature.** Character-level English also removes the padding defect, without our intending it: English utterances average 34.9 characters, so character-level sequences exceed the fixed length of 30 and are truncated rather than padded, with the result that the final timestep is a real token. English BiLSTM therefore moves from 39.06 ± 20.37 under whitespace to **67.69 ± 0.26** under character-level segmentation — again a substantial accuracy gain accompanied by a collapse in variance. Together with the English masking ablation (variance 20.37 → 0.70) and the Korean masking ablation (17.31 → 1.07), this is the third instance of the same signature. We regard the repetition as supportive of the mechanism proposed in Section 3.2; we do not claim it as proof, since all three instances share the same dataset and the same two architectures.

 **Any segmentation that respects Chinese boundaries works; the specific scheme matters less than the choice to stop splitting on spaces.** The full Chinese TextCNN ladder:

| Tokenizer | Chinese TextCNN | Change vs whitespace |
| --- | --- | --- |
| whitespace | 10.31 ± 0.22 | — |
| jieba (morphological) | 76.69 ± 0.57 | +66.4 |
| character | 80.62 ± 0.46 | +70.3 |

Both linguistically motivated schemes recover the task almost completely, and they differ from each other by only 3.93 points. This strengthens the interpretation: the failing condition is not "insufficiently sophisticated segmentation" but "no segmentation at all". A practitioner who cannot adopt a Chinese word segmenter can still use character splitting and recover most of the performance.

 The recurrent model shows the same ordering, and **the corrected values are 75.66 ± 0.58 (`jieba`) against 79.69 ± 0.30 (characters)** — a **4.03-point** gap, or roughly thirteen times the standard deviation of the character condition (0.30) and roughly seven times that of the `jieba` condition (0.58), and therefore a clear difference rather than a marginal one. The larger multiple is the one that belongs to the character condition, and we state it rather than the smaller one. `jieba` produces about 5–6 tokens per Chinese utterance against roughly 10.5 characters, so the `jieba` condition remains the more coarsely segmented of the two, and the direction of the gap is what that granularity difference predicts. **The uncorrected unmasked comparison is 42.24 against 69.96 with a per-seed standard deviation of 18.77 for `jieba`** — the widest spread anywhere in this study, and the signature of a partly corrupted readout rather than of a segmentation effect. 
---

*The train–test overlap table and the accuracy on duplicated versus non-duplicated entries (Tables 6b-ii and 6b-iii) are in Supplementary Section S6. In brief: **7.5% of the Chinese test entries occur verbatim in the training split**, they are classified correctly **83.1%** of the time, they supply **60%** of every correct prediction the Chinese whitespace model makes, and removing them drops accuracy from 10.31% to **4.48%** — below the majority-class rate of 7.67%.*


---

### S9-iii. The same partitioning on all four models, and on the cosine threshold — exploratory, not pre-registered

**This subsection and S9-iv/S9-v below are exploratory.** They were computed after the main analysis, in response to a request to extend the results of S9 beyond the single character-level TextCNN panel, and **none of them was pre-registered**. They are reported as extensions of S9-i's threshold sweep and S9-ii's similarity bins and carry no confirmatory weight.

**Basis.** Every cell here is computed on the **fixed evaluation subset** — **ko-KR 2,662** and **zh-CN 2,608** sentences, the mask of `make_canonical.py` lines 40–53 — although the per-sentence prediction files themselves cover the **full test split of 2,974**. The two conventions are never mixed within a table, and the basis is stated in every table title. Test-side cost is `L→MT − L→L`; overstatement is `MT→MT − MT→L`.

**The similarity that bins these tables.** S9-ii bins on the cosine of a test sentence's **English source sentence** against the English training set. The tables below use the manuscript's **primary** similarity instead — the **per-sentence chrF of the MT translation against the localized text of the same sentence** (character n = 1–6, spaces removed, β = 2), whose quartile cuts reproduce Table 5d exactly (ko-KR 8.9 / 15.6 / 26.4; zh-CN 11.9 / 20.7 / 35.0). That check is what licenses reading these as the manuscript's own §4.2.3 convention rather than a new one. Intervals are 95% percentile intervals from 2,000 paired bootstraps over sentences; seeds are the mean over each model's available seeds (five for TextCNN and BiLSTM, three for mBERT and XLM-R base).

**Table S9-iii. Test-side cost and overstatement by chrF quartile, four models** (fixed evaluation subset; ko-KR 2,662 / zh-CN 2,608; Q1 is the least similar quartile). Each cell is `L→L / L→MT / MT→MT / MT→L`, then cost with its interval, then overstatement with its interval.

| Model | Quartile | n | chrF mean | L→L | L→MT | MT→MT | MT→L | Cost | 95% interval | Overstatement | 95% interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TextCNN | Q1 | 666 | 4.7 | 73.75 | 38.74 | 66.13 | 48.17 | **−35.02** | [−38.68, −31.20] | **+17.96** | [14.23, 21.68] |
| TextCNN | Q2 | 662 | 12.2 | 82.02 | 58.85 | 76.59 | 65.86 | −23.17 | [−26.40, −20.00] | +10.73 | [7.82, 13.69] |
| TextCNN | Q3 | 668 | 20.2 | 80.96 | 64.22 | 79.67 | 68.56 | −16.74 | [−19.52, −13.92] | +11.11 | [8.77, 13.77] |
| TextCNN | Q4 | 666 | 42.8 | 80.00 | 69.58 | 78.74 | 72.61 | −10.42 | [−12.70, −8.14] | +6.13 | [4.20, 8.02] |
| BiLSTM | Q1 | 666 | 4.7 | 71.74 | 34.65 | 61.74 | 44.80 | −37.09 | [−40.69, −33.45] | +16.94 | [13.30, 20.60] |
| BiLSTM | Q2 | 662 | 12.2 | 78.16 | 53.78 | 72.27 | 59.03 | −24.38 | [−27.34, −21.57] | +13.23 | [10.03, 16.35] |
| BiLSTM | Q3 | 668 | 20.2 | 77.43 | 58.14 | 75.69 | 62.87 | −19.28 | [−22.16, −16.38] | +12.81 | [10.24, 15.54] |
| BiLSTM | Q4 | 666 | 42.8 | 75.80 | 62.64 | 73.72 | 67.66 | −13.15 | [−15.59, −10.75] | +6.07 | [3.87, 8.20] |
| mBERT | Q1 | 666 | 4.7 | 76.28 | 48.35 | 68.57 | 57.26 | −27.93 | [−31.53, −24.32] | +11.31 | [7.96, 14.81] |
| mBERT | Q2 | 662 | 12.2 | 83.03 | 65.31 | 77.90 | 72.05 | −17.72 | [−20.59, −14.85] | +5.84 | [2.72, 8.81] |
| mBERT | Q3 | 668 | 20.2 | 81.79 | 65.62 | 79.49 | 72.36 | −16.17 | [−18.91, −13.42] | +7.14 | [4.64, 9.68] |
| mBERT | Q4 | 666 | 42.8 | 81.63 | 73.12 | 79.68 | 76.53 | −8.51 | [−10.81, −6.16] | +3.15 | [1.45, 5.01] |
| XLM-R base | Q1 | 666 | 4.7 | 78.58 | 59.31 | 70.12 | 67.77 | −19.27 | [−22.37, −16.02] | +2.35 | [−0.75, 5.36] |
| XLM-R base | Q2 | 662 | 12.2 | 85.05 | 73.46 | 80.36 | 79.41 | −11.58 | [−14.20, −9.06] | +0.96 | [−1.51, 3.32] |
| XLM-R base | Q3 | 668 | 20.2 | 83.43 | 74.00 | 82.78 | 78.89 | −9.43 | [−11.93, −6.99] | +3.89 | [1.90, 5.99] |
| XLM-R base | Q4 | 666 | 42.8 | 83.63 | 78.73 | 81.78 | 82.08 | −4.90 | [−6.86, −3.00] | −0.30 | [−1.70, 1.15] |
| TextCNN | Q1 | 652 | 6.8 | 75.55 | 47.88 | 70.12 | 55.86 | −27.67 | [−31.07, −24.08] | **+14.26** | [10.80, 17.79] |
| TextCNN | Q2 | 652 | 16.1 | 79.42 | 64.45 | 77.15 | 69.48 | −14.97 | [−17.85, −12.02] | +7.67 | [4.97, 10.46] |
| TextCNN | Q3 | 652 | 27.1 | 80.18 | 72.45 | 79.05 | 73.71 | −7.73 | [−10.21, −5.43] | +5.34 | [3.10, 7.61] |
| TextCNN | Q4 | 652 | 57.7 | 80.12 | 77.79 | 80.61 | 77.79 | −2.33 | [−4.17, −0.52] | +2.82 | [1.04, 4.48] |
| BiLSTM | Q1 | 652 | 6.8 | 74.60 | 44.69 | 69.51 | 53.71 | −29.91 | [−33.34, −26.56] | +15.80 | [12.48, 19.29] |
| BiLSTM | Q2 | 652 | 16.1 | 78.74 | 61.10 | 75.31 | 66.63 | −17.64 | [−20.46, −14.78] | +8.68 | [6.04, 11.26] |
| BiLSTM | Q3 | 652 | 27.1 | 79.08 | 68.59 | 77.39 | 71.81 | −10.49 | [−12.98, −8.22] | +5.58 | [3.31, 7.85] |
| BiLSTM | Q4 | 652 | 57.7 | 78.68 | 73.10 | 77.24 | 74.88 | −5.58 | [−7.58, −3.65] | +2.36 | [0.64, 4.14] |
| mBERT | Q1 | 652 | 6.8 | 80.01 | 60.69 | 74.90 | 70.25 | −19.33 | [−22.70, −16.10] | +4.65 | [1.58, 7.67] |
| mBERT | Q2 | 652 | 16.1 | 83.79 | 73.72 | 81.03 | 78.99 | −10.07 | [−12.53, −7.51] | +2.04 | [−0.26, 4.24] |
| mBERT | Q3 | 652 | 27.1 | 83.18 | 78.02 | 81.85 | 80.32 | −5.16 | [−7.36, −3.17] | +1.53 | [−0.41, 3.37] |
| mBERT | Q4 | 652 | 57.7 | 83.64 | 81.65 | 82.31 | 80.98 | −1.99 | [−3.32, −0.66] | +1.33 | [−0.10, 2.76] |
| XLM-R base | Q1 | 652 | 6.8 | 81.34 | 65.64 | 76.07 | 76.12 | −15.70 | [−18.76, −12.73] | −0.05 | [−3.02, 2.61] |
| XLM-R base | Q2 | 652 | 16.1 | 84.56 | 76.48 | 80.47 | 81.39 | −8.08 | [−10.28, −5.83] | −0.92 | [−2.86, 0.97] |
| XLM-R base | Q3 | 652 | 27.1 | 84.00 | 80.11 | 83.03 | 82.98 | −3.89 | [−5.88, −1.99] | +0.05 | [−1.64, 1.64] |
| XLM-R base | Q4 | 652 | 57.7 | 84.66 | 82.77 | 83.28 | 83.18 | −1.89 | [−3.02, −0.77] | +0.10 | [−1.23, 1.53] |

**The first sixteen rows are ko-KR and the last sixteen are zh-CN.** The bin counts are not equal within a locale (ko 666 / 662 / 668 / 666; zh 652 × 4) because ties are broken by an explicit half-open interval rather than by `np.digitize`; the realized quartile **boundaries** are the ones printed above and they are the boundaries Table 5d uses.

*What the four models agree on.* **The test-side cost falls monotonically as chrF rises, on all four models and both locales** — the least similar quartile is the most damaged, and the damage shrinks by a factor of 2.3–3.4 between Q1 and Q4. **The overstatement is also largest in Q1 on every model and both locales**; on the weakest models the top quartile is where it nearly vanishes, and for XLM-R base on Korean it is **−0.30 [−1.70, +1.15]**, an interval that contains zero. **The stronger the model, the flatter the whole curve**: Q1 cost runs from −37.09 (BiLSTM) to −19.27 (XLM-R base), and Q1 overstatement from +16.94 to +2.35. That is the same shape as the body's claim that the cost shrinks as model strength grows, seen inside the bins rather than across models. **One caution we state rather than bury:** `L→L` itself is lower in Q1 (73.75 / 71.74 / 76.28 / 78.58 on Korean), so part of the Q1 *cost* is sentence difficulty. The overstatement is a difference between two cells evaluated on the **same** sentences, so sentence difficulty cancels in it — which is why "Q1 has the largest overstatement" is not a difficulty artefact.

**Table S9-iv. The same two quantities layered by the manuscript's secondary similarity criterion** — a test sentence's nearest-neighbour cosine of **at least 0.8** against a training cell (character 1–3-gram TF-IDF, pooled vocabulary fit over the localized and translated training sets, ties broken by the lowest training index). Reported as a two-way split rather than as bins, because S9-i already reports the threshold sweep itself.

| Locale | Layer | n (pooled fit) | Share (published-script fit) | TextCNN cost / overst. | BiLSTM cost / overst. | mBERT cost / overst. | XLM-R base cost / overst. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ko-KR | L test → L train ≥ 0.8 | 195 | 7.3% (7.1%) | −38.56 / +9.23 | −41.64 / +13.23 | −27.18 / +7.52 | −19.15 / +0.34 |
| ko-KR | MT test → MT train ≥ 0.8 | 225 | 8.5% (13.7%) | −24.98 / +22.13 | −24.36 / +24.80 | −17.04 / +13.33 | −12.30 / +4.44 |
| ko-KR | MT test → L train ≥ 0.8 | 19 | 0.7% (0.8%) | −1.05 / +8.42 | +1.05 / +8.42 | 0.00 / +12.28 | +3.51 / 0.00 |
| ko-KR | English source → English train ≥ 0.8 | 488 | 18.3% (n/a) | −23.98 / +16.31 | −26.89 / +17.62 | −16.67 / +9.36 | −11.13 / +4.30 |
| zh-CN | L test → L train ≥ 0.8 | 162 | 6.2% (6.5%) | −21.23 / +8.40 | −21.48 / +6.91 | −14.81 / +3.09 | −12.14 / +0.41 |
| zh-CN | MT test → MT train ≥ 0.8 | 184 | 7.1% (12.5%) | −17.17 / +17.17 | −20.22 / +19.02 | −9.78 / +9.42 | −5.98 / +3.26 |
| zh-CN | MT test → L train ≥ 0.8 | 45 | 1.7% (2.7%) | −1.33 / +7.11 | +2.22 / +5.33 | −2.22 / +3.70 | −1.48 / +0.74 |
| zh-CN | English source → English train ≥ 0.8 | 455 | 17.4% (n/a) | −17.01 / +12.40 | −19.96 / +12.13 | −10.33 / +6.37 | −7.69 / +1.76 |

*Reading the layers.* **The largest overstatement in the table sits in the layer that is near-duplicate with respect to the translated training set** (`MT test → MT train ≥ 0.8`: +22.13 ko, +17.17 zh on TextCNN), which is the layer the fixed evaluation subset does not remove because its duplicate test is against a *different* provenance cell. **The layer with almost no cross-source neighbours** (`MT test → L train ≥ 0.8`, 19 Korean and 45 Chinese sentences) has correspondingly almost no measurable behaviour — the estimate flips sign across models and rests on 19 or 45 sentences, so we read nothing from it beyond "the layer is nearly empty", which is what Table 5l reports from the other direction. **Two fitting conventions are shown because they do not agree**: the published `similarity_analysis.py` fits one vectorizer on the localized training set only, which depresses cross-source cosines (see S9-i's note); the pooled fit is used for the layer sizes above. The `MT → MT` share differs between the two by a factor of about 1.6 (8.5% against 13.7% on Korean) and the `MT → L` share is 0.7–0.8% under both, against the **0.3%** printed in Table 5l. **We cannot reproduce 0.3% under either convention and we say so rather than presenting one as the source of the other.**

### S9-iv. Intent and source-length stratification — exploratory, not pre-registered

**Table S9-v. Intent-frequency bands** (fixed evaluation subset; intents are the MASSIVE `intent` field, grouped by how many sentences the class has *inside the fixed subset*, so that the bands are comparable in size). Each cell is `cost / overstatement`.

| Locale | Band (sentences in the subset) | n | Classes | TextCNN | BiLSTM | mBERT | XLM-R base |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ko-KR | 10–19 | 25 | 7 | −16.80 / +12.80 | −17.60 / +6.40 | −9.33 / 0.00 | −12.00 / 0.00 |
| ko-KR | 20–39 | 161 | 11 | −40.99 / +11.30 | −41.37 / +13.91 | −24.22 / +5.80 | −19.46 / +2.90 |
| ko-KR | 40+ | 615 | 21 | −25.27 / +15.80 | −26.08 / +15.09 | −19.19 / +7.37 | −13.44 / +2.38 |
| zh-CN | 10–19 | 38 | 8 | −14.74 / +2.11 | −16.32 / +3.68 | −9.65 / +1.75 | −7.89 / −1.75 |
| zh-CN | 20–39 | 145 | 10 | −18.07 / +20.14 | −25.52 / +21.52 | −19.77 / +16.78 | −9.89 / +5.98 |
| zh-CN | 40+ | 567 | 20 | −18.59 / +11.92 | −22.50 / +12.38 | −11.29 / +5.35 | −9.94 / +0.59 |

**Two facts about coverage, stated because they limit the table.** MASSIVE has **60 intents**, but **only 59 appear in the fixed evaluation subset on either locale** — the mask removes **every** test sentence of `cooking_query` in both Korean and Chinese, so that class cannot be stratified at all. **18 classes have fewer than 20 sentences in the subset** (11 of them fewer than 10), which is why the lowest band is 10–19 and why classes below it are excluded from the bands while their sentences still count toward the 2,662 / 2,608 totals. **41 classes reach at least 20 sentences in each locale** (covering 2,476 Korean and 2,425 Chinese sentences), and the sign of the overstatement was counted on those 41 classes:

| Locale | TextCNN, overstatement > 0 | BiLSTM | mBERT | XLM-R base |
| --- | --- | --- | --- | --- |
| ko-KR | **37 / 41 (90%)** | 39 / 41 (95%) | 28 / 41 (68%) | 16 / 41 (39%) |
| zh-CN | **30 / 41 (73%)** | 32 / 41 (78%) | 26 / 41 (63%) | 12 / 41 (29%) |

*Reading the intent split.* **The overstatement is the majority sign in most intents but not in all of them**, and the count falls with model strength in the same order the headline does. **No class reverses it systematically, and no class is exempt from it** — which is the two-sided statement the data support: the effect is not a property of one intent family, and it is not a universal constant either. **The lowest band (10–19 sentences) is unstable on both cost and overstatement** and we read nothing from it; the two higher bands are where the numbers are estimable. The per-class table for all 41 classes on both locales is **not reproduced here**; it would take 82 rows and the two tables above carry the finding.

**Table S9-vi. Source-length terciles** (fixed evaluation subset; terciles are cut *within* each locale, so the cut points differ between Korean and Chinese). The stratification variable is the character length of the **English source sentence**, which is the variable §4.2.3 uses in the body. Each cell is `cost / overstatement`.

| Locale | Tercile | n | Range | Mean | TextCNN | BiLSTM | mBERT | XLM-R base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ko-KR | T1 | 876 | [2, 28] | 19.94 | −23.65 / +9.86 | −24.11 / +9.50 | −20.40 / +5.21 | −13.17 / +1.33 |
| ko-KR | T2 | 884 | [28, 40] | 33.14 | −22.90 / +10.38 | −24.59 / +13.26 | −16.67 / +6.90 | −11.20 / +1.28 |
| ko-KR | T3 | 902 | [40, 365] | 53.62 | −17.54 / +14.12 | −21.75 / +13.97 | −15.74 / +8.43 | −9.57 / +2.55 |
| zh-CN | T1 | 841 | [4, 28] | 20.15 | −15.05 / +7.06 | −17.36 / +7.28 | −11.65 / +2.81 | −11.30 / −0.75 |
| zh-CN | T2 | 869 | [28, 40] | 33.15 | −13.97 / +8.47 | −16.18 / +8.54 | −9.21 / +2.53 | −6.06 / 0.00 |
| zh-CN | T3 | 898 | [40, 365] | 53.67 | −10.65 / +7.04 | −14.28 / +8.46 | −6.72 / +1.86 | −5.01 / +0.11 |

*Reading the length split, and one contrast we did not expect.* **The overstatement rises monotonically with source length on three of the four models in both locales** (Korean TextCNN +9.86 → +10.38 → +14.12; Chinese TextCNN +7.06 → +8.47 → +7.04 is the exception among the eight series, and Korean TextCNN's middle step is 0.5 points). **The cost does not track length in proportion** — it is *smaller* in the longest tercile on every model, while the overstatement is *larger* there. This is not in tension with the body's report that English source length is essentially uncorrelated with chrF (r = −0.011 and −0.020 on Korean; +0.009 and −0.011 on Chinese, reproduced here to the printed digits): the correlation is between length and the *magnitude of the similarity*, whereas the overstatement is a **difference between two cells**. Under the other three length variables we tried — English source **word** count, MT translation length in characters, and the **difference** between the translation length and the localized length — the overstatement also rises from the shortest to the longest tercile on every model in both locales (Korean, source word count, TextCNN: +7.26 → +11.37 → +13.97; Korean, length difference, TextCNN: +8.74 → +9.74 → +14.93). **We flag this length gradient as a new observation**: it was not pre-registered, it does not appear in the manuscript, and it is offered as a candidate for the confirmation experiment rather than as a result.

**One stratification we deliberately do not draw a conclusion from.** Sentences whose MT translation exceeds the non-pretrained models' 30-position cap number **40 in Korean (1.5%)** and **18 in Chinese (0.7%)** — the truncation problem is a `de-DE`/`vi-VN` problem (66.4% and 62.2% in §3.2), not a Korean/Chinese one. We therefore do not use the truncated layer's cost or overstatement, and the pretrained encoders are not subject to the cap at all.

### S9-v. Every threshold recomputed under the advisor's per-comparison rule — exploratory, not pre-registered

**Why this subsection exists.** S9-i sweeps 1.01 → 0.80 but discards a test sentence if its nearest neighbour reaches the threshold in **any** of the four cells. The reviewer's instruction for the confirmation experiment is stricter about *which* comparisons share a discard list: *"each comparison must remove the same IDs. For example the overstatement is `MT→MT` minus `MT→L`, so if the neighbour in either of the two cells reaches the threshold, that ID is removed from both."* The tables below implement that rule and report the manuscript's rule beside it, so that the two can be compared rather than substituted silently.

| Comparison | Training cell the two cells share | Discard condition |
| --- | --- | --- |
| Test-side cost = `L→MT − L→L` | the **localized** training set | `cos(L test → L train) ≥ t` **or** `cos(MT test → L train) ≥ t` |
| Overstatement = `MT→MT − MT→L` | the **translated** training set | `cos(MT test → MT train) ≥ t` **or** `cos(L test → MT train) ≥ t` |

Fit: character 1–3-gram TF-IDF, `sublinear_tf = True`, **pooled** vocabulary over the localized and translated training sets, L2-normalised, cosine by inner product, ties broken by the lowest training index.

**Table S9-vii. Retained sample size.** The two rules do not remove the same number of sentences, and the difference is the point of the exercise.

| Locale | t | For the cost (rule above) | For the overstatement (rule above) | Any of the four cells (S9-i's rule) |
| --- | --- | --- | --- | --- |
| ko-KR | 0.7 | 2,159 (18.9% removed) | 2,115 (20.5%) | 1,841 (30.8%) |
| ko-KR | 0.8 | 2,455 (7.8%) | 2,433 (8.6%) | 2,276 (14.5%) |
| ko-KR | 0.9 | 2,617 (1.7%) | 2,608 (2.0%) | 2,568 (3.5%) |
| zh-CN | 0.7 | 2,138 (18.0%) | 2,111 (19.1%) | 1,847 (29.2%) |
| zh-CN | 0.8 | 2,419 (7.2%) | 2,393 (8.2%) | 2,261 (13.3%) |
| zh-CN | 0.9 | 2,554 (2.1%) | 2,545 (2.4%) | 2,499 (4.2%) |

**Table S9-viii. Residual cost and residual overstatement on all four models, three thresholds** (fixed evaluation subset; the unremoved column is the same quantity as Table 5a/5b). Cells are `cost / overstatement`.

| Locale | t | TextCNN | BiLSTM | mBERT | XLM-R base |
| --- | --- | --- | --- | --- | --- |
| ko-KR | none | −21.33 / +11.48 | −23.47 / +12.26 | −17.58 / +6.86 | −11.29 / +1.73 |
| ko-KR | 0.7 | −18.81 / +9.86 | −21.25 / +10.51 | −16.30 / +5.94 | −10.45 / +1.15 |
| ko-KR | 0.8 | −20.06 / +10.51 | −22.15 / +11.12 | −16.90 / +6.27 | −10.75 / +1.48 |
| ko-KR | 0.9 | −20.96 / +11.25 | −23.09 / +11.93 | −17.26 / +6.71 | −10.88 / +1.65 |
| zh-CN | none | −13.17 / +7.52 | −15.90 / +8.11 | −9.14 / +2.39 | −7.39 / −0.20 |
| zh-CN | 0.7 | −12.67 / +6.91 | −15.47 / +7.44 | −8.70 / +1.75 | −6.97 / −0.58 |
| zh-CN | 0.8 | −12.79 / +6.94 | −15.73 / +7.51 | −8.83 / +1.99 | −7.14 / −0.38 |
| zh-CN | 0.9 | −13.27 / +7.25 | −15.98 / +7.85 | −9.08 / +2.15 | −7.40 / −0.25 |

**Table S9-ix. The same quantities at the primary threshold t = 0.8, with 95% intervals** (2,000 paired bootstraps over sentences; fixed evaluation subset). This is the table P8 is judged on.

| Locale | Model | Residual cost | 95% interval | Residual overstatement | 95% interval |
| --- | --- | --- | --- | --- | --- |
| ko-KR | TextCNN | −20.06 | [−21.61, −18.47] | +10.51 | [8.98, 11.99] |
| ko-KR | BiLSTM | −22.15 | [−23.63, −20.55] | +11.12 | [9.63, 12.66] |
| ko-KR | mBERT | −16.90 | [−18.55, −15.51] | +6.27 | [4.82, 7.78] |
| ko-KR | XLM-R base | −10.75 | [−12.15, −9.41] | +1.48 | [0.27, 2.66] |
| zh-CN | TextCNN | −12.79 | [−14.20, −11.35] | +6.94 | [5.56, 8.27] |
| zh-CN | BiLSTM | −15.73 | [−17.17, −14.33] | +7.51 | [6.16, 8.83] |
| zh-CN | mBERT | −8.83 | [−10.21, −7.58] | +1.99 | [0.85, 3.19] |
| zh-CN | XLM-R base | −7.14 | [−8.34, −6.01] | −0.38 | [−1.43, 0.67] |

*Reading the two rules together.* **The sign and the order of magnitude of both quantities survive every threshold and both discard rules**; the largest disagreement between the two rules anywhere in Table S9-viii is 0.50 points of overstatement (Korean TextCNN at t = 0.7: +9.86 under the advisor's rule against +10.36 under S9-i's). **At t = 0.8 the manuscript's rule gives +10.83 for Korean TextCNN on the four-cell basis, and this analysis returns +10.83 digit for digit** — which is the check that the two implementations are computing the same quantity. The advisor's rule is the **stricter** of the two on the overstatement (it removes 8.6% of Korean sentences against 14.5% for the four-cell rule at the same threshold, yet yields a *smaller* residual, +10.51 against +10.83); both values lie inside the interval the manuscript prints for its own rule, [9.29, 12.34], so **the choice of discard rule does not change the conclusion**.

**Two self-checks that were run before any of these tables was written.** The four-cell means on the fixed subset reproduce `canonical_numbers.json` **16 of 16 keys digit for digit**; the mBERT and XLM-R base cells, which that file does not carry, reproduce Table 5n **16 of 16 digit for digit**; the chrF quartile boundaries reproduce Table 5d exactly; and the chrF–length correlations reproduce the two body values to the printed digits. A failure in any of these aborts the computation rather than writing a table.

---

---

## S10. What the efficiency measurements do not support (formerly Section 5.4)

### What the measurements do not support

**The measurements themselves are in Section S1; this section states only what they cannot be used for.**

The practical question for a smart-speaker vendor is not which model is most accurate in absolute terms, but which is accurate enough within a memory, latency, and power budget. For pretrained runs in which efficiency instrumentation was active, our design records parameter count, model size, measured inference latency, and peak GPU memory, so the recommendation can be made on measured trade-offs rather than on efficiency figures quoted from the papers that introduced the models.

 **Three limitations must be attached to any comparison across model families, and they prevent the strongest reading of Table 8a and Table 8b.**

*The hardware differs.* The non-pretrained models were trained and evaluated on CPU; the pretrained models on a Tesla T4. Training time and inference latency are therefore directly comparable **within** the non-pretrained group and **within** the pretrained group, but not between them. In particular, the statement that TextCNN evaluates an utterance in 0.16 ms against DistilBERT's 3.62 ms compares a CPU figure with a GPU figure and should be read as an indication of the order of magnitude, not as a measured speed ratio.

 *The training budgets differ.* TextCNN and BiLSTM were trained for 15 epochs and the pretrained models for 3, because the pretrained models converge within 3 and the non-pretrained ones do not. Training-time comparisons across the two groups inherit that difference.

 *A T4 is not an edge device.* The GPU measurements describe a small server accelerator, not the class of hardware an on-device smart-home assistant would use. They bound what a model costs to run; they do not simulate deployment.

What survives these limitations is the comparison that shares both hardware and budget. Section S1 gives those within-block comparisons: DistilBERT is the faster pretrained model for a loss of 0.6 accuracy points in English and 1.6 in Korean, and the two non-pretrained architectures are close in both accuracy and cost. We state those directions and **not** a training-time ratio, because identical configurations varied by up to 3.1× in measured training time between sessions. The cross-group claim we are entitled to make is qualitative: TextCNN is orders of magnitude smaller than the pretrained checkpoints examined here while retaining competitive accuracy in English and Korean; because checkpoint size varies by locale and some English and Korean parameter counts are nominal rather than instrumented measurements, we do not express this trade-off as a single cross-locale parameter-to-accuracy ratio.

*The architectures are close within each group.* Section S1 gives the three between-architecture differences and their standard deviations; all are small relative to the between-seed spread, and **we do not claim that TextCNN is the more accurate of the two** — the claim we make is about deployability, and it rests on measured size rather than on a ratio.


---

## S11. Zero-shot transfer: one-paragraph summary of Section S2

We also trained `bert-base-chinese` on **English** and evaluated it zero-shot on Korean and Chinese. It reaches **65.78 ± 1.70** on Chinese — a 14-point margin over mBERT from a monolingual Chinese checkpoint that has never seen multilingual training — and **6.32 ± 0.38** on Korean, near chance. **Because a result this surprising invites the suspicion of a pipeline error, we ran the four checks that would expose one**, and the result survives all of them: a **randomly initialised** model of the same architecture reaches **4.34 ± 0.20** on Chinese, the loader verifies the resolved path and checksum of every file it opens, accuracy is spread across **37 of the 59 intents**, and stripping digits and Latin characters changes Chinese accuracy by at most 0.3 points. **The result is real and reproducible, and we still cannot explain the mechanism. This experiment is not part of this paper's argument and we build no claim on it**; Section S2 gives the protocol, both tables and the per-intent breakdown.

---

## S12. Supplementary notes on Section 4.3 (English truncation; an earlier ladder; the readout control)

**This section carries three notes on Section 4.3. The third reports the packed-sequence readout control; the first two are the truncation correction and the reverted ladder.**

**A confound we removed rather than annotating.** English utterances average **34.9 characters**, but the non-pretrained models use a fixed maximum sequence length of 30, so **54.2% of English test utterances were truncated** at 30 characters in the character-level condition. The count is taken on the `utt` field as shipped, **spaces included**; the character tokenizer described above then removes spaces before splitting, so this percentage is a property of the raw string and not of the model's own sequence length. That figure therefore mixed tokenization with truncation. **Re-running with the maximum length raised to 64 changes the English character figures from 76.42 and 67.69 to 79.00 and 69.11** — so the earlier "4.1-point cost" attributed 2.58 of its 4.13 points to truncation, and the corrected cost is 1.55 points for TextCNN and 10.86 for BiLSTM (Section 4.3). **We report the corrected values and flag the original as an error of our own instrumentation.**

**A correction we owe the record.** An earlier version of this analysis compared the unmasked BiLSTM across schemes and reported a tidy ladder from whitespace through morphology to characters. **The corrected readout reverses that picture**: the masked whitespace baseline alone (73.36) exceeds every unmasked morphological and character value, so the ladder was an artefact of the readout and not of segmentation. Only two of the three steps survive the correction, and the morphological-to-character step is 2.14 points for TextCNN against a standard deviation of 0.6–0.7.

### S12-iii. BiLSTM packed-sequence readout control, all eight locale–tokenizer conditions

**A reviewer noted that the padding-masked recurrent readout is only a partial repair.** It selects the last non-padding timestep, but the backward half of the LSTM still reaches that timestep only after traversing the padding, and post-hoc masking cannot undo that. **We therefore re-ran every locale and tokenizer condition with two further readouts and report all three side by side.** `lastmasked` is the post-hoc mask used in Table 6: the output at the last non-padding timestep, multiplied by a mask. `packed` uses `pack_padded_sequence`, so the recurrence never sees a padding position at all. `bothfinal` concatenates the final hidden state of each direction (forward at the last real timestep, backward at the first), so neither half is read after a padding traversal.

**Table S12-iii. The three readouts on identical training conditions.** Full test split (2,974 utterances per locale), localized training, `L→L`, five seeds (42–46); accuracy (%), 60 classes, chance 1.67%. Standard deviations are sample standard deviations (denominator $n-1$), the same basis as Table 6. The last two columns are differences against `lastmasked`, in points.

| Locale | Tokenizer | `lastmasked` (post-hoc mask) | `packed` (`pack_padded_sequence`) | `bothfinal` (both directions' final states) | `packed` − `lastmasked` | `bothfinal` − `lastmasked` |
| --- | --- | --- | --- | --- | --- | --- |
| en-US | whitespace | 79.97 ± 0.70 | 79.79 ± 0.34 | 79.82 ± 0.35 | −0.18 | −0.15 |
| ko-KR | whitespace | 73.36 ± 1.07 | 73.23 ± 0.58 | 74.12 ± 1.10 | −0.13 | +0.76 |
| zh-CN | whitespace | 10.63 ± 0.22 | **10.68 ± 0.17** | 10.03 ± 0.24 | +0.05 | −0.60 |
| ko-KR | morphological (Okt) | 77.48 ± 0.44 | 77.29 ± 0.25 | 77.80 ± 0.91 | −0.19 | +0.32 |
| zh-CN | morphological (jieba) | 75.66 ± 0.58 | 75.96 ± 0.46 | 76.03 ± 0.59 | +0.30 | +0.37 |
| en-US | character | 67.69 ± 0.26 | 67.92 ± 0.28 | **70.68 ± 0.69** | +0.23 | **+2.99** |
| ko-KR | character | 77.59 ± 0.43 | 77.42 ± 0.51 | 78.04 ± 0.51 | −0.17 | +0.45 |
| zh-CN | character | 79.69 ± 0.30 | 79.49 ± 0.32 | 79.95 ± 0.49 | −0.20 | +0.26 |

*Every cell is at n = 5 seeds (42–46) and every cell is on the full test split, so the three columns are directly comparable to one another and to Table 6. The `lastmasked` column is the one printed in Table 6 of the main text; the `packed` and `bothfinal` columns come from re-runs that change `--bilstm-readout` and nothing else (same dataset, `localized` training, in-language mode, 15 epochs, batch 32, `SEQ_LEN = 30`, seeds 42–46, padding mask on). One exception is recorded because it is real: the `en-US` whitespace `lastmasked` cell is Table 6's five-seed value (79.97 ± 0.70); the two `en-US` whitespace seeds whose per-sentence predictions are still in the deposit give 79.59 and 79.96. The English character `lastmasked` baseline has two versions in this project: the value on the released run records is 67.69 ± 0.26 and the corrected value printed in Table 6 is 69.11 ± 0.62 (Section S12, first note). The table gives 67.69; **against the corrected baseline the last difference in the table becomes +1.57 instead of +2.99, and the direction of every statement below is unchanged.***

**Three things follow, in the order in which they constrain the paper.**

1. **`packed` and the post-hoc mask differ by at most 0.23 points in all eight conditions** (seed standard deviations 0.17–1.10). Letting the recurrence never see padding — the strictest repair available — therefore moves nothing. **The readout is not the cause of the Chinese collapse.**
2. **`bothfinal` agrees with the mask in seven of the eight conditions; the single exception is English under character segmentation (70.68 ± 0.69 against 67.69 ± 0.26, +2.99 points).** That gap is roughly four times the larger of the two standard deviations and we do not treat it as noise. It is confined to the readout we do not recommend, and it does not reappear on any other locale or tokenizer. The second-largest disagreement in the table is Chinese whitespace (`bothfinal` −0.60 against `lastmasked`), which likewise exceeds that cell's seed variation (sd 0.17–0.24) but moves Chinese from 10.63 to 10.03, inside the same band.
3. **The double dissociation holds under all three readouts.** The best-repaired Chinese whitespace condition, `packed` at **10.68%**, is still above the 1.67% chance rate and above the 7.03% majority-class rate, while the same architecture reaches 75.9–76.0 under morphological and 79.5–80.0 under character segmentation. Masking repairs English and Korean and does not repair Chinese under any readout we tested.

---

## S13. Timeline of corpus generation, runs and pre-registration

**Advisor question 3 asked when each step happened.** The table below reconstructs the sequence
from two kinds of machine-readable evidence rather than from recollection: the `manifest_v2.json`
that every run directory writes (start and finish timestamps), and file mtimes for the translated
corpora. Where evidence is absent the row says so.

| When (local, +0800) | Event | Evidence in the deposit |
| --- | --- | --- |
| 2026-10-03 20:27 | `massive_mt` ko-KR corpus generated (`facebook/nllb-200-distilled-600M`) | corpus mtime, read from the pre-patch backup `massive_mt/ko-KR.jsonl.before_mtsource.bak` |
| 2026-10-03 21:25 | `massive_mt` zh-CN corpus generated (same system) | as above, `massive_mt/zh-CN.jsonl.before_mtsource.bak` |
| 2026-10-04 00:23 | run directory `sci_paper/runs_v2_kaggle/manifest_v2.json` starts | `sci_paper/runs_v2_kaggle/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 00:34 | run directory `sci_paper/pilot_cpu/manifest_v2.json` starts | `sci_paper/pilot_cpu/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 03:01 | run directory `sci_paper/runs_c1/manifest_v2.json` starts | `sci_paper/runs_c1/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 12:39 | run directory `sci_paper/kaggle_v3/runs_e2b/manifest_v2.json` starts | `sci_paper/kaggle_v3/runs_e2b/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 12:39 | run directory `sci_paper/runs_e2b/manifest_v2.json` starts | `sci_paper/runs_e2b/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 17:03 | run directory `sci_paper/kaggle_v3/runs_eff/manifest_v2.json` starts | `sci_paper/kaggle_v3/runs_eff/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 17:03 | run directory `sci_paper/runs_eff/manifest_v2.json` starts | `sci_paper/runs_eff/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 20:52 | run directory `sci_paper/runs_pred_loc/manifest_v2.json` starts | `sci_paper/runs_pred_loc/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 21:10 | run directory `sci_paper/runs_pred_mt/manifest_v2.json` starts | `sci_paper/runs_pred_mt/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 21:27 | run directory `sci_paper/runs_c1b/manifest_v2.json` starts | `sci_paper/runs_c1b/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 21:57 | run directory `sci_paper/runs_pred_mt_mask/manifest_v2.json` starts | `sci_paper/runs_pred_mt_mask/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-04 22:08 | run directory `sci_paper/runs_pred_loc_mask/manifest_v2.json` starts | `sci_paper/runs_pred_loc_mask/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-05 02:13 | run directory `sci_paper/runs_v4a/manifest_v2.json` starts | `sci_paper/runs_v4a/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-05 02:49 | run directory `sci_paper/runs_v4c/manifest_v2.json` starts | `sci_paper/runs_v4c/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-05 06:06 | run directory `sci_paper/runs_c2/manifest_v2.json` starts | `sci_paper/runs_c2/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-05 06:26 | run directory `sci_paper/runs_e4_char/manifest_v2.json` starts | `sci_paper/runs_e4_char/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-05 12:07 | run directory `sci_paper/runs_v4b/manifest_v2.json` starts | `sci_paper/runs_v4b/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-05 14:58 | run directory `sci_paper/runs_unmask/manifest_v2.json` starts | `sci_paper/runs_unmask/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-05 16:07 | `massive_mt2` ko-KR / zh-CN and `massive_mt_600` ko-KR / zh-CN corpora generated | corpus mtimes, read from the pre-patch backups |
| 2026-10-05 19:01 | run directory `sci_paper/runs_ws_extra/manifest_v2.json` starts | `sci_paper/runs_ws_extra/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-07 15:01 | run directory `sci_paper/_x_old/runs_m6/manifest_v2.json` starts | `sci_paper/_x_old/runs_m6/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-07 17:10 | run directory `sci_paper/runs_m5/manifest_v2.json` starts | `sci_paper/runs_m5/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-07 17:22 | `massive_mt_deepl` corpus generated (third translation lineage) | corpus mtime of the current file; this corpus was not re-patched |
| 2026-10-07 20:27 | run directory `sci_paper/_x_old/runs_v4d/manifest_v2.json` starts | `sci_paper/_x_old/runs_v4d/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-08 02:29 | run directory `sci_paper/_x_part2/runs_v4d/manifest_v2.json` starts | `sci_paper/_x_part2/runs_v4d/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-08 08:41 | `_x_mt_ext` de-DE corpus generated | corpus mtime, pre-patch backup |
| 2026-10-08 08:47 | `_x_mt_ext` vi-VN corpus generated | corpus mtime, pre-patch backup |
| 2026-10-08 08:52 | `_x_mt_ext` ja-JP corpus generated | corpus mtime, pre-patch backup |
| 2026-10-08 10:12 | run directory `sci_paper/_x_diag/runs_diag/manifest_v2.json` starts | `sci_paper/_x_diag/runs_diag/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-08 13:29 | run directory `sci_paper/_x_bi/runs_bi/manifest_v2.json` starts | `sci_paper/_x_bi/runs_bi/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-08 13:53 | run directory `sci_paper/_x_ws/runs_bi_ws_packed/manifest_v2.json` starts | `sci_paper/_x_ws/runs_bi_ws_packed/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-08 14:05 | run directory `sci_paper/_x_ws/runs_bi_ws_both/manifest_v2.json` starts | `sci_paper/_x_ws/runs_bi_ws_both/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-08 14:13 | run directory `sci_paper/_x_ws/runs_bi_mo_packed/manifest_v2.json` starts | `sci_paper/_x_ws/runs_bi_mo_packed/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-08 14:18 | run directory `sci_paper/_x_ws/runs_bi_mo_both/manifest_v2.json` starts | `sci_paper/_x_ws/runs_bi_mo_both/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-08 15:59:55 | pre-registration of the confirmation predictions deposited, commit `b676eae1e151da632b12fb045de4920397adf725` | the commit itself; `preregistration/PREDICTION_confirm_3locales_KO.md`, `_ZH.md`, `preregistration/README.md` |
| 2026-10-08 17:09 | run directory `sci_paper/_x_m4p2/runs_v4d/manifest_v2.json` starts | `sci_paper/_x_m4p2/runs_v4d/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| 2026-10-08 19:13 | run directory `sci_paper/_x_conf_colab/runs_confirm/manifest_v2.json` starts | `sci_paper/_x_conf_colab/runs_confirm/manifest_v2.json` (`started_at`; manifest timestamps are UTC, shown here converted to +0800) |
| [to be supplied] | human-evaluation sheets returned by the three judges | **not recorded in this workspace.** The completed sheets carry no return timestamp and no dated covering message survives, so this row is left as an explicit placeholder rather than estimated. |

**Reading the table.** Every row is a file timestamp or a version-control timestamp that a
reader can reproduce from the deposit. `manifest_v2.json` records its timestamps in UTC; the
column above converts them to local time (+0800) so that they can be compared with the corpus
mtimes and the commit timestamp, which are already local. The order of the run and corpus rows
relative to the pre-registration is the point of the table: the `de-DE`, `vi-VN` and `ja-JP`
corpora were generated on 2026-10-08 between 08:41 and 08:52, and the predictions were deposited
at 15:59:55 the same day, **after those corpora existed and before the confirmation runs that
consumed them**. The deposit therefore fixes the predictions ahead of the *runs*, not ahead of
the *corpora*, and we state that distinction rather than presenting the deposit as preceding
everything downstream of it.

**What the table does not establish.** It does not date the exploratory runs of 2026-10-03 to
2026-10-05 against any external event, because no such event is recorded; it does not record
when each individual run inside a directory finished (the manifest holds one `finished_at` per
directory, not per run); and the human-evaluation row is a placeholder for the reason given in it.

**One row is deliberately blank.** The date on which the three judges returned their completed
sheets is not recoverable from anything in this deposit, and we leave `[to be supplied]` rather
than infer a date from the analysis scripts that read them. The human-evaluation numbers
themselves do not depend on it.

---

## S14. Pre-registration of the confirmation experiment

The manuscript's headline quantity — the deployment overstatement, `MT→MT − MT→L` — comes from a cell
added after the prespecified intrinsic-difficulty test failed. To move that result from exploratory to
confirmatory, the **sign and range** of the overstatement and of the test-side cost were fixed in
writing, and deposited in a public version-controlled record, **before any of the confirmation runs**.

| Field | Value |
| --- | --- |
| Repository | `https://github.com/why461642446/mt-testset-provenance` |
| Commit | `b676eae1e151da632b12fb045de4920397adf725` |
| Commit timestamp | `2026-10-08 15:59:55 +0800` |
| Timeline and pre-registration order | Section S13 (the commit hash, the timestamp and the reading rule) |
| Deposited content | `preregistration/PREDICTION_confirm_3locales_KO.md` (submitted to the advisor), `preregistration/PREDICTION_confirm_3locales_ZH.md`, `preregistration/README.md` |
| Locales | `de-DE`, `vi-VN`, `ja-JP` (MASSIVE v1.1; all supported by NLLB) |
| Classifiers | TextCNN, XLM-R base, and the character n-gram lookup baseline |

**Deposit dates.** The `de-DE` and `vi-VN` predictions were submitted to the advisor on 2026-10-08 and
are contained in the commit above. The `ja-JP` predictions (P7–P9) were appended to the same document
**after that submission and before any `ja-JP` run**; their later authorship is stated in the document
itself and is visible in the record's own history rather than being presented as contemporaneous.

**Two registration layers, and what the second one changes.** The confirmation claims in this paper rest on two deposits, and the difference between them decides how the result may be read.

| Layer | Deposit | Quantity fixed in advance | What it licenses |
| --- | --- | --- | --- |
| **First** (this section) | commit `b676eae1e151da632b12fb045de4920397adf725`, 2026-10-08 15:59:55 +0800 | the **same-system** overstatement `MT→MT − MT→L` with its range, and the test-side cost, on `de-DE`, `vi-VN` and `ja-JP` | a **sign-and-range replication of a quantity the manuscript had already reported** in Table 5m: it removes the objection that the `MT→L` cell was chosen for the number it returned. **Its own reading rule was a replicated sign**; the revised registration replaces that criterion for the new layer |
| **Second** (the revised registration) | deposited before the new runs, with its hash and timestamp recorded in the deposit itself | the **cross-system** overstatement `MT→MT2 − MT→L` as the primary quantity; the auxiliary mechanism predictions P3·P4·P7–P10; the locale-specific maximum lengths `73` (`de-DE`) and `72` (`vi-VN`); and a corrected training count (**one model per training source, not one per cell**, because the deployment comparison scores the same translated-trained model on both test sets) | the **only prospective quantity**, because the same-system values were already on record when the first layer's runs finished; promotion of the headline is decided by the interval rule above, on the two-level bootstrap, rather than by a replicated sign |

**What the two layers have in common** is the reading rule: predictions that fail are reported as failures, and no prediction is revised after a run. **What the second layer changes** is (i) the quantity that carries confirmatory weight — `MT→MT2 − MT→L` is computed from a 3.3B test-set translation that did not exist when the first layer was deposited — and (ii) the criterion, from "the sign replicated" to "the fixed interval excluded zero". **The mechanism predictions remain auxiliary** and are reported as exploratory observations, never as separate confirmatory conclusions. **P4 is already known to fail on `de-DE`** (`+4.52` against `+14.44`, a gap of `−9.92`), which is why the manuscript bounds its string-lookup claim to the locales where it holds (Section 4.2.2a).

**Reading rule, fixed in advance.** Predictions that fail are reported as failures. No prediction in
this deposit may be revised after a run; corrections, if any, are appended with their own date.
**The commit hash and timestamp are the evidence that the predictions precede the results**, and a
reader can verify both without trusting this manuscript.

**Scope of the claim this supports, and the criterion that authorises it.** A replicated sign is not the
criterion. **The criterion is the interval rule fixed in the revised registration** (the second layer set
out above), which follows the advisor's rule for judging a confirmation: the abstract, the title and
contribution C2 may state the effect without the `post hoc` qualifier only if, **on every confirmation
locale, the 95% interval of the test-side cost `L→MT − L→L` lies entirely below zero (P1) and the
95% interval of the cross-system overstatement `MT→MT2 − MT→L` lies entirely above zero (P2)** —
the numbering is that of the revised registration, not of the table below — under the two-level
bootstrap over seeds and sentences fixed in advance. **An interval that straddles zero leaves the result
undetermined and the exploratory framing stays**; **an interval that lies entirely on the opposite side
is a failed prediction**, and this section records the discrepancy rather than the ordering that
happened to match.

**Outcome.** All six predictions have now been evaluated. The sign of the overstatement
replicated on every locale and on both model families. **Three of the six predictions failed,
and we record the failures rather than the successes.**

| Prediction | Fixed in advance | Observed (fixed evaluation subset) | Verdict |
| --- | --- | --- | --- |
| P2 | Overstatement positive on all three new locales (TextCNN) | `de` +14.44, `vi` +14.23, `ja` +6.75 | **Holds** |
| P3 | XLM-R base overstatement below TextCNN's | gap positive in all five locales: 9.75, 7.72, 10.99, 10.56, 7.69 | **Holds** |
| P5 | `de` below `vi` | `de` +14.44 **>** `vi` +14.23 | **Fails** |
| P7 | `ja` positive and intermediate | `ja` +6.75 — positive, but the smallest of the five | **Partly holds** |
| P8 | absolute `ja` cost above absolute `de` cost | TextCNN 15.69 < 22.77; XLM-R 9.99 > 3.77 | **Mixed** |
| P9 | `ja` closer to `ko` than to `zh` | 4.73 against 0.77 (TextCNN overstatement) | **Fails** |

**Cost magnitude.** Predicted ranges were `de` −8 to −15, `vi` −12 to −20, `ja` −14 to −22.
Observed: `de` **−22.77** (outside), `vi` **−12.90** (inside), `ja` **−15.69** (inside) — **2 of 3**.

**Overstatement magnitude.** Predicted ranges were `de` +1 to +6, `vi` +3 to +9, `ja` +4 to +10.
Observed: `de` **+14.44** (outside), `vi` **+14.23** (outside), `ja` **+6.75** (inside) — **1 of 3**.

**The failures are the informative part.** We predicted the effect would be *smaller* on the two
space-delimited languages, on the grounds that the Chinese collapse of Section 4.3 was a whitespace
defect. It is not: `de-DE` (+14.44) and `vi-VN` (+14.23) both exceed `ko-KR` (+11.48), and the
smallest value of the five is `ja-JP`, a language with no whitespace at all. **Whitespace
delimitation does not predict the size of the effect, and the manuscript makes no such claim.**

**P5 is a borderline failure, and the failure is a change of evaluation basis.** On the full test
split the ordering was `de` +14.48 < `vi` +15.47, as predicted. On the fixed evaluation subset used
throughout this paper it reverses to **+14.44 > +14.23**. We report the reversal rather than the
ordering that happened to match, take the fixed subset because it is the basis on which every other
result here is computed, and note that a 0.21-point margin is not a difference we would defend in
either direction.

**What the confirmation licenses, and what it does not.** Because the sign replicated on five
locales, two model families and two evaluation bases under predictions deposited before any run, the
first layer licenses the statement that **the sign and range of the same-system overstatement were
fixed before the runs and replicated**. **The qualifier-free statement in the abstract and
contribution C2 is carried by the interval criterion of the second layer**, and it is withdrawn if
that criterion is not met. **The `MT→L`
cell remains an addition made after the prespecified test failed**, and Section 3 still records it
as such; what the confirmation removes is the objection that the cell was chosen because it
returned a particular number. It does not license a claim about the *magnitude*, which varied by
more than a factor of two across the five locales and did not follow the ordering we predicted.

---

## S15. The P8 / P9 / P10 mechanism predictions on ko-KR and zh-CN — exploratory

**What this section is, and what it is not.** The confirmation experiment registered three auxiliary mechanism predictions — P8 (residual overstatement after near-duplicate removal), P9 (overstatement by quartile of English-source similarity) and P10 (overstatement with and without a localized slot) — and the reviewer's instruction was that **the Korean and Chinese panels be recomputed under the same conventions and reported alongside**, with the explicit statement that **"the Korean/Chinese part is an exploratory result"**. This section is that recomputation. **All three panels are exploratory: none of them was pre-registered, and none of them carries confirmatory weight.** The prospective versions of P8, P9 and P10 remain the `de-DE`, `vi-VN` and `ja-JP` ones deposited in the registration of S14.

**Basis.** The classifier panels here are on the **fixed evaluation subset** (ko-KR 2,662 / zh-CN 2,608), the same basis as every other classifier number in this paper, and the intervals are 95% percentile intervals from 2,000 paired bootstraps over sentences. The **one** P10 result that is not on that basis is the slot-locality **share**, which is a field count over the **full test split of 2,974** because that is the basis on which the predicted shares were given; the basis is named in that table's title and again in S15-iv.

### S15-i. P8 — residual overstatement after near-duplicate removal

**Prediction.** The overstatement survives removal of every test sentence with a cosine of 0.7, 0.8 or 0.9 or more against its training cell, 0.8 primary. **Criterion fixed in advance: the lower bound of the 95% interval is above zero.**

| Locale | t | Retained | Model | Overstatement | Residual | 95% interval | Lower bound > 0 | Four-cell residual | 95% interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ko-KR | 0.7 | 2,115 (20.5% removed) | TextCNN | +11.48 | **+9.86** | [8.26, 11.46] | yes | +10.36 | [8.63, 12.02] |
| ko-KR | 0.7 | | BiLSTM | +12.26 | **+10.51** | [8.82, 12.14] | yes | +10.86 | [9.14, 12.59] |
| ko-KR | 0.7 | | mBERT | +6.86 | **+5.94** | [4.48, 7.47] | yes | +5.76 | [4.16, 7.37] |
| ko-KR | 0.7 | | XLM-R base | +1.73 | **+1.15** | [−0.25, 2.47] | **no** | +1.07 | [−0.38, 2.46] |
| ko-KR | 0.8 | 2,433 (8.6%) | TextCNN | +11.48 | **+10.51** | [8.98, 11.98] | yes | +10.83 | [9.29, 12.34] |
| ko-KR | 0.8 | | BiLSTM | +12.26 | **+11.12** | [9.57, 12.62] | yes | +11.28 | [9.62, 12.88] |
| ko-KR | 0.8 | | mBERT | +6.86 | **+6.27** | [4.89, 7.67] | yes | +6.31 | [4.85, 7.73] |
| ko-KR | 0.8 | | XLM-R base | +1.73 | **+1.48** | [0.23, 2.69] | yes | +1.58 | [0.35, 2.83] |
| ko-KR | 0.9 | 2,608 (2.0%) | TextCNN | +11.48 | **+11.25** | [9.80, 12.70] | yes | +11.29 | [9.83, 12.70] |
| ko-KR | 0.9 | | BiLSTM | +12.26 | **+11.93** | [10.47, 13.39] | yes | +11.99 | [10.44, 13.57] |
| ko-KR | 0.9 | | mBERT | +6.86 | **+6.71** | [5.34, 8.05] | yes | +6.72 | [5.36, 8.14] |
| ko-KR | 0.9 | | XLM-R base | +1.73 | **+1.65** | [0.46, 2.81] | yes | +1.84 | [0.64, 3.01] |
| zh-CN | 0.7 | 2,111 (19.1%) | TextCNN | +7.52 | **+6.91** | [5.38, 8.49] | yes | +7.33 | [5.83, 8.90] |
| zh-CN | 0.7 | | BiLSTM | +8.11 | **+7.44** | [5.94, 8.99] | yes | +8.15 | [6.57, 9.63] |
| zh-CN | 0.7 | | mBERT | +2.39 | **+1.75** | [0.47, 3.02] | yes | +1.95 | [0.56, 3.29] |
| zh-CN | 0.7 | | XLM-R base | −0.20 | **−0.58** | [−1.75, 0.54] | **no** | −0.49 | [−1.70, 0.70] |
| zh-CN | 0.8 | 2,393 (8.2%) | TextCNN | +7.52 | **+6.94** | [5.53, 8.38] | yes | +6.93 | [5.53, 8.31] |
| zh-CN | 0.8 | | BiLSTM | +8.11 | **+7.51** | [6.15, 8.83] | yes | +7.66 | [6.25, 9.07] |
| zh-CN | 0.8 | | mBERT | +2.39 | **+1.99** | [0.82, 3.13] | yes | +1.98 | [0.77, 3.18] |
| zh-CN | 0.8 | | XLM-R base | −0.20 | **−0.38** | [−1.39, 0.71] | **no** | −0.37 | [−1.43, 0.75] |
| zh-CN | 0.9 | 2,545 (2.4%) | TextCNN | +7.52 | **+7.25** | [5.98, 8.61] | yes | +7.19 | [5.83, 8.51] |
| zh-CN | 0.9 | | BiLSTM | +8.11 | **+7.85** | [6.58, 9.13] | yes | +7.84 | [6.48, 9.20] |
| zh-CN | 0.9 | | mBERT | +2.39 | **+2.15** | [1.03, 3.30] | yes | +2.09 | [0.92, 3.15] |
| zh-CN | 0.9 | | XLM-R base | −0.20 | **−0.25** | [−1.26, 0.76] | **no** | −0.24 | [−1.28, 0.79] |

*Verdict on the Korean/Chinese panel.* **On Korean at t = 0.8 the lower bound is above zero for all four models** (+10.51, +11.12, +6.27, +1.48); at t = 0.7 XLM-R base's interval straddles zero (+1.15 [−0.25, 2.47]). **On Chinese, TextCNN, BiLSTM and mBERT clear the criterion at all three thresholds and XLM-R base clears it at none of them** (−0.58, −0.38, −0.25). That is not a removal artefact: **XLM-R base has no overstatement on Chinese to begin with** (its unremoved value is −0.20), so "residual overstatement" is not a quantity it can have. We state that rather than presenting four models as if they behaved alike.

### S15-ii. P9 — overstatement by quartile of English-source similarity

**Prediction.** Partitioning test sentences by the nearest-neighbour similarity of their **English source sentence** to the **English training set**, the overstatement in Q4 exceeds that in Q1. The demonstration values given with the instruction were **Korean +2.6 → +12.3** and **Chinese +4.0 → +11.7** (Q1 → Q4). The criterion is deliberately independent of the translation and of the localized text, so the same partition applies in every locale.

**⚠️ A substantive downgrade, stated before the numbers.** The prediction as deposited says *"predict the overstatement of **LR**"* — the character n-gram logistic regression of §3.7. **This repository contains no per-sentence LR predictions.** Every per-sentence file in the deposit was enumerated (all **361** `pred_*.csv` names) and **not one is an LR file**, so P9 cannot be recomputed on the model the prediction names. **The panel below therefore substitutes four neural models** — TextCNN, BiLSTM, mBERT and XLM-R base — **which share the similarity standard but are not the model under prediction.** This is a **substitution for the demonstration value, not a reproduction of it**, and it is reported here rather than silently presented as the requested analysis. Recovering the intended quantity would require re-running the LR baseline and storing its per-sentence predictions, which this round did not do.

Basis: fixed evaluation subset; similarity is the character 1–3-gram TF-IDF cosine of the English source sentence against the **English training set** (11,514 sentences), pooled fit, ties broken by the lowest training index. Quartile cuts: ko-KR 0.539 / 0.648 / 0.763; zh-CN 0.538 / 0.646 / 0.757.

**Table S15-i. Overstatement by English-source similarity quartile** (fixed evaluation subset; Q1 is the least similar quartile).

| Locale | Model | Q1 | Q2 | Q3 | Q4 | Q4 − Q1 |
| --- | --- | --- | --- | --- | --- | --- |
| ko-KR | TextCNN | +5.62 | +11.01 | **+15.19** | +14.11 | **+8.49** |
| ko-KR | BiLSTM | +5.71 | +11.52 | **+16.09** | +15.74 | **+10.03** |
| ko-KR | mBERT | +1.65 | +6.32 | **+11.33** | +8.16 | **+6.51** |
| ko-KR | XLM-R base | −2.05 | +2.36 | **+3.81** | +2.80 | **+4.85** |
| zh-CN | TextCNN | +5.61 | +3.96 | +8.83 | **+11.69** | **+6.08** |
| zh-CN | BiLSTM | +7.24 | +5.37 | +7.55 | **+12.27** | **+5.03** |
| zh-CN | mBERT | −1.58 | +0.72 | +3.78 | **+6.65** | **+8.23** |
| zh-CN | XLM-R base | −3.48 | −0.82 | +1.43 | **+2.04** | **+5.52** |

**Against the demonstration values.** **On Chinese our TextCNN panel is close to the demonstration**: +5.61 → +11.69 against **+4.0 → +11.7**, with Q4 almost digit for digit. **On Korean the levels differ**: +5.62 → +14.11 against **+2.6 → +12.3**, our Q1 being three points higher and our Q4 nearly two points higher. The intervals on the four Korean TextCNN quartiles are [2.73, 8.44], [8.06, 13.96], [12.36, 18.26] and [11.41, 16.73]; on Chinese [2.94, 8.41], [1.26, 6.50], [6.07, 11.63] and [9.20, 14.26].

**⚠️ The direction holds; "monotonically increasing" does not, and we report that rather than the summary that would fit.** **All four models on both locales have Q4 above Q1** — by 8.49, 10.03, 6.51 and 4.85 points on Korean and by 6.08, 5.03, 8.23 and 5.52 on Chinese — so the direction the prediction asserts is reproduced. **But the sequence is not monotone on Korean: all four models peak in Q3 and fall back in Q4** (TextCNN 5.62 → 11.01 → 15.19 → 14.11; BiLSTM 5.71 → 11.52 → 16.09 → 15.74; mBERT 1.65 → 6.32 → 11.33 → 8.16; XLM-R base −2.05 → 2.36 → 3.81 → 2.80). On Chinese, TextCNN and BiLSTM dip in Q2 and then rise to a Q4 peak, and mBERT and XLM-R base rise without decreasing; the Chinese panel is therefore *consistent* with the demonstration's shape while the Korean panel is not. **"The most similar quartile is where the effect is largest" is false on Korean, under a partition that does not involve the translation at all.** The Q1 and Q4 groups are disjoint sets of sentences, so the difference is read against each group's own interval rather than as a paired interval.

**This is not a binning artefact, and we can show that from the manuscript's own table.** The same partition, computed on the published implementation's basis, reproduces **all sixteen numbers of Table S9-ii digit for digit** (ko-KR −19.22 / +5.62, −21.29 / +11.01, −21.98 / +15.19, −22.82 / +14.11; zh-CN −12.12 / +5.61, −11.50 / +3.96, −14.08 / +8.83, −15.00 / +11.69). **So the peak in Q3 and the fall in Q4 are in the table this paper already prints**: the text of §4.2.3 and of S9-i/S9-ii reads the bins as cost and overstatement rising together with similarity, and on Korean the overstatement's maximum is in Q3, not Q4. The only implementation difference is the bin sizes (the explicit half-open convention gives 666 / 665 / 665 / 666 on Korean where `np.digitize` gives 666 / 666 / 665 / 665; Chinese is 652 × 4 under both), and **that difference changes none of the sixteen values**. **We record the discrepancy between the table and the sentence that reads it rather than reconciling them in the direction of the sentence.**

### S15-iii. P10 — overstatement with and without a localized slot

**Prediction.** On the localized sentences that carry **at least one slot whose value was localized rather than translated** — the mechanism being that a city or a name replaced by a locale-appropriate entity does not appear in the translated training set, so `MT→L` falls and the overstatement grows — the overstatement is larger; and the open question registered alongside it is **whether the overstatement still exists in sentences with no localized slot at all**.

**Table S15-ii. Overstatement by whether the sentence's localized text carries a localized slot** (fixed evaluation subset: ko-KR 2,662, of which **658 (24.7%)** carry a localized slot; zh-CN 2,608, of which **639 (24.5%)** do).

| Locale | Model | All | With a localized slot | 95% interval | Lower bound > 0 | **Without any localized slot** | 95% interval | Lower bound > 0 | Difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ko-KR | TextCNN | +11.48 | +14.62 | [11.67, 17.57] | yes | **+10.45** | [8.86, 12.11] | **yes** | +4.17 |
| ko-KR | BiLSTM | +12.26 | +16.29 | [13.37, 19.42] | yes | **+10.94** | [9.24, 12.65] | **yes** | +5.35 |
| ko-KR | mBERT | +6.86 | +8.41 | [5.52, 11.25] | yes | **+6.35** | [4.84, 7.90] | **yes** | +2.06 |
| ko-KR | XLM-R base | +1.73 | +2.08 | [−0.31, 4.36] | no | **+1.61** | [0.32, 2.96] | **yes** | +0.47 |
| zh-CN | TextCNN | +7.52 | +12.55 | [9.73, 15.59] | yes | **+5.89** | [4.40, 7.37] | **yes** | +6.66 |
| zh-CN | BiLSTM | +8.11 | +12.74 | [9.86, 15.59] | yes | **+6.60** | [5.19, 8.07] | **yes** | +6.14 |
| zh-CN | mBERT | +2.39 | +4.54 | [2.09, 6.78] | yes | **+1.69** | [0.41, 2.96] | **yes** | +2.85 |
| zh-CN | XLM-R base | −0.20 | −0.94 | [−3.29, 1.36] | no | **+0.03** | [−1.17, 1.22] | **no** | −0.97 |

*Verdict on the Korean/Chinese panel, and the sentence we would write.* **The overstatement is larger in sentences that carry a localized slot, on seven of the eight model–locale pairs** (Korean +4.17, +5.35, +2.06, +0.47; Chinese +6.66, +6.14, +2.85, −0.97), **so the mechanism the prediction names is present and points the right way.** **But it does not account for the overstatement**: in sentences with **no** localized slot the overstatement is still there, with its lower bound above zero for **all four Korean models** and for Chinese TextCNN, BiLSTM and mBERT. Chinese XLM-R base is the exception on both sides (+0.03 [−1.17, +1.22] without a slot; −0.94 with one), which is consistent with that model having no Chinese overstatement at all. **The right summary is that entity localization is a real contributor and not the whole of the effect** — which is what the confirmation prediction needs to say, and which the Korean and Chinese panels support.

**Table S15-iii. The cost side of the same split**, reported for completeness (fixed evaluation subset; cells are `L→MT − L→L` with their intervals).

| Locale | Model | Without any localized slot | 95% interval | With a localized slot | 95% interval |
| --- | --- | --- | --- | --- | --- |
| ko-KR | TextCNN | −21.32 | [−23.13, −19.54] | −21.37 | [−24.44, −18.24] |
| ko-KR | BiLSTM | −22.86 | [−24.66, −21.14] | −25.32 | [−28.12, −22.37] |
| ko-KR | mBERT | −16.70 | [−18.35, −14.90] | −20.26 | [−23.15, −17.22] |
| ko-KR | XLM-R base | −11.54 | [−12.97, −10.01] | −10.54 | [−12.97, −7.90] |
| zh-CN | TextCNN | −11.46 | [−12.96, −9.95] | −18.47 | [−21.47, −15.34] |
| zh-CN | BiLSTM | −14.17 | [−15.78, −12.59] | −21.25 | [−24.38, −18.25] |
| zh-CN | mBERT | −8.16 | [−9.55, −6.84] | −12.15 | [−15.08, −9.34] |
| zh-CN | XLM-R base | −6.25 | [−7.50, −5.01] | −10.90 | [−13.56, −8.24] |

### S15-iv. The slot-method shares, and why counting them on de-DE and vi-VN is not a breach of the pre-registration bar

**The measured shares reproduce the predicted ones exactly.** The prediction gives **ko-KR 23.0%, zh-CN 23.6%, de-DE 13.0%, vi-VN 19.5%** for the share of test sentences with at least one localized slot.

**Table S15-iv. Share of test sentences with at least one localized slot** — a **field count over the full test split of 2,974** (and, in the last column, over the 11,514-sentence training split), not a classifier result and not a similarity statistic.

| Locale | Predicted (test) | Measured (test, n = 2,974) | Match | Training split (n = 11,514) |
| --- | --- | --- | --- | --- |
| ko-KR | 23.0% | **685 / 2,974 = 23.0%** | **yes, digit for digit** | 2,710 / 11,514 = 23.5% |
| zh-CN | 23.6% | **701 / 2,974 = 23.6%** | **yes, digit for digit** | 2,787 / 11,514 = 24.2% |
| de-DE | 13.0% | **387 / 2,974 = 13.0%** | **yes, digit for digit** | 1,522 / 11,514 = 13.2% |
| vi-VN | 19.5% | **581 / 2,974 = 19.5%** | **yes, digit for digit** | 2,259 / 11,514 = 19.6% |

**Why this is not a breach of the instrumentation bar.** The instruction for the registration period is that **no classifier result, similarity statistic or chrF may be computed on `de-DE` or `vi-VN` before deposit**. This count does none of those three things: it reads the **`slot_method` text field** of the official MASSIVE v1.1 `de-DE.jsonl` and `vi-VN.jsonl` files, **does not read a single machine-translated string, does not open a single `pred_*.csv`, does not fit a similarity model and does not compute chrF.** It is also not a new measurement of the phenomenon — **it is the same statistic the instruction itself supplied** (13.0% and 19.5%), recomputed in order to check it. Counting a provided statistic to verify it is not computing a classifier result on `de-DE` or `vi-VN`. **Every de-DE, vi-VN and ja-JP classifier result, similarity statistic and chrF in this supplementary material remains zero.**

**The slot-method distribution, and one loose end** (test split, counted over slot entries rather than sentences):

| Locale | translation | localization | unchanged | other |
| --- | --- | --- | --- | --- |
| ko-KR | 1,947 | 721 | 75 | — |
| zh-CN | 1,974 | 749 | 20 | — |
| de-DE | 1,738 | 401 | 603 | `unchanged_translation` × 1 |
| vi-VN | 1,792 | 618 | 332 | `unchanged_translation` × 1 |

**The field is not strictly three-valued.** `de-DE` and `vi-VN` each carry **one entry whose method is `unchanged_translation`** — a fourth value that is neither of the three named in the description — while Korean and Chinese carry none. We count only `localization` toward "has a localized slot", which is the convention the predicted shares use, so the four matches above are unaffected; but any downstream method description should leave room for the fourth value rather than asserting that the field takes exactly three.

### S15-v. What is exploratory here, and what may be quoted

| Item | Why it is exploratory |
| --- | --- |
| S9-iii / S9-iv / S9-v (all of them) | Post-hoc stratification and recomputation on Korean and Chinese; the instruction asked for the *prediction* in the deposit, and these are the "recompute and report under the same conventions" companion, not registered content |
| **P8, P9, P10 on ko-KR and zh-CN** | The instruction says so explicitly: the Korean and Chinese part is an exploratory result. What was to be **predicted** is the `de-DE` / `vi-VN` / `ja-JP` panel of S14 |
| **P9's four-model panel** | Both exploratory **and** a substitution for the demonstration value — there are no LR per-sentence predictions to recompute it on |
| **P10's `de-DE` / `vi-VN` shares** | A **check** of values supplied with the instruction, not a new measurement |
| The length gradient in S9-iv | A pattern that has not appeared in this paper before and was not pre-registered; offered for the confirmation experiment, not asserted |

**What may be quoted as agreeing with the supplied values**: the four slot-method shares (digit for digit); P9's **direction** (Q4 above Q1 on all four models and both locales) and the **Chinese level**; S9-v's four-cell figure at t = 0.8 agreeing with the manuscript's +10.83 digit for digit; and the sixteen values of Table S9-ii. **What may not**: P9's Korean levels or its monotonicity, the length gradient, and anything in S9-iii to S9-v as a confirmatory finding.

---

