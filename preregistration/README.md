# Pre-registration — MT-test-set provenance confirmation experiment

**Deposited:** 2026-10-08 (this commit)
**Author:** Haoyi Wang (WANG HAOYI), PhD candidate
**Advisor:** MinPo Jung
**Response to:** advisor review §2.5, option (1) — keep the generalization claim and run a
confirmation experiment on further NLLB-supported MASSIVE locales.

## What is pre-registered here

The manuscript's headline quantity — the **deployment overstatement**, defined as
`MT→MT − MT→L` on a fixed evaluation subset — comes from a cell that was added *after* the
prespecified intrinsic-difficulty test failed. This deposit fixes, before any run, the
**sign and range** of that quantity and of the test-side cost (`L→MT − L→L`) on three new
locales.

| File | Language | Content |
| --- | --- | --- |
| `PREDICTION_confirm_3locales_KO.md` | Korean | Submitted to the advisor 2026-10-08 |
| `PREDICTION_confirm_3locales_ZH.md` | Chinese | Same content, working copy |

## Locales

`de-DE`, `vi-VN`, `ja-JP` — all three are provided by MASSIVE v1.1 and supported by NLLB.

- `de-DE`, `vi-VN`: space-delimited, so they test whether the effect is independent of the
  whitespace-segmentation defect.
- `ja-JP`: CJK without whitespace, so it tests within-CJK consistency against `ko-KR`.

## Classifiers

Per the advisor's §2.5 instruction: **TextCNN**, **XLM-R base**, and the **character n-gram
lookup baseline** (no trained weights).

## Reading rule fixed in advance

Predictions that fail will be reported as failures. No prediction in this deposit may be
revised after a run; corrections, if any, are appended with their own date.
