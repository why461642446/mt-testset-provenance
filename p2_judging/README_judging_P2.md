# Judge instructions (please read before starting)

## What you are doing

You have English sentences that have been **machine-translated** into Korean or Chinese. Your task is to judge **whether the translation preserves the intent** of the original.

"Intent" means what the sentence asks a device to do. For example, the English `play some jazz` has the intent `PlayMusic`, and `turn off the bedroom light` has the intent `iot_hue_lightoff`.

**You are not judging whether the translation is good, fluent, or elegant.** Only whether the intent changed.

## What you see

| Column | Content |
| --- | --- |
| `english_source` | the English source sentence |
| `target_text` | the machine translation |
| `intent_label` | the intent label of the English source |

The matching rows of `P2_ko_key.csv` / `P2_zh_key.csv` (withheld from you) record the group
assignment and the source-sentence identity, and pair duplicates by `src_id`. See the column
glossary and the archive note at the end of this file.

The matching rows of `P2_ko_key.csv` / `P2_zh_key.csv` (withheld from you) record the same
`order` and `item_id` plus the group assignment and the source-sentence identity; duplicates
are paired there by `src_id`. See the column glossary in the analysis README.

**You do not see** the professional localization, any model's prediction, or which items are duplicates. That is deliberate.

## Three categories (enter A / B / C only)

- **A — intent and meaning preserved.** A native reader recovers the source meaning; the intent is clear.
- **B — intent preserved, content differs.** The intent is still right, but words are omitted, added, or generalised.
- **C — intent changed.** The translation would be read as a different intent, or is unintelligible.

## How to fill it in

1. Open `P2_ko_blind.csv` or `P2_zh_blind.csv` (Excel opens it directly; UTF-8 with BOM, so Korean and Chinese display correctly).
2. Enter **A / B / C** in the `judge1_category` column — **the single letter only**.
3. If you are unsure, give your best judgement and add one sentence in `judge1_note`.
4. **Complete all 270 rows.** Do not skip any.

## Working independently

- Each judge works **independently**; do not discuss answers and do not look at each other's columns.
- Each person uses **one fixed column** (`judge1`, `judge2` or `judge3`).

## On duplicates

**20 of the 270 rows are duplicates** — the same sentence appears under two different `item_id`s. This is deliberate and measures consistency.

**Do not look for them and do not try to be consistent.** Judge each row as you see it.

## Column glossary for the key files

| Column | Meaning |
| --- | --- |
| `order` | presentation order in the blind sheet |
| `item_id` | the identifier the judge sees; `ko-001`-`ko-270` and `zh-001`-`zh-270` were assigned after shuffling |
| `group` | `uniform` for the 250 sampled items, `repeat_of_<n>` for the 20 duplicated items |
| `src_id` | **the pairing key.** The MASSIVE test-row index of the underlying sentence |
| `is_flip` | 1 where the duplicate is the flipped pair member |
| `intent_label`, `english_source`, `localized_text`, `target_text` | the underlying text, for the analysis only |

## How the duplicated items are paired

**The 20 duplicated items are paired with their baseline by `src_id`, not by `item_id`.**
Each duplicated sentence appears twice in the sheet under two different `item_id` values, and
the two rows share one `src_id`. To recover a pair, join the key to itself on `src_id` and keep
the `src_id` values that occur more than once: exactly 20 do, in each locale, and each occurs
exactly twice. Any analysis of within-judge consistency must use that join.

**The `N` in a `repeat_of_N` label is an internal sequence number and must not be read as an
`item_id`.** The label was generated while the duplicate pairs were being built, before the
presentation-order `item_id` values were assigned, so `repeat_of_232` does not mean "the
duplicate of `ko-232`" — no such pairing exists. For example, `ko-006` carries
`group = repeat_of_232` and is paired with `ko-013`; the two share `src_id = 15968` and neither
is `ko-232`. The labels are retained unchanged because the deposited sheet was built from them,
but they carry no pairing information. `src_id` is the only pairing key in this directory.

## The key files are not for judges

`P2_ko_key.csv` and `P2_zh_key.csv` hold the group assignment, the source-sentence identity and
the flip flag. **They are withheld from the judges during evaluation**: a judge who could see
`group` would know which rows are duplicates, and a judge who could see `src_id` could look up
the sentence already judged. They were previously named `P2_ko_key_PRIVATE.csv` /
`P2_zh_key_PRIVATE.csv`; the `_PRIVATE` suffix was dropped because it reads as a variable name
rather than as a handling instruction, and the prohibition is stated here instead.

## What is archived, and what each judge actually received

**The archive releases the consolidated sheet only.** `P2_ko_blind.csv` and
`P2_zh_blind.csv` are that sheet: 270 rows each, carrying all three judges'
`judge1/2/3_category` and `judge1/2/3_note` columns side by side.

**Each judge received a separate file containing only seven columns** — `order`,
`item_id`, `english_source`, `target_text`, `intent_label`, and that judge's own
`judgeN_category` and `judgeN_note`. No judge could see another judge's column,
which is what makes the three judgements independent rather than negotiated.
The per-judge files are **not** deposited: they are the consolidated sheet with
columns removed, so they carry no information the consolidated sheet does not,
and shipping blank templates alongside the answers invites exactly the wrong
reading.

**The judges consented to the publication of their de-identified scores.**
The release identifies them only as judge 1, 2 and 3 and records no name,
affiliation or other identifier.

## Column glossary for the key files

| Column | Meaning |
| --- | --- |
| `order` | presentation order in the blind sheet |
| `item_id` | the identifier the judge sees; `ko-001`-`ko-270` and `zh-001`-`zh-270` were assigned after shuffling |
| `group` | `uniform` for the 250 sampled items, `repeat_of_<n>` for the 20 duplicated items |
| `src_id` | **the pairing key.** The MASSIVE test-row index of the underlying sentence |
| `is_flip` | 1 where the duplicate is the flipped pair member |
| `intent_label`, `english_source`, `localized_text`, `target_text` | the underlying text, for the analysis only |

## How the duplicated items are paired

**The 20 duplicated items are paired with their baseline by `src_id`, not by `item_id`.**
Each duplicated sentence appears twice in the sheet under two different `item_id` values,
and the two rows share one `src_id`. To recover a pair, join the key to itself on `src_id`
and keep the `src_id` values that occur more than once: exactly 20 do, in each locale,
and each of those `src_id` values occurs exactly twice. Any analysis of within-judge
consistency must use that join.

**The `N` in a `repeat_of_N` label is an internal sequence number and must not be read as an
`item_id`.** The label was generated while the duplicate pairs were being built, before the
presentation-order `item_id` values were assigned, so `repeat_of_232` does not mean 'the
duplicate of `ko-232`' - no such pairing exists. For example, `ko-006` carries
`group = repeat_of_232` and is paired with `ko-013`; the two share `src_id = 15968` and
neither is `ko-232`. The labels are retained unchanged because the deposited annotation
sheets quote them verbatim, but they carry no pairing information. `src_id` is the only
pairing key in this directory.

## The key files are not for judges

`P2_ko_key.csv` and `P2_zh_key.csv` hold the group assignment, the source-sentence
identity and the flip flag. **They are withheld from the judges during evaluation**:
a judge who could see `group` would know which rows are duplicates, and a judge who
could see `src_id` could look up the sentence already judged. They were previously named
`P2_ko_key_PRIVATE.csv` / `P2_zh_key_PRIVATE.csv`; the `_PRIVATE` suffix was dropped
because it read as a variable name rather than as a handling instruction, and the
prohibition is stated here instead. **Nothing in this directory is anonymised**: the
blind sheets are blind to the study's hypothesis, not to the text.

## Time

**About 250 items at 10–20 seconds each: roughly 50–70 minutes.** Two sittings are fine.

**Return the file unchanged: do not rename columns, reorder rows, or delete any.**
