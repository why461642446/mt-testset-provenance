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

## Time

**About 250 items at 10–20 seconds each: roughly 50–70 minutes.** Two sittings are fine.

**Return the file unchanged: do not rename columns, reorder rows, or delete any.**
