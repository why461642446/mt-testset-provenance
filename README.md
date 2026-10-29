# mt-testset-provenance

**The maintained version of this project is on Zenodo, not here.**
Code, run records, machine translations, human-evaluation sheets and the archived supplementary material for
"What Does a Machine-Translated Test Set Measure? Provenance Effects in Multilingual Intent Classification"
(Haoyi Wang and MinPo Jung) are published at:

- Zenodo, version 1.3: <https://doi.org/10.5281/zenodo.23608631>
- Concept DOI (always resolves to the latest version): <https://doi.org/10.5281/zenodo.23214347>

This repository is kept as the public record of the first pre-registration layer. Its earlier commits are left unchanged:

- `b676eae` (2026-10-08): the deposited predictions for de-DE / vi-VN / ja-JP (first layer);
- `4297586` (2026-10-10): the registered provenance, length rule and analysis code.

The second and third registration layers are on OSF (kmjb7, k53bx).

DeepL translations are not redistributed. The folder `massive_mt_deepl/` was removed on 2026-10-29; its files remain
only in the commit history, as they were at `0ef48b0`. The script that generates them (`make_massive_mt.py`) is in
the Zenodo deposit. The other files here are the state of 2026-10-10 and are superseded by the Zenodo deposit.

Licences: see the Zenodo deposit (code MIT; NLLB translations CC BY-NC 4.0; everything else CC BY 4.0).
