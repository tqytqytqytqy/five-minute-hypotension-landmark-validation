# Public release decision record

**Candidate:** version 1.0.1

**Prepared:** 2026-08-02

**Status:** authorized DOI-reserved candidate; not yet published

## Intended public object

- Repository name: `five-minute-hypotension-landmark-validation`
- Archival title: *Five-minute landmark model for intraoperative hypotension: reproducibility materials*
- Resource type: software
- Creators: Qingyu Teng; Qi Li; Sijia Yao; Xiaoxiao Chen; Mengya Ni; Jin Zhao; Yingya Zhao; Hui Zhang

The repository name, archival title, metadata, and public file names are journal-neutral.

## Publication scope

Permitted for public release:

- analysis source code and tests;
- locked analysis configuration and a placeholder-only path template;
- model coefficients and preprocessing metadata;
- feature, endpoint, and threshold contracts;
- synthetic numerical verification;
- freeze receipts stripped of host/process/local-path metadata;
- non-patient-level aggregate results and audit evidence;
- 19 non-patient-level aggregate source-data CSV files and their README;
- five programmatically generated figures in PDF and PNG formats;
- protocol and deviation log.

Excluded from public release:

- raw INSPIRE, MOVER, or VitalDB files;
- patient-level analytic data or predictions;
- real patient-derived test vectors;
- split manifests, patient/case identifiers, or hashes derived from them;
- source manifests with machine-specific locations;
- local paths, host names, process identifiers, credentials, or tokens.

## Release gates

- [x] Creator order and spelling match the locked submission metadata.
- [x] Public project and file names do not contain a journal name.
- [x] Public source-data redistribution is prohibited and documented.
- [x] Automated tests passed on 2026-08-02 (65/65) using Python 3.12.13 and the locked dependency set.
- [x] Machine-readable JSON, TOML, and CFF/YAML metadata validate syntactically.
- [x] Final privacy, secret, path, and filename audit passes.
- [x] All release files have a recorded SHA-256 checksum in `qa/SHA256SUMS.txt`.
- [x] The user explicitly confirmed all-author consent for public release and the MIT (code) plus CC BY 4.0 (documentation and aggregate materials) licensing for v1.0.1 on 2026-08-02.
- [x] The user explicitly authorized publication through `tqytqytqytqy` GitHub and Zenodo v1.0.1 on 2026-08-02.

Version DOI `10.5281/zenodo.21753664` is reserved. Publication is authorized but remains pending until the GitHub and Zenodo records have been created and verified.

## Verification evidence

- Automated tests: `.venv/bin/python -m unittest discover -s 09_QA_reproducibility/tests -p 'test_*.py' -v` on 2026-08-02 under Python 3.12.13: `Ran 65 tests in 0.386s`, `OK`.
- Frozen-code manifest: 18 included frozen files matched their recorded SHA-256 values; the sole unavailable manifest entry was the deliberately excluded private `02_code_configs/configs/paths.json`, which is replaced by `paths.example.json`.
- Metadata: the 13 tracked JSON files, `pyproject.toml` (TOML), and `CITATION.cff` (CFF/YAML) parsed successfully on 2026-08-02 under Python 3.12.13.
- Checksum manifest: regenerated deterministically from `git ls-files`, sorted as `./path`, excluding only `qa/SHA256SUMS.txt`; independent path-set and SHA-256 verification found no missing, extra, or mismatched entries.
- Naming/privacy scan: an auditable 2026-08-02 scan of every tracked filename and text file found zero banned raw/patient-level file extensions, local user/volume paths, email addresses, common credential/token/private-key signatures, or journal-brand terms. Binary PDFs and PNGs were included in the filename scan and in the checksum manifest.
