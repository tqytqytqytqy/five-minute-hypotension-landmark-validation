# Public release decision record

**Candidate:** version 1.0.2

**Prepared:** 2026-08-11

**Status:** authorized metadata-only corrective candidate; publication verification pending

## Intended public object

- Repository name: `five-minute-hypotension-landmark-validation`
- Archival title: *Five-minute landmark model for intraoperative hypotension: reproducibility materials*
- Resource type: software
- Creators: Qingyu Teng; Qi Li; Sijia Yao; Xiaoxiao Chen; Mengya Ni; Jing Zhao; Yingya Zhao; Hui Zhang

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
- [x] The unchanged v1.0.1 baseline passed all 65 existing automated tests on 2026-08-11 using Python 3.12.13.
- [x] Version 1.0.2 passed all 65 existing automated tests plus the metadata-consistency test on 2026-08-11 using Python 3.12.13.
- [x] Machine-readable JSON, TOML, and CFF/YAML metadata validate syntactically.
- [x] Final privacy, secret, path, and filename audit passes.
- [x] All v1.0.2 release files have a recorded SHA-256 checksum in `qa/checksums_v1.0.2.sha256`.
- [x] The MIT (code) plus CC BY 4.0 (documentation and aggregate materials) licensing boundary from v1.0.1 is unchanged.
- [x] The user explicitly authorized the metadata-only corrective publication through `tqytqytqytqy` GitHub and Zenodo v1.0.2 on 2026-08-11.

The release-series concept DOI is `10.5281/zenodo.21454384`. The v1.0.2 version DOI must be copied only from the live Zenodo record after publication and independently verified.

## Verification evidence

- Automated tests: `python -m unittest discover -s 09_QA_reproducibility/tests -p 'test_*.py' -v` on 2026-08-11 under Python 3.12.13: `Ran 66 tests`, `OK`.
- Frozen-code manifest: 18 included frozen files matched their recorded SHA-256 values; the sole unavailable manifest entry was the deliberately excluded private `02_code_configs/configs/paths.json`, which is replaced by `paths.example.json`.
- Metadata: `.zenodo.json` (JSON), `pyproject.toml` (TOML), and `CITATION.cff` (CFF/YAML structure) were validated on 2026-08-11 under Python 3.12.13.
- Checksum manifest: regenerated deterministically from the v1.0.2 release file set, sorted as `./path`, excluding only `qa/checksums_v1.0.2.sha256`; independent path-set and SHA-256 verification found no missing, extra, or mismatched entries.
- Artifact-diff audit: model files, aggregate CSV results, figures, protocol, analysis code/configuration, and the 65 pre-existing tests are byte-identical to v1.0.1; changes are limited to approved release metadata, one metadata-specific test, and v1.0.2 receipt/checksum evidence.
