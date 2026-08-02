# Public release decision record

**Candidate:** version 1.0.1

**Prepared:** 2026-08-02

**Status:** DOI-reserved release candidate; publication pending

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
- [x] Candidate automated tests passed in a compatible environment (65 tests; 2026-07-20); v1.0.1 verification is recorded in the release-task report.
- [x] Machine-readable JSON, TOML, and CFF/YAML metadata validate syntactically.
- [x] Final privacy, secret, path, and filename audit passes.
- [x] All release files have a recorded SHA-256 checksum in `qa/SHA256SUMS.txt`.
- [ ] All authors have confirmed public-release approval and the proposed MIT plus CC BY 4.0 licensing for v1.0.1.
- [ ] An authorized account owner has approved GitHub publication and Zenodo DOI registration for v1.0.1.

This record does not authorize external publication. Version DOI `10.5281/zenodo.21753664` is reserved; publication remains pending until the required approvals and the GitHub and Zenodo records have been verified.

## Verification evidence

- Automated tests: `Ran 65 tests ... OK`.
- Frozen-code manifest: 18 included frozen files matched their recorded SHA-256 values; the sole unavailable manifest entry was the deliberately excluded private `02_code_configs/configs/paths.json`, which is replaced by `paths.example.json`.
- Metadata: the candidate's JSON, TOML, and CFF/YAML metadata parsed successfully; v1.0.1 metadata verification is recorded in the release-task report.
- Naming/privacy scan: no public filename or content contains a journal name, local user/volume path, author email address, private-key marker, API-key marker, or access-token marker.
