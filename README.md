# Five-minute hypotension landmark validation


Reproducibility materials for the development and independent external validation of a frozen five-minute landmark model for persistent or high-burden intraoperative hypotension after an initial observed mean arterial pressure (MAP) below 65 mmHg.


The model was developed using INSPIRE and evaluated without model updating in MOVER. The repository is intentionally journal-neutral and contains no patient-level data.


## What is included


- `02_code_configs/src/lm5_validation/`: reusable cohort, observation-process, frozen-model, validation, and statistical functions.
- `02_code_configs/scripts/`: the numbered analysis workflow used for source audit, cohort construction, technical freeze, one-time external validation, secondary analyses, tables, figures, and final quality assurance.
- `02_code_configs/configs/analysis.json`: the locked analysis contract.
- `02_code_configs/configs/paths.example.json`: a local path template containing no real machine locations.
- `model/`: frozen model coefficients, preprocessing metadata, endpoint and feature contracts, thresholds, synthetic prediction example, freeze receipts, and non-patient-level aggregate evidence.
- `00_protocol_SAP/`: the statistical analysis plan and deviation log.
- `09_QA_reproducibility/tests/`: automated contract and statistical tests.


## What is not included


This release excludes raw INSPIRE and MOVER files, patient-level analytic datasets, patient-level predictions, real patient-derived test vectors, split manifests, patient or case hashes, host/process metadata, access credentials, and local filesystem paths. Data access remains governed by the source repositories and their data-use terms.


## Reproducing the software environment


Python 3.12 was used for the frozen analysis. Create an isolated environment, install the recorded dependencies, and run the automated tests:


```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s 09_QA_reproducibility/tests -p 'test_*.py'
```


The public repository cannot execute the patient-level workflow by itself because source data are not redistributed. After obtaining authorized local copies of INSPIRE and MOVER, copy `02_code_configs/configs/paths.example.json` to `02_code_configs/configs/paths.json`, replace the placeholders with local paths, and follow the numbered scripts. Never commit `paths.json`, source archives, derived patient-level files, or credentials.


## Applying the frozen model without patient data


`model/synthetic_prediction_example.json` provides a fully synthetic numerical check. The model definition, feature order, imputation rules, scaling parameters, coefficients, and expected output are stored in machine-readable JSON files under `model/`.


## Data availability


INSPIRE version 1.4.2 is available through PhysioNet subject to its access requirements and data-use agreement. MOVER access and reuse are governed by its provider. This repository redistributes neither database.


## Citation


Citation metadata are provided in `CITATION.cff`. A version-specific DOI will be added after the first public archival release.


## Licensing


Source code is released under the MIT License. Documentation, protocol text, model metadata, and non-patient-level aggregate materials are released under the Creative Commons Attribution 4.0 International License. These licenses do not grant rights to the underlying INSPIRE or MOVER data.


## Contact and responsibility


The listed creators are responsible for the released materials. Users must independently verify local data permissions, variable mappings, and clinical interpretation. The model is a research artifact and is not validated for autonomous clinical decision-making.

