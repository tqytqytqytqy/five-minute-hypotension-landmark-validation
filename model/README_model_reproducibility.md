# LM5 frozen-model reproducibility package

This directory contains non-patient-level material needed to inspect and apply the frozen LM5 model used in the MOVER U0 external validation.

## Frozen study contract

- Study ID: `LM5_COMMON18_INSPIRE_MOVER_20260712`
- Bundle version: `1.1.0`
- Development database: `INSPIRE_1.4.2`
- External-validation database: `MOVER_EPIC`
- Primary model: 18-feature ridge logistic regression
- Selected ridge penalty: `1e-05`
- Final technical freeze: `2026-07-12T04:04:30.198832Z`
- Independent freeze receipt: `2026-07-12T04:04:54.012337Z`
- U0 start: `2026-07-12T04:06:58.536924Z`
- Model state: `U0_no_update`
- U0 cohort: 7,177 patients; 1,398 events; 2,000 bootstrap repetitions

The freeze metadata records that patient-level MOVER predictions and MOVER performance had not been seen before freeze. It also transparently records a known prior aggregate MOVER proxy-event summary; this was not patient-level prediction or model-performance information.

## Included files

- `model.json`: model family, intercepts, coefficients, feature order, development metadata.
- `preprocess.json`: frozen imputation and transformation metadata.
- `feature_contract.json`: feature definitions and ordering.
- `cohort_endpoint.json`: locked cohort and outcome rules.
- `thresholds.json`: frozen operating thresholds.
- `ridge_cv_summary.csv`: aggregate development cross-validation summary.
- `formal_external_validation_precision_plan.csv`: pre-U0 precision targets.
- `freeze_provenance.json`: freeze timing and aggregate provenance.
- `pre_U0_freeze_receipt.json`: independently stored pre-U0 receipt.
- `environment.lock`: software-environment metadata.
- `code_SHA256SUMS.csv`: hashes of the frozen analysis code.
- `test_vector_verification.json`: aggregate numerical verification only.
- `synthetic_prediction_example.json`: fully synthetic aggregate-median input with expected LM5 linear predictor and probability.
- `U0_execution_summary_redacted.json`: de-identified execution facts.
- `source_aggregate_evidence/`: non-patient-level development, cohort-flow, missingness, and outcome-observability audit files used for transparent reporting.
- `../11_QA_manifest/final_SHA256SUMS.csv`: release-relative hashes generated after final QA.

The source analysis field `ici_equal_frequency` is retained in immutable machine-readable results for backward compatibility. Its implementation compares each patient's predicted risk with the observed event rate in the corresponding equal-frequency bin and then averages the absolute patient-level deviations. Public-facing materials therefore call it “10-bin patient-level absolute calibration deviation”; it is neither standard expected calibration error nor a smooth calibration-curve integral. No numerical result was changed.

The frozen cohort contract also retains a legacy Stage-1 recurrence criterion. Under the H5 observation contract that criterion was structurally inactive/unobservable; operational Stage-1 membership was determined by early AUC below MAP 65 of at least 50 mm Hg·min. The public workflow is therefore described as a severe early-burden branch rather than as a recurrence branch.

## Deliberately excluded

The original `test_vectors.csv` contains real INSPIRE-derived patient feature rows and is not synthetic. The original `split_manifest.csv` contains patient/case hashes and outcomes. Both are excluded, as are the original U0 execution lock, host name, process identifier, staging directory, local paths, and source manifests containing machine-specific locations. The included synthetic example is constructed from aggregate frozen imputation medians and contains no patient row.

The included aggregate verification summary demonstrates that the frozen implementation reproduced the original expected probabilities to numerical tolerance. Access to source patient data remains governed by INSPIRE and MOVER terms.
