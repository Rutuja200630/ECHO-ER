# ECHO-ER Data Audit

*(Completed during Phase 1: Data Discovery)*

## S1 Summary (Train & Test combined concepts)
- **number of rows**: Train = 2,206,821 | Test = 1,732,544
- **number of columns**: 4
- **columns**: `entity_id`, `business_name`, `business_address`, `country`
- **dtypes**: strings
- **missingness**: No missing values in S1.
- **duplicates**: 0 duplicate entity IDs.

## S2 Summary
- **number of rows**: Train = 5,034,616 | Test = 4,887,273
- **number of columns**: 4
- **columns**: `entity_id`, `business_name`, `business_address`, `country`
- **dtypes**: strings
- **missingness**: 
  - Train: `business_name` (2 missing), `business_address` (168,967 missing)
  - Test: `business_name` (46 missing), `business_address` (129,408 missing)
- **duplicates**: 0 duplicate entity IDs.

## S3 Summary
- **number of rows**: Train = 5,285,603 | Test = 5,082,316
- **number of columns**: 4
- **columns**: `entity_id`, `business_name`, `business_address`, `country`
- **dtypes**: strings
- **missingness**: 
  - Train: `business_name` (13 missing), `business_address` (175,916 missing)
  - Test: `business_name` (59 missing), `business_address` (136,098 missing)
- **duplicates**: 0 duplicate entity IDs.

## TRAIN Summary (Ground Truth)
- **number of rows**: 2,206,821 (exactly matches S1 Train rows)
- **columns**: `source1_entity_id`, `matched_entity_ids`
- **positive examples (has matches)**: 2,083,574
- **negative examples (singletons)**: 123,247
- **multiple valid matches**: 1,964,417 S1 records have multiple valid matches
- **label distribution**: ~94.4% positive, ~5.6% singletons
- **duplicate pairs check**: 0 (all source1_entity_ids are unique)

## TEST Summary
- **structure**: Same schema as S1/S2/S3.
- **expected output format**: Two TSVs (`matching_results.tsv`, `candidate_pairs.tsv`), each with exactly 1,732,544 rows (one for each S1 Test ID).
