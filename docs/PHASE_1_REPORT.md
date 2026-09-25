# Phase 1: Data Audit Report

## Dataset Summary
- S1 contains the deduplicated reference source of business records. Train set has 2,206,821 rows, Test set has 1,732,544 rows.
- S2 contains partial fragments with 5,034,616 rows in train and 4,887,273 rows in test.
- S3 contains more fragments with 5,285,603 rows in train and 5,082,316 rows in test.

## Important Columns
- `entity_id`: Unique identifier for each business record fragment.
- `business_name`: Name of the business.
- `business_address`: Physical location of the business.
- `country`: The country of the business. (Training set contains US and India, while Test set introduces France).

## Missingness
- S1 has no missing values.
- S2 has minimal missing `business_name` but significant missing `business_address` (~168K in train, ~129K in test).
- S3 has minimal missing `business_name` but significant missing `business_address` (~175K in train, ~136K in test).

## Duplicates
- Across all datasets (S1, S2, S3), there are exactly 0 duplicate `entity_id` values.

## Label Distribution
- The Ground Truth contains exactly one row for every S1 entity in the train set (2,206,821 total rows).
- Positive examples (has matches): 2,083,574 (~94.4%)
- Negative examples (singletons): 123,247 (~5.6%)

## Matching Structure
- The schema is perfectly consistent across S1, S2, and S3 for both Train and Test data.
- An S1 record can map to multiple S2/S3 IDs. Specifically, 1,964,417 S1 records have multiple valid matches, making this a multi-target matching problem.
- Exact required submission format: Two TSVs (`matching_results.tsv` and `candidate_pairs.tsv`), each with exactly 1,732,544 rows mapping one-to-one with S1 Test IDs. Output is a comma-separated list of matched S2/S3 IDs.

## Potential Risks
- High rate of missing addresses in S2 and S3 could hinder accurate matching where business names are similar or generic.
- A new country (France) appears in the test data but not in train, meaning the system cannot solely memorize US/India patterns.
- High cardinality matching (most positive examples have multiple matches) requires the model to correctly identify *all* relevant matches rather than just the top-1 match.

## Exact Conclusions
- The data audit passes successfully, confirming dataset counts, column names, datatypes, missingness, and distributions all match the expected structures.
- `DATA_AUDIT.md` documentation has been corrected to include the multiple match count (1,964,417 records).
- The required data shapes and exact formats are thoroughly verified. Phase 1 is fully complete.
