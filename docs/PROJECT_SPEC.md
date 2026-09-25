# ECHO-ER: Project Specification

*(Completed during Phase 0: Understand the Challenge)*

## Challenge Understanding
- **What is S1?**: The deduplicated reference source of business records.
- **What is S2/S3?**: Two independent sources contributing partial, noisy fragments of business information.
- **What does one row represent?**: A single record/fragment for a business entity, containing its ID, name, address, and country.
- **What constitutes a true match?**: Records from S2 and S3 that refer to the same real-world business entity as the given S1 record.
- **Can one S1 match multiple S2/S3 records?**: Yes. An S1 record can match zero, one, or many records across S2 and S3.
- **What constitutes a no-match?**: An S1 entity with no corresponding records in S2 or S3 (a singleton). The output for this is an empty list.
- **What is the exact target format?**: Tab-separated file (`.tsv`). Columns: `source1_entity_id` and `matched_entity_ids` (comma-separated list of matched S2/S3 IDs).
- **What is the exact evaluation metric?**: F_0.5 Score (macro-averaged per Source 1 entity). It heavily penalizes false merges (precision-heavy).
- **What fields are available?**: `entity_id`, `business_name`, `business_address`, `country`.
- **What countries/regions appear?**: Training data: `US`, `India`. Test data: `US`, `India`, `France`.
- **What are the training/test differences?**: Test data introduces a new country (`France`) not seen in the training data.
- **What external resources are prohibited?**: STRICTLY PROHIBITED to use external databases, APIs, geocoding services, or commercial ER APIs.
- **What is allowed?**: Local processing, open-source models up to 8 Billion parameters (MIT/Apache 2.0 License).
- **What are the submission requirements?**: A zip file containing `output/` (`matching_results.tsv`, `candidate_pairs.tsv`), `code/` (`src/`, `README.md`, `requirements.txt`), and `Documentation_template.md`.
