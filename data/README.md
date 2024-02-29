# SIGA Dataset

A natural language inference dataset for assessing NLI models' ability to draw scalar inferences.

## Splits
1) Training dataset: Training examples for fine-tuning NLI models.
2) In-domain test dataset (`test_id`): Test examples containing the same scalar adjective pairs that are present in the training dataset.
3) Out-of-domain test dataset (`test_ood`): Test examples containing scalar adjective pairs that are not present in the training dataset.

## Fields
1) `premise`: Contains the context +  premise
2) `hypothesis`: contains the hypothesis
3) `label`: the true NLI label
4) `adjective_premise`: the scalar adjective in the premise causing the scalar implicature.
5) `adjective_hypothesis`: the corresponding scalar adjective in the hypothesis causing the scalar implicature.

## Files
1) Train, in-domain and out-of-domain test splits are provided in both CSV and JSON format.
2) `raw_numberical_data.csv` contains the raw numerical responses by annotators before conversion to NLI labels.
