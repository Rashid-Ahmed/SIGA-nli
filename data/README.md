# Information regarding the SIGA dataset.

## The siga dataset has been split in 3 parts.
1) Training dataset: These are examples used for finetuning NLI models.
2) In-domain test dataset (`test_id`): These test examples contain the same scalar adjective pairs that are present in the training dataset.
3) Out-of-domain test dataset (`test_ood`): These test examples conain scalar adjective pairs that are not present in the training dataset.

## Information regarding the fields in the datasets
1) premise: Contains the context +  premise
2) hypothesis: contains the hypothesis
3) label: the true NLI label
4) adjective_premise: the scalar adjective in the premise causing the scalar implicature.
5) adjective_hypothesis: the corresponding scalar adjective in the hypothesis causing the scalar implicature.

## Information regarding the files
1) Each file is provided in both csv and json format
2) The raw_numberical_data.csv file contains the raw responses by annotators.
