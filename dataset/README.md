# Datasets


## SGDD-TST

The file consists of the following columns
- INPUT:text_first - the original text
- INPUT:text_second - formality transferred text
- OUTPUT:result - automatically assigned the label of the annotation (David-Skene aggregation method is used)
- CONFIDENCE:result - confidence of the annotation
- vote_type - 
- vote_different - number of votes for the option "The texts are completely different"
- vote_some_details_lost  - number of votes for the option "The texts are similar but have significant differences"
- vote_OK - number of votes for the option "The texts mean the same or have minor differences"
- **average - an averaged score of content similarity. This score can be used for evaluating the quality of content similarity measures, e.g. by calculating the Spearman Rank Correlation Coefficient between these scores and automatic scores**


## SGDD_self_annotated_subset 

[SGDD_self_annotated_subset](/SGDD_self_annotated_subset.csv) is a subset of SGDD-TST manually annotated to perform an error analysis of the pre-trained formality transfer model.

The columns in this dataset are similar to the ones in SGDD-TST. The results of manual annotation are in the following columns
- NE - what happened to the Named Entity (lost, corrupted, or kept)
- POS	- which part of speech (not related to Named Entity) is corrupted or lost
- sentence_type_lost - whether the original sentence is related to one sentence type (declarative, imperative, interrogative, and exclamatory) and the newly generated one becomes related to another one. The labels show the initial sentence type
