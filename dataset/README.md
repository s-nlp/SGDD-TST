# Datasets


## SGDD-TST

The file consists of the following columns
- INPUT:text_first - the original text
- INPUT:text_second - formality transferred text
- OUTPUT:result - automatically assigned label of the annotation (David-Skene aggregation method is used)
- CONFIDENCE:result - confidence of the annotation
- vote_type - 
- vote_different - number of votes for the option "The texts are compltely different"
- vote_some_details_lost  - number of votes for the option "The texts are similar but have significant differences"
- vote_OK - number of votes for the option "The texts mean the same or have minor differences"
- **average - averaged score of content similarity. This score can be used for the for evaluating the quality of content similarity measures, e.g. by calculation Spearman Rank Correlation Coefficient between these scores and automatic scores**


## SGDD_self_annotated_subset 

[SGDD_self_annotated_subset](/SGDD_self_annotated_subset.csv) is a subset of SGDD-TST manually annotated to perform an error analysis of the pre-trained formality transfer model.


## SGDD_self_annotated_subset 
SGDD_self_annotated_subset - the subset of SGDD-TST manually annotated to perform an error analysis of the pre-trained formality transfer model.
