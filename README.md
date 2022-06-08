This repository presents the results of the research descirbed in Studying the role of named entities for content preservation in text style transfer

# Datasets

## SGDD-TST

### Overview

[SGDD-TST - Schema-Guided Dialogue Dataset for Text Style Transfer](/dataset/SGDD-TST.csv) is a dataset for evaluating the quality of content similarity measures for text style transfer in the domain of the personal plans. The original texts were obtained from [The Schema-Guided
Dialogue Dataset](https://arxiv.org/pdf/1909.05855.pdf) and were paraphrased by the [T5-based model](https://huggingface.co/ceshine/t5-paraphrase-paws-msrp-opinosis) trained on [GYAFC formality dataset](https://aclanthology.org/N18-1012/). The results were annotated by the crowdsource workers using [Yandex.Toloka](https://toloka.yandex.ru/).


<p align="center">
  <img src="/dataset/img/toloka_interface_example.jpg" alt="drawing" width="500"/>
</p>

<p align = "center">
Fig.1 The example of crowdourcing task
</p>

### Statistics

The dataset consists of 10,287 samples. Krippendorf's alpha agrrement score is 0.64

<p align="center">
  <img src="/dataset/img/hist_distrib.jpg" alt="drawing" width="500"/>
</p>

<p align = "center">
Fig.2 The distribution of the similarity scores in the collected dataset
</p>

## SGDD_self_annotated_subset 

### Investigating the reasons of content loss in formality transfer

[SGDD_self_annotated_subset](/dataset/SGDD_self_annotated_subset.csv) is a subset of SGDD-TST manually annotated to perform an error analysis of the pre-trained formality transfer model. According to the error analysis, we learned that loss or corruption of named entities and some essential parts of speech like verbs, prepositions, adjectives, etc. play a significant role in the problem of the content loss in formality transfer.


<p align="center">
  <img src="/dataset/img/ling_changes_page.jpg" alt="drawing" width="1000"/>
</p>
<p align = "center">
Fig.3 Statistics of different reasons of content loss in TST
</p>


<p align="center">
  <img src="/dataset/img/all_phenomena_page.jpg" alt="drawing" width="500"/>
</p>
<p align = "center">
Fig.4 Frequency of the reasons for the change of content between original and generated sentences: named entities (NE), parts of speech (POS), named entities with parts of speech (NE+POS), and other reasons (Other).
</p>

### Error analysis of metrics

We also perform an error analysis of some content preservation metrics. We produce two rankings of sentences: a ranking based on their automatic scores and another one based on the manual scores, then sort the sentences by the absolute difference between their automatic and manual ranks, so the sentences scored worse with automatic metrics are at the top of the list. We manually annotate the top 35 samples for the metrics based on various calculation logic.

<p align="center">
  <img src="/dataset/img/corrupted_phenomena.jpg" alt="drawing" width="1000"/>
</p>
<p align = "center">
Fig.5 Errors statistics of the analyzed metrics. BertScore/DeBERTa is referred as BertScore here.
</p>

# Named Entities based metric as an auxiliary signal for standard content preservation metrics

Our findings show that Named Entities play a significant role in the content loss, thus we try to improve existing metrics with NE-based signals. To make the results of this analysis more generalizable we use the simple open-sourced [Spacy NER-tagger](https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-3.2.0) to extract entities from the collected dataset. These entities are processed with lemmatization and then used to calculate the Jaccard index over the intersection between entities from original and generated sentences. This score is used as a baseline Named Entity-based content similarity metric. This signal is merged with the main metrics according to the following formula,

$$M_{weigted} = M_{strong}\times (1-p) + M_{NE}\times p$$ where $p$ is a percentage of Named Entity tokens within all tokens in both texts, $M_{strong}$ is an initial metric and $M_{NE}$ is a Named Entity-based signal. The intuition behind the formula is that the Named Entity-based auxiliary signal is useful in the proportion equal to the proportion of NEs tokens in the text.

<center>
  
| Metric                                  | Correlation with pure metric | Correlation with merged metric | Is increase significant? |
|-----------------------------------------|------|---------------------|------------------|
| Elron/bleurt-large-512                  | 0.56 |                0.56 |       False      |
| bertscore/microsoft/deberta-xlarge-mnli | 0.47 |                0.45 |       False      |
| bertscore/roberta-large                 |  0.4 |                0.37 |       False      |
| bleu                                    | 0.35 |                0.38 |       True       |
| rouge1                                  | 0.29 |                0.36 |       True       |
| bertscore/bert-base-multilingual-cased  | 0.28 |                0.36 |       True       |
| rougeL                                  | 0.27 |                0.35 |       True       |
| chrf                                    | 0.27 |                 0.3 |       True       |
| w2v_cossim                              | 0.22 |                0.33 |       True       |
| fasttext_cossim                         | 0.22 |                0.32 |       True       |
| rouge2                                  | 0.15 |                0.22 |       True       |
| rouge3                                  | 0.09 |                0.14 |       True       |

 </center>
  
  <p align = "center">
Fig.6 Spearman correlation of automatic content similarity metrics with human content similarity scores with and without using auxiliary named Entitis-based metric on the collected SGDD-TST dataset.
</p>
  
Refer to [reproduce_experiments.ipynb](/metric/reproduce_experiments.ipynb) for the implementation of this approach. In this notebook, we show that it yields significant improvement in correlation with human judgments for most of the standardly used content similarity metrics. 

# Contact and Citations

If you have any questions feel free to drop a line to [Nikolay](mailto:bbkhse@gmail.com)

