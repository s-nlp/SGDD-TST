This repository presents the results of the research descirbed in Studying the role of named entities for content preservation in text style transfer

# Datasets

## SGDD-TST

### Overview

[SGDD-TST - Schema-Guided Dialogue Dataset for Text Style Transfer](/dataset/SGDD-TST.csv) is a dataset for evaluating the quality of content similarity measures for text style transfer in the domain of the personal plans. The original texts were obtained from [The Schema-Guided
Dialogue Dataset](https://arxiv.org/pdf/1909.05855.pdf) and were paraphrased by the [T5-based mode](https://huggingface.co/ceshine/t5-paraphrase-paws-msrp-opinosis) trained on [GYAFC formality dataset](https://aclanthology.org/N18-1012/). The results were annotated by the crowdsource workers using [Yandex.Toloka](https://toloka.yandex.ru/).

[SGDD-TST - Schema-Guided Dialogue Dataset for Text Style Transfer](/SGDD-TST.csv) is a dataset for evaluating the quality of content similarity measures for text style transfer in the domain of the personal plans. The original texts were obtained from [The Schema-Guided
Dialogue Dataset](https://arxiv.org/pdf/1909.05855.pdf) and were paraphrased by the [T5-based model](https://huggingface.co/ceshine/t5-paraphrase-paws-msrp-opinosis) trained on [GYAFC formality dataset](https://aclanthology.org/N18-1012/). The results were annotated by the crowdsource workers using [Yandex.Toloka](https://toloka.yandex.ru/).

<p align="center">
  <img src="/dataset/img/toloka_interface_example.jpg" alt="drawing" width="500"/>
</p>

<p align = "center">
Fig.1 The example of crowdourcing task
</p>

### Statistics

The dataset consists of 10,277 samples. Krippendorf's alpha agrrement score is 0.64

<p align="center">
  <img src="/dataset/img/hist_distrib.jpg" alt="drawing" width="500"/>
</p>

<p align = "center">
Fig.2 The distribution of the similarity scores in the collected dataset
</p>

## SGDD_self_annotated_subset 

[SGDD_self_annotated_subset](/dataset/SGDD_self_annotated_subset.csv) is a subset of SGDD-TST manually annotated to perform an error analysis of the pre-trained formality transfer model. According to the error analysis we learned that loss or corruption of named entities and some essential parts of speech like verbs, prepositions, adjective, etc. play significant role in the problem of the content loss in formality transfer.


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


