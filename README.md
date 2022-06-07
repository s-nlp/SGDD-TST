This repository presents the results of the research descirbed in Studying the role of named entities for content preservation in text style transfer

# Datasets

## SGDD-TST

[SGDD-TST - Schema-Guided Dialogue Dataset for Text Style Transfer](/dataset/SGDD-TST.csv) is a dataset for evaluating the quality of content similarity measures for text style transfer in the domain of the personal plans. The original texts were obtained from [The Schema-Guided
Dialogue Dataset](https://arxiv.org/pdf/1909.05855.pdf) and were paraphrased by the [T5-based mode](https://huggingface.co/ceshine/t5-paraphrase-paws-msrp-opinosis) trained on [GYAFC formality dataset](https://aclanthology.org/N18-1012/). The results were annotated by the crowdsource workers using [Yandex.Toloka](https://toloka.yandex.ru/).


## SGDD_self_annotated_subset 

[SGDD_self_annotated_subset](/dataset/SGDD_self_annotated_subset.csv) is a subset of SGDD-TST manually annotated to perform an error analysis of the pre-trained formality transfer model.
