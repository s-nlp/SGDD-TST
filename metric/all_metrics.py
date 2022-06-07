from tqdm.auto import tqdm
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import gensim
from nltk import wordpunct_tokenize
from scipy.spatial.distance import cosine
import chrfpp
from importlib import reload
import numpy as np
import gc

os.environ['CUDA_LAUNCH_BLOCKING']='1'
sys.path.append(os.path.abspath('..')) # similarity
sys.path.append(os.path.abspath('../..')) # repository root

class PairsDataset(Dataset):
    def __init__(self,texts_first, texts_second):
        self.texts_first =texts_first
        self.texts_second = texts_second
    def __len__(self):
        return len(self.texts_first)
    def __getitem__(self, idx):
        return self.texts_first[idx], self.texts_second[idx]


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def chrf(references, texts, nworder=2, ncorder=6, beta=2):
    totalF, averageTotalF, totalPrec, totalRec = chrfpp.computeChrF(references, texts, nworder=2, ncorder=6, beta=2)
    return totalF

def chrf_no_agg(references, texts, nworder=2, ncorder=6, beta=2):
    return np.array([chrfpp.computeChrF([r], [t], nworder=2, ncorder=6, beta=2)[0] for r, t in zip([references], [texts])])

###sentence bleu
from nltk.translate.bleu_score import sentence_bleu
def calc_bleu(s1,s2):
    return sentence_bleu([s1], s2)

###meteor
from datasets import load_metric
meteor = load_metric("meteor")
def meteor_pair(x, y):
    return meteor.compute(predictions=[x], references=[y])['meteor']

### rouge
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3', 'rougeL'], use_stemmer=True)

def score(sent1_list, sent2_list, human_scores, file_save_path, calculate_long = False):

    collected_exceptions = []
    
    if os.path.isfile(file_save_path):
        df_report = pd.read_csv(file_save_path)
    else:
        df_report = pd.DataFrame({'text1':sent1_list,'text2':sent2_list,"human":human_scores})
        
    collected_metrics = list(df_report.columns)
    
    if 'chrf' not in collected_metrics:
        try:
            print('chrf')
            chrf_res = [chrf_no_agg(t1,t2)[0] for t1,t2 in zip(tqdm(sent1_list),sent2_list)]
            df_report['chrf'] = chrf_res
            df_report.to_csv(file_save_path, index = None)
        except Exception as e:
            collected_exceptions.append(('chrf',e))

    if 'bleu' not in collected_metrics:
        try:
            print('bleu')
            bleu = [calc_bleu(t1,t2) for t1,t2 in zip(tqdm(sent1_list),sent2_list)]
            df_report['bleu'] = bleu
            df_report.to_csv(file_save_path, index = None)
        except Exception as e:
            print("Failed with <{}>".format(e))
            collected_exceptions.append(('bleu',e))
            
    if 'meteor' not in collected_metrics and calculate_long == True:
        try:
            print('meteor')
            meteor = [meteor_pair(t1,t2) for t1,t2 in zip(tqdm(sent1_list),sent2_list)]
            df_report['meteor'] = meteor
            df_report.to_csv(file_save_path, index = None)
        except Exception as e:
            print("Failed with <{}>".format(e))
            collected_exceptions.append(('meteor',e))

    if 'rouge1' not in collected_metrics:
        try:
            print('rouge')
            r1 = []
            r2 = []
            r3 = []
            rl = []
            for t1,t2 in zip(tqdm(sent1_list),sent2_list):
                scores = scorer.score(t1,t2)
                r1.append(scores['rouge1'].fmeasure)
                r2.append(scores['rouge2'].fmeasure)
                r3.append(scores['rouge3'].fmeasure)
                rl.append(scores['rougeL'].fmeasure)


            df_report['rouge1'] = r1
            df_report['rouge2'] = r2
            df_report['rouge3'] = r3
            df_report['rougeL'] = rl

            df_report.to_csv(file_save_path, index = None)
        except Exception as e:
            print("Failed with <{}>".format(e))
            collected_exceptions.append(('rouge',e))
 
    #prepare for larger models
    inference_dataset = PairsDataset(sent1_list, sent2_list)
    batch_size = 8
#     print(f"len dataset is {len(sent1_list)}")
    inference_dataloader = DataLoader(inference_dataset, batch_size = batch_size, drop_last = False)
        
    if "Elron/bleurt-large-512" not in collected_metrics:
        try:
            def score_a_pair_bleurt(text1, text2, model, tokenizer):
                inputs = tokenizer(text1, text2, padding=True, truncation=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    result = model(**inputs)[0]#.item()
                return [e.item() for e in result]

            print("BLEURT")
            for model_path in ["Elron/bleurt-large-512"]:
                if model_path in collected_metrics: continue
                cleanup()
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
                model.eval();

                metric = []
                for b in tqdm(inference_dataloader):
                    metric_curr = score_a_pair_bleurt(*b, model, tokenizer)
                    metric.extend(metric_curr)

                df_report[model_path] = metric
                df_report.to_csv(file_save_path, index = None)
        except Exception as e:
                collected_exceptions.append(('bleurt',e))

    if "bertscore/microsoft/deberta-xlarge-mnli" not in collected_metrics:
        try:
            prefix = 'bertscore/'
            from bert_score import score

            print("bertscore")
            for modelname in ["roberta-large","bert-base-multilingual-cased",'microsoft/deberta-xlarge-mnli']:
                if modelname in collected_metrics:
                    continue
                full_path = prefix + modelname
                print(modelname)
                if full_path not in collected_metrics:
                    P, R, F1 = score(sent1_list,sent2_list, model_type=modelname, verbose=True, batch_size  = 8)
                    df_report[full_path] = F1.tolist()

                df_report.to_csv(file_save_path, index = None)
                
        except Exception as e:
                collected_exceptions.append(('bertscore',e))
               
    if any ([el not in collected_metrics for el in ['fasttext_cossim', 'w2v_cossim']]):
        print("w2v/fasttext")
        
        def maybe_tokenize(text, min_len=2):
            if isinstance(text, str):
                text = [t for t in wordpunct_tokenize(text) if len(t) >= min_len]
            return text

        def get_vectors(document, model):
            document_vectors = [model[w] for w in document]
            if len(document_vectors) == 0:
                shape = model['mother'].shape[0]
                nul_vector = np.zeros(shape)
                document_vectors.append(nul_vector)
            return document_vectors

        def cossim(t1,t2,model):

            document1 = maybe_tokenize(t1)
            document2 = maybe_tokenize(t2)

            document1 = [token for token in document1 if token in model]
            document2 = [token for token in document2 if token in model]   

            document1_vectors = get_vectors(document1, model)
            document2_vectors = get_vectors(document2, model)  

            document1_emb = np.mean(np.stack(document1_vectors),0)
            document2_emb = np.mean(np.stack(document2_vectors),0)

            return 1 - cosine(document1_emb, document2_emb)
        
        if 'fasttext_cossim' not in collected_metrics:
            print('fasttext')
            try:
                model = gensim.models.fasttext.load_facebook_vectors('cc.en.300.bin')
                
                metric = [cossim(t1,t2, model) for t1,t2 in zip(tqdm(sent1_list),sent2_list)]
                df_report['fasttext_cossim'] = metric

                df_report.to_csv(file_save_path, index = None)
            except Exception as e:
                collected_exceptions.append(('fasttext',e))
           
        
        if 'w2v_cossim' not in collected_metrics:
#             model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
                
#             metric = [cossim(t1,t2, model) for t1,t2 in zip(tqdm(sent1_list),sent2_list)]
            try:
                print('w2v')
                model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
                
                metric = [cossim(t1,t2, model) for t1,t2 in zip(tqdm(sent1_list),sent2_list)]
                df_report['w2v_cossim'] = metric              

                df_report.to_csv(file_save_path, index = None)
            except Exception as e:
                collected_exceptions.append(('w2v',e))

    if len(collected_exceptions) == 0:
        print("No exception raised!")
    else:
         for f, exc in collected_exceptions:
            print("Calculation of <{}> failed with <{}>".format(f,exc))
            