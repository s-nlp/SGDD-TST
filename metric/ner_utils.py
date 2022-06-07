from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer 
lemmatizer_wnet = WordNetLemmatizer()
from nltk.corpus import wordnet
from tqdm import tqdm
import nltk
nltk.download('averaged_perceptron_tagger')
import spacy  # version 3.0.6'
from num2words import num2words
import re

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def en_lemmatize(line):
    pos_tagged_ngramm = pos_tag(line.split())
    lemmatized_line_list = []
    for word_el in pos_tagged_ngramm:
        pos = get_wordnet_pos(word_el[1])
        if pos:
            lemma = lemmatizer_wnet.lemmatize(word_el[0], pos =pos)
        else:
            lemma = word_el[0]
        lemmatized_line_list.append(lemma)
    return ' '.join(lemmatized_line_list)


#                 nlp = spacy.load("en_core_web_md")
#                 nlp.add_pipe("entity_linker", last=True)

def normalize_ner(ner_string):
    lemma = en_lemmatize(ner_string.lower())
    if lemma.isdigit():
        lemma = num2words(lemma)
    return lemma

def ner_processor(text, nlp):#normalized_ner_list, noner_token_count, ner_token_count
    
    spacy_processed_entity = nlp(text)
    
    all_ner_strings = []
    for ent in spacy_processed_entity.ents:
        all_ner_strings.append(ent.text)
    
    ner_token_count = 0
    noner_token_count = 0
    for token in spacy_processed_entity:
        if len(re.findall('[\d\w]',token.text))>0:
            if any([token.text in ner_str for ner_str in all_ner_strings]):
#                 print("<{}>".format(token))
                ner_token_count += 1
#                 print(ner_token_count)
            else:
                noner_token_count += 1
    
    return all_ner_strings, ner_token_count, noner_token_count

def unfold_list(lst):
    ner, ner_cnt, noner_cnt = [],[],[]
    for ner_i, ner_cnt_i, noner_cnt_i in lst:
        ner.append(ner_i)
        ner_cnt.append(ner_cnt_i)
        noner_cnt.append(noner_cnt_i)
        
    return ner, ner_cnt, noner_cnt

def get_ner_lists_smart_intersection(ner_list_orig, ner_list_gener, preprocess_func, 
                                     eps=1e-10, print_inersection = False, substring_search = False):
    if len(ner_list_orig) == 0 or len(ner_list_gener) == 0:  return 0

    ner_list_orig_lemm = [preprocess_func(n.lower().strip()) for n in ner_list_orig]
    ner_list_gener_lemm = [preprocess_func(n.lower().strip()) for n in ner_list_gener]

    collected_hits = []

    for ner_original, ner_original_lemm in zip(ner_list_orig, ner_list_orig_lemm):
        if ner_original in ner_list_gener:
            if print_inersection == True: print("ner_original {} found".format(ner_original))
            collected_hits.append(1)

        elif ner_original_lemm in ner_list_gener_lemm:
            if print_inersection == True: print("ner_original_lemm {} found".format(ner_original_lemm))
            collected_hits.append(1)

        elif substring_search == True:
            for ner_orig_substr in ner_original.split():
                if ner_original in ner_list_gener:
                    if print_inersection == True: print("ner_original substring {} found".format(ner_original))
                    collected_hits.append(1)

                elif ner_original_lemm in ner_list_gener_lemm:
                    if print_inersection == True: print("ner_original_lemm substring {} found".format(ner_original_lemm))
                    collected_hits.append(1)
                else:
                    collected_hits.append(0)
        else:
            if print_inersection == True: print("Nothing found for {}".format(ner_original))
            collected_hits.append(0)

    return sum(collected_hits)/max(eps, len(collected_hits))
