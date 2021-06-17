# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 02:47:43 2021

@author: sgawalsh
"""

# from nltk.corpus import reuters
from nltk import ngrams
import os
import pickle
    
def create_n_char_level(n = 4, path = "data\\LibriSpeech\\train-clean-100\\"):
    # Create a placeholder for model
    model = {}
    
    # Count frequency of co-occurance  
    for p ,d, f in os.walk(path):
        for file in f:
            if os.path.splitext(os.path.join(p, file))[1] == '.txt':
                with open(os.path.join(p, file)) as trans:
                    for line in trans.readlines():
                        line = line.split(' ')[1:]
                        
                        line = " ".join(line)
                        line = line.rstrip("\n")
                        line = [c for c in line]
                        
                        for ngram in ngrams(line, n + 1, pad_left=True):
                            pre = ngram[:-1]
                            last = ngram[-1]
                            if pre not in model:
                                model[pre] = {}
                            entry = model[pre]
                            if last not in entry:
                                entry[last] = 1
                            else:
                                entry[last] += 1
     
    # Let's transform the counts to probabilities
    for entry in model:
        total_count = float(sum(model[entry].values()))
        for char in model[entry]:
            model[entry][char] /= total_count
            
    pickle.dump(model, open("lang_models\\char_model_{}gram.pck".format(str(n)), "wb"))
    
def create_n_phoneme_level(n = 4, path = "processed_data\\train\\train\\phones\\"):
    # Create a placeholder for model
    model = {}
    
    for file in os.listdir(path):
        phone_sentence = pickle.load(open(path + file, "rb"))
        phone_sequence = []
        for word in phone_sentence:
            phone_sequence.extend(word)
            phone_sequence.append(" ")
        phone_sequence.pop()
        for phone_gram in ngrams(phone_sequence, n + 1, pad_left=True):
            pre = phone_gram[:-1]
            last = phone_gram[-1]
            if pre not in model:
                model[pre] = {}
            entry = model[pre]
            if last not in entry:
                entry[last] = 1
            else:
                entry[last] += 1
                
    for entry in model:
        total_count = float(sum(model[entry].values()))
        for char in model[entry]:
            model[entry][char] /= total_count
    
    pickle.dump(model, open("lang_models\\phoneme_model_{}gram.pck".format(str(n)), "wb"))
    
   
def create_word_level(n, path = "data\\LibriSpeech\\train-clean-100\\"):
    # Create a placeholder for model
    model = {}
    
    # Count frequency of co-occurance  
    for p ,d, f in os.walk(path):
        for file in f:
            if os.path.splitext(os.path.join(p, file))[1] == '.txt':
                with open(os.path.join(p, file)) as trans:
                    for line in trans.readlines():
                        line = line.split(' ')[1:]
                        
                        end_word = line.pop()
                        if "\n" in end_word:
                            end_word = end_word.rstrip("\n")
                        line.append(end_word)
                        
                        # for i, word in enumerate(line):
                        #     if "\n" in word:
                        #         line[i] = word.rstrip("\n")
                        
                        line.append(None) # Pad right once, None acts as EOS
                        for ngram in ngrams(line, n + 1, pad_left=True):
                            pre = ngram[:-1]
                            last = ngram[-1]
                            if pre not in model:
                                model[pre] = {}
                            tri = model[pre]
                            if last not in tri:
                                tri[last] = 1
                            else:
                                tri[last] += 1
     
    # Let's transform the counts to probabilities
    for entry in model:
        total_count = float(sum(model[entry].values()))
        for char in model[entry]:
            model[entry][char] /= total_count
            
    pickle.dump(model, open("word_model_dict.pck", "wb"))
    
# create_n_phoneme_level(5)