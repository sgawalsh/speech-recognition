# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 23:01:03 2021

@author: sgawalsh
"""

import os
import soundfile as sf
import pickle
import librosa
import random
import shutil
import project_vars
from numpy import diff
import nltk
from pathlib import Path

SEED = 0

def gen_db(path = "data\\dev-clean.tar\\dev-clean\\"):
    db = []
    
    for p ,d, f in os.walk(path):
        for file in f:
            if os.path.splitext(os.path.join(p, file))[1] == '.flac':
                ids = p.split('\\')[-2:]
                num = file.split('.')[0].split('-')[2]
                with open(os.path.join(p, "{}-{}.trans.txt".format(ids[0], ids[1]))) as trans:
                    while True:
                        l = trans.readline().split()
                        if l[0] == "{}-{}-{}".format(ids[0], ids[1], num):
                            words = l[1:]
                            break
                data = sf.read(os.path.join(p, file))[0]
                db.append([data, words])
                
    pickle.dump(db, open("db.pck", "wb"))
    print("done")
    
def gen_db_files(path = "data\\LibriSpeech\\train-clean-100\\", is_dev = False):
    
    cwd = os.getcwd()
    arpabet = nltk.corpus.cmudict.dict()
    
    for p ,d, f in os.walk(path):
        for file in f:
            if os.path.splitext(os.path.join(p, file))[1] == '.flac':
                ids = p.split('\\')[-2:]
                num = file.split('.')[0].split('-')[2]
                with open(os.path.join(p, "{}-{}.trans.txt".format(ids[0], ids[1]))) as trans:
                    while True:
                        l = trans.readline().split()
                        if l[0] == "{}-{}-{}".format(ids[0], ids[1], num):
                            words = " ".join(l[1:])
                            break
                        
                data = sf.read(os.path.join(p, file))[0]
                t_data = librosa.feature.mfcc(data, 16000, n_mfcc = project_vars.base_channels, hop_length = 160, win_length = 400) # hop_length = 10 ms, win_length = 25 ms
                d_1 = diff(t_data, prepend = 0) # 1st derivative
                d_2 = diff(d_1, prepend = 0) # 2nd derivative
                
                with open(os.path.abspath("{}\\processed_data\\{}\\{}\\{}-{}-{}.pck".format(cwd, "dev" if is_dev else "train", project_vars.base_channels, ids[0], ids[1], num)), 'wb') as dump_file:
                    pickle.dump([[t_data, d_1, d_2], words], dump_file)
                    
                
                phones = []
                problem = False
                
                for word in words.split():
                    try:
                        sounds = arpabet[word.lower()][0]
                    except KeyError:
                        word = word.replace("'", "") 
                        try:
                            sounds = arpabet[word.lower()][0]
                        except KeyError:
                            problem = True
                            break
                    
                    sounds = [sound.strip("012") for sound in sounds] # ignore syllable accents
                    phones.append(sounds)
                
                if not problem:
                    with open(os.path.abspath("{}\\processed_data\\{}\\{}\\phones\\{}-{}-{}.pck".format(cwd, "dev" if is_dev else "train", project_vars.base_channels, ids[0], ids[1], num)), 'wb') as dump_file:
                        pickle.dump(phones, dump_file)
    
def test_train_split(ratio = .8, path = "processed_data\\", is_dev = False):
    
    dataset = "dev" if is_dev else "train"
    
    path += dataset + "\\{}\\".format(project_vars.base_channels)
    
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    random.Random(SEED).shuffle(files)
    split_num = round(len(files) * ratio)
    
    train = files[:split_num]
    test = files[split_num:]
    
    train_folder = "{}train\\".format(path)
    test_folder = "{}eval\\".format(path)
    Path(train_folder + "phones\\").mkdir(parents=True, exist_ok=True)
    Path(test_folder + "phones\\").mkdir(parents=True, exist_ok=True)
    
    for f in train:
        shutil.move("{}{}".format(path, f), train_folder)
        try:
            shutil.move("{}{}".format(path + "phones\\", f), train_folder + "phones\\")
        except:
            pass
        
    for f in test:
        shutil.move("{}{}".format(path, f), test_folder)
        try:
            shutil.move("{}{}".format(path + "phones\\", f), test_folder + "phones\\")
        except:
            pass
    
def inv_phoneme_dict():
    arpabet = nltk.corpus.cmudict.dict()
    inv_dict = {}
    
    for word, pronounciations in arpabet.items():
        for p in pronounciations:
            p = tuple([el.strip("012") for el in p])
            if p in inv_dict:
                word = word.strip(".")
                dup = False
                for el in inv_dict[p]:
                    if el == word:
                        dup = True
                        break
                if not dup:
                    inv_dict[p].append(word)
            else:
                inv_dict[p] = [word]
    
    with open("phones_2_words.pck", "wb") as f:
        pickle.dump(inv_dict, f)