# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 00:44:54 2021

@author: sgawalsh
"""

import os
import random
import pickle
import torch
from numpy import concatenate, expand_dims
import project_vars

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
def normalize(x): #mean = 0, std = 1
    return (x - x.mean()) / x.std()

def make_get_x(is_2d, norm):
    if is_2d:
        if norm:
            def get_x(x_in):
                x_in = [expand_dims(el, 1) for el in x_in]
                return normalize(concatenate(x_in, axis = 1))
        else:
            def get_x(x_in):
                x_in = [expand_dims(el, 1) for el in x_in]
                return concatenate(x_in, axis = 1)
    else:
        if norm:
            def get_x(x_in):
                return normalize(concatenate(x_in))
        else:
            def get_x(x_in):
                return concatenate(x_in)
    
    return get_x
            
def data_generator_torch_batch(batch_size, device, is_train = True, is_dev = False, path = "processed_data\\", norm = False, is_2d = False):
    
    path = "{}{}{}\\{}\\".format(path, "dev\\" if is_dev else "train\\", project_vars.base_channels, "train" if is_train else "eval")
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    random.shuffle(files)
    targ_map = project_vars.letter_map
    
    get_x = make_get_x(is_2d, norm)
        
    for f_batch in batch(files, batch_size):
        x, y, y_lengths = [], [], []
        
        for f in f_batch:
            xy = pickle.load(open("{}{}".format(path, f), "rb"))
            
            sentence = [targ_map[char] for char in xy[1]]
            x.append(torch.from_numpy(get_x(xy[0])).float().to(device))
            
            y.append(sentence)
            y_lengths.append(len(sentence))
        
        y = [(el + (max(y_lengths) - len(el)) * [0]) for el in y] # pad target sequences
        
        yield x, y, y_lengths
            
def phoneme_generator_torch_batch(batch_size, device, is_train = True, is_dev = False, path = "processed_data\\", norm = False, is_2d = False):
    
    path = "{}{}{}\\{}\\".format(path, "dev\\" if is_dev else "train\\", project_vars.base_channels, "train" if is_train else "eval")
    files = [f for f in os.listdir(path + "phones\\")]
    random.shuffle(files)
    
    targ_map = project_vars.phoneme_map
    
    get_x = make_get_x(is_2d, norm)
        
    for f_batch in batch(files, batch_size):
        x, y, y_lengths = [], [], []
        
        for f in f_batch:
            x_in = pickle.load(open("{}{}".format(path, f), "rb"))[0] # audio data
            sentence_phones = pickle.load(open("{}{}".format(path + "phones\\", f), "rb"))
            phone_sequence = []
            
            for word_phones in sentence_phones:
                for phone in word_phones:
                    phone_sequence.append(targ_map[phone])
                phone_sequence.append(40) # space after each word
            y.append(phone_sequence[:-1]) # remove last space
            
            x.append(torch.from_numpy(get_x(x_in)).float().to(device))
            y_lengths.append(len(phone_sequence) - 1)
        
        y = [(el + (max(y_lengths) - len(el)) * [0]) for el in y] # pad target sequences
        
        yield x, y, y_lengths

        
def data_generator_torch_single(device, is_train = True, path = "processed_data\\", is_dev = False, norm = False, is_2d = False):
    
    path = "{}{}{}\\".format(path, "dev\\" if is_dev else "train\\", "train" if is_train else "eval")
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    random.shuffle(files)
    
    targ_map = project_vars.letter_map
    
    get_x = make_get_x(is_2d, norm)
    
    for f in files:
        xy = pickle.load(open("{}{}".format(path, f), "rb"))
        
        x = torch.from_numpy(get_x(xy[0])).float().to(device)
        y = torch.tensor([targ_map[char] for char in xy[1]], dtype=torch.float32)
        y_length = torch.full(size=(1,), fill_value = len(y), dtype=torch.int32).to(device)
        
        yield x, y, y_length
        
def get_n(device, phones, n = 10, is_train = True, data_path = "processed_data\\train\\", seed = 0, norm = False, is_2d = False):
    
    data_path = "{}{}\\{}\\".format(data_path, project_vars.base_channels, "train" if is_train else "eval")
    
    file_path = data_path + "phones\\" if phones else data_path
    
    files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    
    if seed >= 0:
        random.seed(seed)
        
    random.shuffle(files)
    
    files = files[:n]
    
    get_x = make_get_x(is_2d, norm)
    
    for f in files:
        xy = pickle.load(open("{}{}".format(data_path, f), "rb"))
        
        x = torch.from_numpy(get_x(xy[0])).float().to(device)
        sentence = xy[1]
        
        if phones:
            sentence_phones = pickle.load(open("{}{}".format(file_path, f), "rb"))
            phone_sequence = []
            
            for word_phones in sentence_phones:
                for phone in word_phones:
                    phone_sequence.append(phone)
                phone_sequence.append(" ")# space after each word
            phones = "".join(phone_sequence[:-1]) # remove last space
            
            yield x, sentence, phones
        else:
            yield x, sentence