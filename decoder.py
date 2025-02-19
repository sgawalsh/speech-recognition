# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:48:47 2021

@author: sgawalsh
"""

import torch
import data_gen_torch
import project_vars
from wer import wer
import math
import pickle
import numpy as np
import models
from pathlib import Path
import matplotlib.pyplot as plt


def show_predictions(n_samples = 3, data_path = "processed_data\\", phonemes = False, model_name = "best_weights.pt", is_2d = False, norm = False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if phonemes:
        model = models.wav2phoneme_torch_2d() if is_2d else models.wav2phoneme_torch()
        to_target = [None] + list(project_vars.phoneme_map.keys())
    else:
        model = models.wav2letter_torch_2d() if is_2d else models.wav2letter_torch_test2()
        to_target = [None] + list(project_vars.letter_map.keys())
        
    model.to(device)
    model_path = "models\\{}\\{}\\{}\\{}".format("phonemes" if phonemes else "letters", model.__class__.__name__, project_vars.base_channels, "norm\\" if norm else "")
    Path(model_path).mkdir(parents=True, exist_ok=True)
    model.load_state_dict(torch.load(model_path + model_name))
    
    for i, xy in enumerate(data_gen_torch.get_n(device, phonemes, n = n_samples, is_train = True, seed = -1), 1):
        
        pred = model(torch.unsqueeze(xy[0], 0))
        pred = pred.squeeze().T
        
        # with open("test_preds.pck", "wb") as df: # using test outputs for debugging
        #     pickle.dump([pred.detach().cpu().numpy(), xy[1]], df)
        
        # greedy = greedy_decoder(pred, to_target)
        # exploded = explode_preds(pred, to_target)
        beam_output = log_beam_decoder(pred, to_target, phonemes)
        
        # if phonemes:
        #     beam_output = phones_2_words(beam_output) # words can be found from phonemes, but homonyms are a problem, can just use the phonemes as they are
        # else:
        #     beam_output = "".join(beam_output)
            
        beam_output = "".join(beam_output)
        
        if phonemes:
            print("Target Phonemes: {}\n".format(xy[2]))
            
        print("{})\n Target: {}\n\nOutput: {}\n".format(i, xy[1], beam_output))
        # print("Greedy: {}\n\nExploded: {}\n".format(greedy, exploded))
        
        print("WER: {}".format(round(wer(xy[1].split(' '), beam_output.split(' ')), 2)))
        
class ctc_beam:
    
    def __init__(self, init_string = (), pb = 0, pnb = 0, ptxt = 0):
        self.string = init_string
        self.pb = pb
        self.pnb = pnb
        self.ptxt = ptxt
        
    def __str__(self):
        return "'{}' Abs: {}, Ptot: {}, Pnb: {}, Pb: {}, Ptxt: {}".format(self.string, round(self.ptot() + self.ptxt, 3), round(self.ptot(), 3), round(self.pnb, 3), round(self.pb, 3), round(self.ptxt, 3))
    
    def ptot(self):
        return self.pb + self.pnb
    
    def last_char(self):
        try:
            return self.string[len(self.string) - 1]
        except IndexError:
            return None
        
    def set_ptxt(self, new_letter, model, size = 2):
        chars = list(self.string[-size:])
        
        while len(chars) < size:
            chars.insert(0, None)
        
        try:
            self.ptxt = model[tuple(chars)][new_letter]
        except KeyError:
            self.ptxt = .01
            
class ctc_log_beam(ctc_beam):
    
    def __init__(self, init_string = (), pb = -np.inf, pnb = -np.inf, ptxt = -np.inf, c_txt = 1):
        super().__init__()
        self.string = init_string
        self.pb = pb # probability of trailing blank
        self.pnb = pnb # probability of trailing non-blank
        self.ptxt = ptxt # language model probability
        self.c_txt = c_txt # language model constant
        
    def __str__(self):
        return "'{}' Abs: {}, Ptot: {}, Pnb: {}, Pb: {}, Ptxt: {}".format(self.string, round(merge_logs(self.pb, self.pnb) + self.c_txt * self.ptxt, 3), round(merge_logs(self.pb, self.pnb), 3), round(self.pnb, 3), round(self.pb, 3), round(self.ptxt, 3))
        
    def set_log_ptxt(self, new_letter, model, size = 2): # sets language model probability
        chars = list(self.string[-size:])
        
        # while len(chars) < size:
        #     chars.insert(0, None)
            
        if len(chars) < size:
            chars = (size - len(chars)) * [None] + chars
        
        try:
            self.ptxt = math.log(model[tuple(chars)][new_letter])
        except KeyError:
            self.ptxt = math.log(.001)
            
    def set_log_ptot(self):
        self.log_ptot = merge_logs(self.pb, self.pnb)
        return self.log_ptot
    
class ctc_beam_dict: # holds hypothesis beams, handles score merging, returns top n
    
    def __init__(self):
        self.dict = {}
        
    def merge(self, new_beam):
        if new_beam.string in self.dict: # merge beams
            self.dict[new_beam.string].pb += new_beam.pb
            self.dict[new_beam.string].pnb += new_beam.pnb
        else:
            self.dict[new_beam.string] = new_beam
            
    def log_beam_merge(self, new_beam):
        if new_beam.string in self.dict: # merge beams
            self.dict[new_beam.string].pb = merge_logs(self.dict[new_beam.string].pb, new_beam.pb)
            self.dict[new_beam.string].pnb = merge_logs(self.dict[new_beam.string].pnb, new_beam.pnb)
        else:
            self.dict[new_beam.string] = new_beam
            
    def get_log_best(self, n, c_txt):
        d_list = list(self.dict.values())
        d_list.sort(key = lambda x: merge_logs(x.pb, x.pnb) + c_txt * x.ptxt, reverse = True)
        
        return d_list[:n]
    
    def get_best(self, n):
        d_list = list(self.dict.values())
        d_list.sort(key = lambda x: x.ptot() * x.ptxt, reverse = True)
        
        return d_list[:n]
    
def merge_logs(a, b):
    if a == -np.inf or b == -np.inf:
        return max(a, b)
    else:
        mx, mn = (a, b) if a > b else (b, a)
        
        return mx + torch.log1p(torch.exp(mn - mx)) # = b + ln(exp(a - b) + 1) = ln(exp(a) + exp(b))
        # return mx + np.log1p(np.exp(mn - mx)) # numpy equivalent for debugging

def test_fn(phonemes = True): # testing using saved predictions
    
    if phonemes:
        pred = pickle.load(open("test_preds_phones.pck", "rb"))
        to_targ = [None] + list(project_vars.phoneme_map.keys())
    else:
        pred = pickle.load(open("test_preds.pck", "rb"))
        to_targ = [None] + list(project_vars.letter_map.keys())
    
    print(pred[1])
    
    #beam = beam_decoder(pred[0], to_letter)
    exploded = explode_preds(pred[0], to_targ)
    print(exploded)
    greedy = greedy_decoder(pred[0], to_targ)
    print(greedy)
    log_beam = log_beam_decoder(pred[0], to_targ, phonemes)
    
    
    # print(beam)
    print("".join(log_beam))

def beam_decoder(pred, to_letter, n_beams = 3, n_chars = 5):
    
    char_model = pickle.load(open("lang_models\\char_model_{}gram.pck".format(str(n_chars)), "rb"))
    
    best_beams = [ctc_beam(pnb = 1, ptxt = .5)]
    # initialized = False
    
    for inc, step in enumerate(pred, 1):
        #print("{} of {}".format(inc, len(pred)))
        probs = torch.exp(step) # log scores to probabilities
        
        beams = ctc_beam_dict()
        
        for bb in best_beams:
            
            copy_beam = ctc_beam(init_string = bb.string, ptxt = bb.ptxt)
            copy_beam.pb = bb.ptot() * probs[0] # blank character
            beams.merge(copy_beam)
            
            for i, prob in enumerate(probs[1:], 1):
                new_beam = ctc_beam(init_string = bb.string)
                char = to_letter[i]
                
                if char == new_beam.last_char(): 
                    new_beam.pnb = bb.pb * prob # repeat character after blank
                    
                    copy_beam = ctc_beam(init_string = bb.string, ptxt = bb.ptxt) # repeated char with no blank
                    copy_beam.pnb = bb.pnb * prob
                    beams.merge(copy_beam)
                else: # new character
                    new_beam.pnb = bb.ptot() * prob
                    
                new_beam.set_ptxt(char, char_model, size = n_chars)
                new_beam.string += char
                
                beams.merge(new_beam)
                
        best_beams = beams.get_best(n_beams)
        # normalize_best(best_beams) # can normalize to prevent numerical underflow
        
    end_beam = beams.get_best(1)[0]
        
    return end_beam.string.replace(",", "")
            
def log_beam_decoder(pred, to_target, phonemes, n_beams = 3, n_chars = 5, c_txt = .5):
    
    lang_model = pickle.load(open("lang_models\\{}_model_{}gram.pck".format("phoneme" if phonemes else "char", str(n_chars)), "rb"))
    best_beams = [ctc_log_beam(pnb = 0, ptxt = math.log(.1))]
    
    for inc, step in enumerate(pred, 1):
        
        beams = ctc_beam_dict()
        
        for bb in best_beams:
            
            copy_beam = ctc_log_beam(init_string = bb.string, ptxt = bb.ptxt)
            copy_beam.pb = bb.set_log_ptot() + step[0] # add blank character
            beams.log_beam_merge(copy_beam)
            
            for i, log_prob in enumerate(step[1:], 1):
                new_beam = ctc_log_beam(init_string = bb.string)
                char = to_target[i]
                
                if char == new_beam.last_char(): 
                    new_beam.pnb = bb.pb + log_prob # repeat character after blank
                    
                    copy_beam = ctc_log_beam(init_string = bb.string, ptxt = bb.ptxt) # repeated char with no blank
                    copy_beam.pnb = bb.pnb + log_prob
                    beams.log_beam_merge(copy_beam)
                else: # new character
                    new_beam.pnb = bb.log_ptot + log_prob
                    
                new_beam.set_log_ptxt(char, lang_model, size = n_chars)
                new_beam.string += (char,)
                
                beams.log_beam_merge(new_beam)
                
        best_beams = beams.get_log_best(n_beams, c_txt)
        
    end_beam = beams.get_log_best(1, c_txt)[0]
        
    return end_beam.string
                    
def normalize_best(best): # can be used to normalize beam scores to prevent underflow in non-log decoder
    prob_list = []
    
    for beam in best:
        prob_list.append(beam.ptot())
    
    prob_sum = sum(prob_list)
        
    for beam in best:
        beam.pb /= prob_sum
        beam.pnb /= prob_sum
        
def list_scores(beams): # for debugging beam search, lists beams and some stats
    
    for i, beam in enumerate(beams, 1):
        print("{}) {}".format(i, beam))
        
def greedy_decoder(pred, to_letter):
    prev_guess = 0
    output = ""
    
    for step in pred:
        guess = torch.argmax(step)
        # guess = step.argmax()
        if guess and guess != prev_guess:
            output += to_letter[guess]
        prev_guess = guess
    
    return output

def explode_preds(pred, to_letter, show_blank = True):
    output = ""
    for step in pred:
        guess = step.argmax()
        if guess:
            output += to_letter[guess]
        elif show_blank:
            output += '~'
    
    return output

def phones_2_words(phones):
    phones_2_words = pickle.load(open("phones_2_words.pck", "rb"))
    word_phones = []
    sentence = ""
    for el in phones:
        if el == ' ':
            if tuple(word_phones) in phones_2_words:
                sentence += phones_2_words[tuple(word_phones)][0] + " "
            else:
                sentence += "".join(word_phones) + " "
            word_phones = []
        else:
            word_phones.append(el)
    
    return sentence[:-1]

def binary_search(scores, weighted_scores, best_score, best_c, last_c, jump, going_up, progress, curr_depth, depth, n_samples, model, to_target, device, phonemes): # using recursion makes you feel smart
    
    if curr_depth == depth:
        return best_c, best_score, scores, weighted_scores
    
    score = 0.0
    weighted_score_list = []
    new_c = last_c + (1 if going_up else -1) * jump
    
    for xy in data_gen_torch.get_n(device, phonemes, n = n_samples, is_train = True, seed = -1):
        pred = model(torch.unsqueeze(xy[0], 0))
        pred = pred.squeeze().T
        
        beam_output = log_beam_decoder(pred, to_target, phonemes, c_txt = new_c)
        beam_output = "".join(beam_output)
        
        new_score = wer(xy[1].split(' '), beam_output.split(' '))
        score += new_score
        weighted_score_list.append((len(xy[1].split(' ')), new_score))
        
        
    score /= n_samples # average wer
    scores.append((new_c, score))
    weighted_scores.append((new_c, weighted_average(weighted_score_list)))
    
    print("Depth: {}, Testing: {}, Score: {}".format(curr_depth + 1, new_c, score))
    
    if score < best_score:
        print("New Best!!!!!!!!!")
        best_score = score
        best_c = new_c
        jump /= 2
        progress = True
    else:
        going_up = not going_up
        new_c = last_c
        if not progress:
            jump /= 2
            print("zeroing in")
        progress = not progress
        
    return binary_search(scores, weighted_scores, best_score, best_c, new_c, jump, going_up, progress, curr_depth + 1, depth, n_samples, model, to_target, device, phonemes)
            
def binary_search_init(floor, ceil, depth, n_samples = 5, phonemes = False, is_2d = False, norm = False, model_name = "best_weights.pt", save = True): # finds best value for c_txt between floor (inclusive) and ceil(exclusive)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if phonemes:
        model = models.wav2phoneme_torch_2d() if is_2d else models.wav2phoneme_torch()
        to_target = [None] + list(project_vars.phoneme_map.keys())
    else:
        model = models.wav2letter_torch_2d() if is_2d else models.wav2letter_torch_test2()
        to_target = [None] + list(project_vars.letter_map.keys())
        
    model.to(device)
    model_path = "models\\{}\\{}\\{}\\{}".format("phonemes" if phonemes else "letters", model.__class__.__name__, project_vars.base_channels, "norm\\" if norm else "")
    Path(model_path).mkdir(parents=True, exist_ok=True)
    model.load_state_dict(torch.load(model_path + model_name))
    
    best_c, best_score, scores, appended_scores = binary_search([], [], 100, None, -(ceil - floor), (ceil - floor), True, True, 0, depth, n_samples, model, to_target, device, phonemes)
    
    make_plot(scores, model.__class__.__name__, model_name, phonemes, n_samples, "binary", save)
    make_plot(scores, model.__class__.__name__, model_name, phonemes, n_samples, "binary - appended", save)
    make_plot_combined([[scores, "Average"], [appended_scores, "Appended"]], model.__class__.__name__, model_name, phonemes, n_samples, "binary - combined", save)
    
def grid_search(start, end, step = .1, n_samples = 5, phonemes = False, is_2d = False, model_name = "best_weights.pt", norm = False, save = True, seed = 0):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if phonemes:
        model = models.wav2phoneme_torch_2d() if is_2d else models.wav2phoneme_torch()
        to_target = [None] + list(project_vars.phoneme_map.keys())
    else:
        model = models.wav2letter_torch_2d() if is_2d else models.wav2letter_torch_test2()
        to_target = [None] + list(project_vars.letter_map.keys())
        
    model.to(device)
    model_path = "models\\{}\\{}\\{}\\{}".format("phonemes" if phonemes else "letters", model.__class__.__name__, project_vars.base_channels, "norm\\" if norm else "")
    Path(model_path).mkdir(parents=True, exist_ok=True)
    model.load_state_dict(torch.load(model_path + model_name))
    
    scores, weighted_scores = [], []
    best_score = 100
    
    for c in np.arange(start, end + step, step):
        print("checking {}".format(round(c, 2)))
        total_score = 0.0
        weighted_score_list = []
        for xy in data_gen_torch.get_n(device, phonemes, n = n_samples, is_train = True, seed = seed):
            pred = model(torch.unsqueeze(xy[0], 0))
            pred = pred.squeeze().T
            
            beam_output = log_beam_decoder(pred, to_target, phonemes, c_txt = c)
            beam_output = "".join(beam_output)
            
            new_score = wer(xy[1].split(' '), beam_output.split(' '))
            total_score += new_score
            weighted_score_list.append((len(xy[1].split(' ')), new_score))
            
        total_score /= n_samples # average wer
        scores.append((c, total_score))
            
        weighted_scores.append((c, weighted_average(weighted_score_list)))
        
        if total_score < best_score:
            best_score = total_score
            best_c = c
        
    print("Best Score: {}, C: {}".format(best_score, best_c))
    
    make_plot(scores, model.__class__.__name__, model_name, phonemes, n_samples, "grid", save)
    make_plot(weighted_scores, model.__class__.__name__, model_name, phonemes, n_samples, "grid - weighted", save)
    make_plot_combined([[scores, "Average"], [weighted_scores, "Weighted"]], model.__class__.__name__, model_name, phonemes, n_samples, "grid - combined", save)
        
def make_plot(scores, class_name, model_name, phonemes, n_samples, method, save):
    title = "{} - Name {} - Method {} - Phonemes {} - n {}".format(class_name, model_name, method, phonemes, n_samples)
    plt.title(class_name + " - " + model_name)
    plt.scatter(*zip(*scores))
    
    if save:
        plt.savefig("saved_plots\\" + title + ".png")
    else:
        plt.show()
        
    plt.clf()
        
def make_plot_combined(scores, class_name, model_name, phonemes, n_samples, method, save):
    title = "{} - Name {} - Method {} - Phonemes {} - n {}".format(class_name, model_name, method, phonemes, n_samples)
    plt.title(class_name + " - " + model_name)
    for set in scores:
        plt.scatter(*zip(*set[0]), label = set[1])
    plt.legend()
    
    if save:
        plt.savefig("saved_plots\\" + title + ".png")
    else:
        plt.show()
        
    plt.clf()

def weighted_average(weight_scores):
    num, den = 0, 0
    for el in weight_scores:
        num += el[0] * el[1]
        den += el[0]
        
    return num / den

# test_fn(phonemes = False)
# show_predictions(3, phonemes = False, norm = False)

# grid_search(.3, .7, .15, n_samples = 5)
# binary_search_init(0, 2, 8, n_samples = 30)
# grid_search(.3, 1.95, .15, n_samples = 40)