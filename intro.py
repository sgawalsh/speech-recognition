# -*- coding: utf-8 -*-
"""
Created on Thu May 27 03:43:06 2021

@author: sgawalsh
"""

from playsound import playsound
import os
import models
import torch
import soundfile as sf
import librosa
from numpy import diff, concatenate
import project_vars
from decoder import log_beam_decoder

def intro(phonemes, c_txt = .5, n_beams = 3, norm = True, model_name = "best_weights.pt", write_file = True):

    print("c_lang = {}, n_beams = {}\n".format(c_txt, n_beams))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if phonemes:
        model = models.wav2phoneme_torch()
        model.load_state_dict(torch.load("models\\phonemes\\best_weights.pt"))
        to_targ = [None] + list(project_vars.phoneme_map.keys())
    else:
        model = models.wav2letter_torch()
        to_targ = [None] + list(project_vars.letter_map.keys())
        
    model_path = "models\\{}\\{}\\{}\\{}".format("phonemes" if phonemes else "letters", model.__class__.__name__, project_vars.base_channels, "norm\\" if norm else "")
    model.load_state_dict(torch.load(model_path + model_name))
    model.to(device)

    for i, a_f in enumerate(os.listdir("intro\\audio\\"), 1):
        try:
            data, fs = sf.read(os.path.join("intro\\audio\\", a_f))
            f_name = a_f.split(".")[0]
        except:
            continue
        
        t_data = librosa.feature.mfcc(data, 16000, n_mfcc = project_vars.base_channels, hop_length = 160, win_length = 400) # hop_length = 10 ms, win_length = 25 ms
        d_1 = diff(t_data, prepend = 0) # 1st derivative
        d_2 = diff(d_1, prepend = 0) # 2nd derivative
        
        in_data = concatenate([t_data, d_1, d_2])
        if norm:
            in_data = (in_data - in_data.mean()) / in_data.std()
        
        x = torch.from_numpy(in_data).float().to(device)
        pred = model(torch.unsqueeze(x, 0))
        
        with open("intro\\trans\\{}.txt".format(f_name)) as f:
                print("Target: {}".format(f.read()))
                
        playsound("intro\\wav\\{}.wav".format(f_name))
        
        for c in c_txt:
            output = log_beam_decoder(pred.squeeze().T, to_targ, phonemes, n_beams = n_beams, n_chars = 5, c_txt = c)
            print("Predicted: {}\n".format("".join(output)))
        
        # if write_file:
            # sf.write("intro\\wav\\{}.wav".format(f_name), data, fs) 
            # playsound("intro\\wav\\{}.wav".format(f_name))
        

beams = 3

intro(False, c_txt = [.8], n_beams = beams, write_file = True)