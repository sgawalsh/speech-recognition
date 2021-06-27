# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:02:14 2021

@author: sgawalsh
"""

import torch
import torch.nn as nn
import models

import data_gen_torch
import math
import project_vars

# from copy import deepcopy
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
from datetime import datetime
from os import listdir, system, path
from pathlib import Path
from pickle import dump, load

class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]

def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + math.cos(math.pi*t/t_max))/2
    
    return scheduler
    
def train_single(epochs = 3, draw_plot = True, roll_avg_period = 100):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = models.wav2letter_torch()
    model.to(device)
    
    ctc_loss = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    for param_group in optimizer.param_groups:
        print("Using lr: {}".format(param_group['lr']))
    
    losses = []
    i, avg_sum = 0, 0
    
    for epoch in range(epochs):
        print("Epoch {} of {}".format(epoch + 1, epochs))
        gen = data_gen_torch.data_generator_torch_single(device)
        for x, y, y_length in gen:
            
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("found nan or inf")
            x = x.reshape(1, project_vars.base_channels * 3, -1)
            try:
                y_pred = model(x) # B*C*T
            except RuntimeError:
                continue
            y_pred_length = torch.full(size=(1,), fill_value=y_pred.shape[2], dtype=torch.int32).to(device)
            
            
            if y_pred_length.item() <= y_length.item() :# targ length must be less than pred length
                continue
                
            y_pred = torch.transpose(y_pred, 0, 1)# -> C*B*T
            y_pred = torch.transpose(y_pred, 0, 2)# -> T*B*C
            
            if torch.isnan(y_pred.detach()).any() or torch.isinf(y_pred.detach()).any():
                print("found nan or inf in preds")
            
            loss = ctc_loss(y_pred, y, y_pred_length, y_length)
            #print(round(loss.item(), 2))
            if torch.isinf(loss):
                print(y_pred.shape, y.shape, y_pred_length, y_length)
                print("infinite loss, exiting")
                return
            #losses.append(loss.item())
            avg_sum += loss.item()
            i += 1
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
            
            if not i % roll_avg_period:
                print("Rolling Average: {}".format(round(avg_sum / roll_avg_period, 2)))
                losses.append(avg_sum / roll_avg_period)
                i, avg_sum = 0, 0
                
    # if draw_plot:
    #     plt.plot(losses)
    #     plt.show()

def train_batch(epochs = 10, batch_size = 10, roll_avg_period = 200, patience = 10, load_model = True, logs = True, model_name = "best_weights.pt", learning_rate = 1e-5, phonemes = False, shutdown = False, norm = False, is_2d = False, reset = True, batch_loops = 1):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if phonemes:
        model = models.wav2phoneme_torch_2d() if is_2d else models.wav2phoneme_torch()
    else:
        model = models.wav2letter_torch_2d() if is_2d else models.wav2letter_torch_test2()
        
    model_path = "models\\{}\\{}\\{}\\".format("phonemes" if phonemes else "letters", model.__class__.__name__, project_vars.base_channels) + ("norm\\" if norm else "")
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
    model.to(device)
    loss_fn = nn.CTCLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-4, max_lr = 1e-3, step_size_up=100, cycle_momentum=False, verbose = False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, threshold = 0.001)
    
    #losses = []
    #roll_avg_period = round(roll_avg_period / batch_size)
    
    # lr_history = []
    no_improvements = 0
    fmt = '[{epoch:03d}/{total:03d}] train: {train:.4f} - eval: {eval:.4f}'
    
    if logs:
        sess_name = model.__class__.__name__ + " - " + datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        log_file = open("logs\\{}.txt".format(sess_name), "w")
        log_file.write("Using learning rate of {}\n".format(learning_rate))
        log_file.close()
    else:
        sess_name = None
        
    if load_model:
        model.load_state_dict(torch.load(model_path + model_name))
        best_loss = eval_cycle(model, device, phones = phonemes, norm = norm)
        print_log_msg("Using initial loss value of {}".format(round(best_loss, 3)), sess_name)
        # best_weights = deepcopy(model.state_dict())
        try:
            history = load(open(model_path + model_name + "_history.pck", "rb"))
        except:
            history = []
    else:
        best_loss = float("Inf")
        history = []
    
    try:
        for epoch in range(epochs):
            stats = {'epoch': epoch + 1, 'total': epochs}
            
            # print("Epoch {} of {}".format(epoch + 1, epochs))
            for phase in ["train", "eval"]:
                training_phase = phase == 'train'
                
                # print("{} phase!".format(phase))
                if training_phase:
                    model.train()
                else:
                    model.eval()
                    
                gen = data_gen_torch.phoneme_generator_torch_batch(batch_size, device, is_train = training_phase, norm = norm, is_2d = is_2d) if phonemes else data_gen_torch.data_generator_torch_batch(batch_size, device, is_train = training_phase, norm = norm, is_2d = is_2d)
                
                # j, avg_sum = 0, 0
                i, batch_loop_inc, running_loss = 0, 0, 0.0
                
                for x, y, y_lengths in gen:
                    
                    y_preds = []
                    # pred_lengths = torch.zeros(size = (len(y_lengths), ), dtype=torch.int32)
                    pred_lengths = torch.IntTensor([math.floor(el.shape[1] / 2 - 1) for el in x])
                    
                    for idx, in_data in enumerate(x): # predictions are made individually due to varied length
                        pred = model(torch.unsqueeze(in_data, 0)) # 1 * C * T
                        y_preds.append(pred)
                    
                    # for pair in zip(pred_lengths, y_lengths): # targ length must be less than pred length
                    #     if pair[0] < pair[1]:
                    #         print("skipping")
                    #         continue
                        
                    y_preds = [el.squeeze().T for el in y_preds] # -> T * C
                    y_preds = torch.nn.utils.rnn.pad_sequence(y_preds) # -> T * B * C
                    
                    # if torch.isnan(y_preds.detach()).any() or torch.isinf(y_preds.detach()).any():
                    #     print("found nan or inf in preds")
                    
                    loss = loss_fn(y_preds, torch.Tensor(y), pred_lengths, torch.Tensor(y_lengths).to(torch.int32))
                    
                    running_loss += loss.item()
                    i += 1
                    
                    if training_phase:
                        batch_loop_inc += 1
                        loss.backward()
                        if batch_loop_inc == batch_loops:
                            optimizer.step()
                            optimizer.zero_grad()
                            batch_loop_inc = 0
                            
                        
                        # scheduler.step() # cyclic scheduler
                        # lr_history.extend(scheduler.get_last_lr())
                        
                        # avg_sum += loss.item()
                        # j += 1
                        # if not j % roll_avg_period:
                            # print("Rolling Average: {}".format(round(avg_sum / roll_avg_period, 3)))
                            # losses.append(avg_sum / roll_avg_period)
                            # j, avg_sum = 0, 0
                
                epoch_loss = running_loss / i
                stats[phase] = epoch_loss
                
                if not training_phase:
                    scheduler.step(epoch_loss) # plateau scheduler
                    history.append(stats)
                    print_log_msg(fmt.format(**stats), sess_name)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        print_log_msg('Loss improvement on epoch {}! New best: {}'.format(epoch + 1, round(best_loss, 3)), sess_name)
                        # best_weights = deepcopy(model.state_dict())
                        torch.save(model.state_dict(), model_path + model_name)
                        no_improvements = 0
                    else:
                        no_improvements += 1
                        print_log_msg("No improvement: {} > {}".format(round(epoch_loss, 3), round(best_loss, 3)), sess_name)
                        if reset:
                            model.load_state_dict(torch.load(model_path + model_name))
                        
            if no_improvements >= patience:
                print_log_msg("Out of patience!", sess_name)
                break
                
        # if draw_plot:
        #     plt.plot(losses)
        #     plt.show()
            
        # torch.save(best_weights, model_path + model_name)
        dump(history, open(model_path + model_name + "_history.pck", "wb"))
        
    except Exception as e:
        print_log_msg(e, sess_name)
        
        try:
            torch.save(model.state_dict(), model_path + "fail_weights.pt")
            print_log_msg("saved weights", sess_name)
        except Exception as e:
            print_log_msg("Unable to save weights", sess_name)
            print_log_msg(e, sess_name)
            
    if shutdown:
        system('shutdown -s')

def alt_loop(x, loss_fn, model, y, y_lengths): # alternative training loop where inputs are padded, slower than non-padded inputs
    
    preds = model((torch.nn.utils.rnn.pad_sequence([el.T for el in x])).permute(1, 2, 0))
    loss = loss_fn(preds.permute(2, 0, 1), torch.Tensor(y), torch.IntTensor([math.floor(el.shape[1] / 2 - 1) for el in x]), torch.Tensor(y_lengths).to(torch.int32))
    
    return loss
        
def eval_cycle(model, device, batch_size = 10, phones = False, norm = False): # tests model on eval dataset and returns score
    
    ctc_loss = nn.CTCLoss()
    
    gen = data_gen_torch.phoneme_generator_torch_batch(batch_size, device, is_train = False, norm = norm) if phones else data_gen_torch.data_generator_torch_batch(batch_size, device, is_train = False, norm = norm)
    i = 0
    running_loss = 0.0
    
    for x, y, y_lengths in gen:
        
        y_preds = []
        pred_lengths = torch.zeros(size = (len(y_lengths), ), dtype=torch.int32)
        
        for idx, in_data in enumerate(x):
            pred = model(torch.unsqueeze(in_data, 0))
            pred_lengths[idx] = pred.shape[2]
            y_preds.append(pred)
            
        y_preds = [el.squeeze().T for el in y_preds] # -> T * C
        y_preds = torch.nn.utils.rnn.pad_sequence(y_preds) # -> T * B * C
        
        loss = ctc_loss(y_preds, torch.Tensor(y), pred_lengths, torch.Tensor(y_lengths).to(torch.int32))
        
        running_loss += loss.item()
        i += 1
    
    return running_loss / i

def test_weights(model_class, phones = False, norm = False): # takes existing weights and outputs eval scores
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_path = "models\\{}\\{}\\{}\\".format("phonemes" if phones else "letters", model_class().__class__.__name__, project_vars.base_channels) + ("norm\\" if norm else "")
    
    model_weights = [f for f in listdir(model_path) if path.isfile(path.join(model_path, f))]
    eval_scores = []
    
    for model_name in model_weights:
        
        model = model_class()
        model.to(device)
        
        try:
            model.load_state_dict(torch.load(model_path + model_name))
        except:
            continue
        
        print(model_name)
        
        score = eval_cycle(model, device, phones = phones, norm = norm)
        
        eval_scores.append((model_name, score))
        
    eval_scores.sort(key = lambda x: x[1])
    
    sess_name = "Rankings - " + model.__class__.__name__ + " - " + datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    print_log_msg("Rankings:", sess_name, "w")
    
    for i, score in enumerate(eval_scores, 1):
        print_log_msg("{}) {}: {}".format(str(i), score[0], str(round(score[1], 3))), sess_name)

def print_log_msg(msg, sess_name, mode = "a"):
    print(msg)
    if sess_name:
        with open("logs\\{}.txt".format(sess_name), mode) as f:
            f.write(str(msg) + "\n")
            
def plot_history(history): # plots training and eval score progress over epochs
    train_scores, eval_scores = [], []
    for el in history:
        train_scores.append(el["train"])
        eval_scores.append(el["eval"])
        
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Loss Scores')
    ax1.plot(train_scores)
    ax1.set_title("Train")
    ax2.plot(eval_scores)
    ax2.set_title("Eval")
    
    # fig.show()
    
    # input("hi")
    
def save_new(): # save existing weights to new model
    model = models.wav2letter_torch_test2()
    model.load_state_dict(torch.load("models\\letters\\wav2letter_torch\\13\\best_weights.pt"))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    e_loss = eval_cycle(model, device, batch_size = 10, phones = False, norm = False)
    print(e_loss)
    torch.save(model.state_dict(), "models\\letters\\wav2letter_torch_test2\\13\\best_weights.pt")
    
# save_new()

train_batch(
    batch_size = 20,
    epochs = 15,
    learning_rate = 3e-7,
    shutdown = True,
    load_model = True,
    phonemes = False,
    logs = True,
    is_2d = False,
    norm = False,
    batch_loops = 10,
    reset = False)

# plot_history(load(open("models\\letters\\wav2letter_torch_test\\13\\norm\\best_weights.pt_history.pck", "rb")))

# train_single(epochs = 3, draw_plot = True, roll_avg_period = 100)
#test_weights(models.wav2letter_torch, phones = False)