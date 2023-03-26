import torch
import torch.nn as nn
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random
import tqdm
import sys
import shutil
import copy
import os

from model import MidiBert
from modelLM import MidiBertLM


class BERTTrainer:
    def __init__(self, midibert: MidiBert, train_dataloader, valid_dataloader,
                 lr, batch, max_seq_len, mask_percent, accum_grad, cpu, cuda_devices=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        self.midibert = midibert  # save this for ckpt
        self.model = MidiBertLM(midibert).to(self.device)
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('# total parameters:', self.total_params)

        if torch.cuda.device_count() > 1 and not cpu:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        # For regular training
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        # For Warmup (linear)
        # self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0)
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optim, start_factor=0.00000001, end_factor=1,
        #                                                    total_iters=200, last_epoch=-1, verbose=False)

        # For loading from checkpoint
        # self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        # checkpoint = torch.load('F:/CDF/MIDI-BERT/result/pretrain/Pretrain1960/model.ckpt')
        # self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

        self.batch = batch
        self.max_seq_len = max_seq_len
        self.mask_percent = mask_percent
        self.Lseq = [i for i in range(self.max_seq_len)]
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.accum_grad = accum_grad

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def get_mask_ind(self):
        mask_ind = random.sample(self.Lseq, round(self.max_seq_len * self.mask_percent))
        mask80 = random.sample(mask_ind, round(len(mask_ind) * 0.8))
        left = list(set(mask_ind) - set(mask80))
        rand10 = random.sample(left, round(len(mask_ind) * 0.1))
        cur10 = list(set(left) - set(rand10))
        return mask80, rand10, cur10

    def train(self):
        torch.cuda.empty_cache()
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, self.max_seq_len)
        torch.cuda.empty_cache()
        return train_loss, train_acc

    def valid(self):
        torch.cuda.empty_cache()
        self.model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = self.val_iteration(self.valid_data, self.max_seq_len, train=False)
        torch.cuda.empty_cache()
        return valid_loss, valid_acc

    def iteration(self, training_data, max_seq_len, train=True):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_losses = [0] * len(self.midibert.e2w), 0
        avg_accum_acc, avg_accum_loss = [0] * len(self.midibert.e2w), 0

        for count, ori_seq_batch in enumerate(pbar):
            # To warmup, not using data in the front so it will not be processed by AdamW again immediately
            # if count < 20000:
            #     continue
            batch = ori_seq_batch.shape[0]
            ori_seq_batch = ori_seq_batch.to(self.device)  # (batch, seq_len, 4) 
            input_ids = copy.deepcopy(ori_seq_batch)
            loss_mask = torch.zeros(batch, max_seq_len)

            for b in range(batch):
                # get index for masking
                mask80, rand10, cur10 = self.get_mask_ind()
                # apply mask, random, remain current token
                for i in mask80:
                    mask_word = torch.tensor(self.midibert.mask_word_np).to(self.device)
                    input_ids[b][i] = mask_word
                    loss_mask[b][i] = 1
                for i in rand10:
                    rand_word = torch.tensor(self.midibert.get_rand_tok()).to(self.device)
                    input_ids[b][i] = rand_word
                    loss_mask[b][i] = 1
                for i in cur10:
                    loss_mask[b][i] = 1

            loss_mask = loss_mask.to(self.device)

            # avoid attend to pad word
            attn_mask = (input_ids[:, :, 0] != self.midibert.bar_pad_word).float().to(self.device)  # (batch, seq_len)

            y = self.model.forward(input_ids, attn_mask)

            # get the most likely choice with max
            outputs = []
            for i, etype in enumerate(self.midibert.e2w):
                output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                outputs.append(output)
            outputs = np.stack(outputs, axis=-1)
            outputs = torch.from_numpy(outputs).to(self.device)  # (batch, seq_len)

            # accuracy
            all_acc = []
            for i in range(4):
                acc = torch.sum((ori_seq_batch[:, :, i] == outputs[:, :, i]).float() * loss_mask)
                acc /= torch.sum(loss_mask)
                all_acc.append(acc)
            total_acc = [sum(x) for x in zip(total_acc, all_acc)]

            # reshape (b, s, f) -> (b, f, s)
            for i, etype in enumerate(self.midibert.e2w):
                # print('before',y[i][:,...].shape)   # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
                y[i] = y[i][:, ...].permute(0, 2, 1)

            # calculate losses
            losses, n_tok = [], []
            for i, etype in enumerate(self.midibert.e2w):
                n_tok.append(len(self.midibert.e2w[etype]))
                losses.append(self.compute_loss(y[i], ori_seq_batch[..., i].to(self.device, torch.long), loss_mask))
            total_loss_all = [x * y for x, y in zip(losses, n_tok)]
            total_loss = sum(total_loss_all) / sum(n_tok)  # weighted
            total_loss = total_loss / self.accum_grad
            avg_accum_loss += total_loss

            # udpate only in train
            if train:
                # Accumulated gradient
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.0)
                if (count + 1) % self.accum_grad == 0:
                    self.optim.step()
                    self.model.zero_grad()

            # acc
            accs = list(map(float, all_acc))
            tempaccs = np.array(accs)
            avg_accum_acc += tempaccs / self.accum_grad
            sys.stdout.write(
                'Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f} | acc: {:03f}, {:03f}, {:03f}, {:03f} \r'.format(
                    total_loss * self.accum_grad, *losses, *accs))
            if (count + 1) % self.accum_grad == 0:
                # print('Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f} | acc: {:03f}, {:03f}, {:03f}, {:03f} \r'.format(
                # avg_accum_loss, *losses, *avg_accum_acc.tolist()))
                with open(os.path.join('D:/UT/ECE324/MIDI-BERT/result/pretrain/Pretrain1960/', 'full_log'), 'a') as outfile:
                    outfile.write(
                        'Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f} | acc: {:03f}, {:03f}, {:03f}, {:03f} \r'.format(
                            avg_accum_loss, *losses, *avg_accum_acc.tolist()))
                print(
                    'Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f} | acc: {:03f}, {:03f}, {:03f}, {:03f} \n'.format(
                        avg_accum_loss, *losses, *avg_accum_acc.tolist()))
                avg_accum_acc, avg_accum_loss = [0] * len(self.midibert.e2w), 0
                self.scheduler.step()

            losses = list(map(float, losses))
            total_losses += total_loss.item()

            del ori_seq_batch, y, outputs

            # To save warmup checkpoint
            # if count == 20000+6400:
            #     state = {
            #         'epoch': 0,
            #         'state_dict': self.midibert.state_dict(),
            #         'best_acc': 0,
            #         'valid_acc': 0,
            #         'valid_loss': 0,
            #         'train_loss': 0,
            #         'optimizer': self.optim.state_dict()
            #     }
            #
            #     torch.save(state, 'D:/UT/ECE324/MIDI-BERT/result/pretrain/Pretrain1960/warmup.ckpt')
            #     break

        return round(total_losses / len(training_data), 3), [round(x.item() / len(training_data), 3) for x in total_acc]

    def val_iteration(self, training_data, max_seq_len, train=True):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_losses = [0] * len(self.midibert.e2w), 0

        with torch.no_grad():
            for count, ori_seq_batch in enumerate(pbar):
                batch = ori_seq_batch.shape[0]
                ori_seq_batch = ori_seq_batch.to(self.device)  # (batch, seq_len, 4) 
                input_ids = copy.deepcopy(ori_seq_batch)
                loss_mask = torch.zeros(batch, max_seq_len)

                for b in range(batch):
                    # get index for masking
                    mask80, rand10, cur10 = self.get_mask_ind()
                    # apply mask, random, remain current token
                    for i in mask80:
                        mask_word = torch.tensor(self.midibert.mask_word_np).to(self.device)
                        input_ids[b][i] = mask_word
                        loss_mask[b][i] = 1
                    for i in rand10:
                        rand_word = torch.tensor(self.midibert.get_rand_tok()).to(self.device)
                        input_ids[b][i] = rand_word
                        loss_mask[b][i] = 1
                    for i in cur10:
                        loss_mask[b][i] = 1

                loss_mask = loss_mask.to(self.device)

                # avoid attend to pad word
                attn_mask = (input_ids[:, :, 0] != self.midibert.bar_pad_word).float().to(
                    self.device)  # (batch, seq_len)

                y = self.model.forward(input_ids, attn_mask)

                # get the most likely choice with max
                outputs = []
                for i, etype in enumerate(self.midibert.e2w):
                    output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                    outputs.append(output)
                outputs = np.stack(outputs, axis=-1)
                outputs = torch.from_numpy(outputs).to(self.device)  # (batch, seq_len)

                # accuracy
                all_acc = []
                for i in range(4):
                    acc = torch.sum((ori_seq_batch[:, :, i] == outputs[:, :, i]).float() * loss_mask)
                    acc /= torch.sum(loss_mask)
                    all_acc.append(acc)
                total_acc = [sum(x) for x in zip(total_acc, all_acc)]

                # reshape (b, s, f) -> (b, f, s)
                for i, etype in enumerate(self.midibert.e2w):
                    # print('before',y[i][:,...].shape)   # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
                    y[i] = y[i][:, ...].permute(0, 2, 1)

                # calculate losses
                losses, n_tok = [], []
                for i, etype in enumerate(self.midibert.e2w):
                    n_tok.append(len(self.midibert.e2w[etype]))
                    losses.append(self.compute_loss(y[i], ori_seq_batch[..., i].to(self.device, torch.long), loss_mask))
                total_loss_all = [x * y for x, y in zip(losses, n_tok)]
                total_loss = sum(total_loss_all) / sum(n_tok)  # weighted
                total_loss = total_loss / self.accum_grad

                # udpate only in train
                if train:
                    # Accumulated gradient
                    total_loss.backward()
                    clip_grad_norm_(self.model.parameters(), 3.0)
                    if (count + 1) % self.accum_grad == 0:
                        self.optim.step()
                        self.model.zero_grad()

                # acc
                accs = list(map(float, all_acc))
                sys.stdout.write(
                    'Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f} | acc: {:03f}, {:03f}, {:03f}, {:03f} \r'.format(
                        total_loss, *losses, *accs))

                losses = list(map(float, losses))
                total_losses += total_loss.item()

                del ori_seq_batch, y, outputs

        return round(total_losses / len(training_data), 3), [round(x.item() / len(training_data), 3) for x in total_acc]

    def save_checkpoint(self, epoch, best_acc, valid_acc,
                        valid_loss, train_loss, is_best, filename):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.midibert.state_dict(),
            'best_acc': best_acc,
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'optimizer': self.optim.state_dict()
        }

        torch.save(state, filename)

        best_mdl = filename.split('.')[0] + '_best.ckpt'
        if is_best:
            shutil.copyfile(filename, best_mdl)
