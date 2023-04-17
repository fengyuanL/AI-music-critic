import shutil
import numpy as np
import tqdm
import torch
import torch.nn as nn
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_

from MidiBERT.finetune_model import TokenClassification, SequenceClassification, SequenceRegression


class FinetuneTrainer:
    def __init__(self, midibert, train_dataloader, valid_dataloader, test_dataloader, layer,
                 lr, class_num, hs, testset_shape, cpu, cuda_devices=None, model=None, SeqClass=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        # print('   device:', self.device)
        self.midibert = midibert
        self.SeqClass = SeqClass
        self.layer = layer

        if model != None:  # load model
            # print('load a fine-tuned model')
            self.model = model.to(self.device)
        else:
            # print('init a fine-tune model, sequence-level task?', SeqClass)
            if SeqClass and class_num != 2:
                self.model = SequenceClassification(self.midibert, class_num, hs).to(self.device)
            elif SeqClass and class_num == 2:
                self.model = SequenceRegression(self.midibert, 1, hs).to(self.device)
                checkpoint = torch.load('result/finetune/acclaimed_1820_1/FTmodel_best.ckpt')
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model = TokenClassification(self.midibert, class_num, hs).to(self.device)

        #        for name, param in self.model.named_parameters():
        #            if 'midibert.bert' in name:
        #                    param.requires_grad = False
        #            print(name, param.requires_grad)

        if torch.cuda.device_count() > 1 and not cpu:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader

        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.reg_loss_func = nn.MSELoss(reduction='mean')

        self.testset_shape = testset_shape

    def compute_loss(self, predict, target, loss_mask, seq):
        loss = self.loss_func(predict, target)
        if not seq:
            loss = loss * loss_mask
            loss = torch.sum(loss) / torch.sum(loss_mask)
        else:
            loss = torch.sum(loss) / loss.shape[0]
        return loss

    def reg_compute_loss(self, predict, target, loss_mask, seq):
        loss = self.reg_loss_func(predict, target)
        if not seq:
            loss = loss * loss_mask
            loss = torch.sum(loss) / torch.sum(loss_mask)
        else:
            loss = torch.sum(loss) / 1
        return loss

    def accuracy(self, t1, t2):
        if len(t1) != len(t2):
            print("Warning: Not equal length in accuracy")
        if len(t1) > len(t2):
            t1 = t1[:len(t2)]
        elif len(t2) > len(t1):
            t2 = t2[:len(t1)]
        return torch.tensor(sum(torch.logical_and(t1, t2)) / len(t1), dtype=float)

    def accuracy_local(self, t1, t2):
        # t1 is the reference
        res = 0
        t_count = 0
        for count, i in enumerate(t1):
            if t1[count]:
                t_count += 1
                if t2[count] == True:
                    res += 1
        if t_count == 0:
            return torch.tensor(-1, dtype=float)
        return torch.tensor(res / t_count, dtype=float)

    def train(self):
        self.model.train()
        train_loss, train_acc, train_accl, train_accr, train_acclf, train_acclt, train_type1, train_type2 = self.reg_iteration(
            self.train_data, 0, self.SeqClass)
        return train_loss, train_acc, train_accl, train_accr, train_acclf, train_acclt, train_type1, train_type2

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            valid_loss, valid_acc, valid_accl, valid_accr, valid_acclf, valid_acclt, val_type1, val_type2 = self.reg_iteration(
                self.valid_data, 1, self.SeqClass)
        return valid_loss, valid_acc, valid_accl, valid_accr, valid_acclf, valid_acclt, val_type1, val_type2

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_loss, test_acc, test_accl, test_accr, test_acclf, test_acclt, test_type1, test_type2, all_output = self.reg_iteration(
                self.test_data, 2, self.SeqClass)
        return test_loss, test_acc, test_accl, test_accr, test_acclf, test_acclt, test_type1, test_type2, all_output

    def iteration(self, training_data, mode, seq):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_cnt, total_loss = 0, 0, 0

        if mode == 2:  # testing
            all_output = torch.empty(self.testset_shape)
            cnt = 0

        for x, y in pbar:  # (batch, 512, 768)
            batch = x.shape[0]
            x, y = x.to(self.device, dtype=float), y.to(self.device,
                                                        dtype=float)  # seq: (batch, 512, 4), (batch) / token: , (batch, 512)

            # avoid attend to pad word
            if not seq:
                attn = (y != 0).float().to(self.device)  # (batch,512)
            else:
                attn = torch.ones((batch, 512)).to(self.device)  # attend each of them

            y_hat = self.model.forward(x, attn, self.layer)  # seq: (batch, class_num) / token: (batch, 512, class_num)

            # get the most likely choice with max
            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = torch.from_numpy(output).to(self.device)
            if mode == 2:
                all_output[cnt: cnt + batch] = output
                cnt += batch

            # accuracy
            if not seq:
                acc = torch.sum((y == output).float() * attn)
                total_acc += acc
                total_cnt += torch.sum(attn).item()
            else:
                acc = torch.sum((y == output).float())
                total_acc += acc
                total_cnt += y.shape[0]

            # calculate losses
            if not seq:
                y_hat = y_hat.permute(0, 2, 1)
            loss = self.compute_loss(y_hat, y, attn, seq)
            total_loss += loss.item()

            # udpate only in train
            if mode == 0:
                self.model.zero_grad()
                loss.backward()
                self.optim.step()

        if mode == 2:
            return round(total_loss / len(training_data), 4), round(total_acc.item() / total_cnt, 4), all_output
        return round(total_loss / len(training_data), 4), round(total_acc.item() / total_cnt, 4)

    def reg_iteration(self, training_data, mode, seq):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc = 0
        total_accl = 0
        total_accr = 0
        total_acclf = 0
        total_acclt = 0
        total_type1 = 0
        total_type2 = 0
        total_cnt = 0
        total_lf_cnt = 0
        total_lt_cnt = 0
        total_loss = 0
        local_acc = 0
        local_accl = 0
        local_accr = 0
        local_acclf = 0
        local_acclt = 0
        local_type1 = 0
        local_type2 = 0
        local_cnt = 0
        local_lf_cnt = 0
        local_lt_cnt = 0
        local_loss = 0

        # total_acc = torch.tensor(0, dtype=float, device=self.device)
        # total_accl = torch.tensor(0, dtype=float, device=self.device)
        # total_accr  = torch.tensor(0, dtype=float, device=self.device)
        # total_acclf = torch.tensor(0, dtype=float, device=self.device)
        # total_acclt = torch.tensor(0, dtype=float, device=self.device)
        # total_type1 = torch.tensor(0, dtype=float, device=self.device)
        # total_type2 = torch.tensor(0, dtype=float, device=self.device)
        # total_cnt = torch.tensor(0, dtype=float, device=self.device)
        # total_lf_cnt = torch.tensor(0, dtype=float, device=self.device)
        # total_lt_cnt = torch.tensor(0, dtype=float, device=self.device)
        # total_loss = torch.tensor(0, dtype=float, device=self.device)

        accum_grad = 1

        if mode == 2:  # testing
            all_output = torch.empty(self.testset_shape)
            cnt = 0

        for count, (x, y) in enumerate(pbar):  # (batch, 512, 768)
            batch = x.shape[0]
            x, y = x.to(self.device), y.to(self.device)  # seq: (batch, 512, 4), (batch) / token: , (batch, 512)

            if len(x) != training_data.batch_size:
                count += 1
                break
            # avoid attend to pad word
            if not seq:
                attn = (y != 0).float().to(self.device)  # (batch,512)
            else:
                attn = torch.ones((batch, 512)).to(self.device)  # attend each of them

            y_hat = self.model.forward(x, attn, self.layer)  # seq: (batch, var_num) / token: (batch, 512, class_num)

            # get the most likely choice with max
            # output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            # output = torch.from_numpy(output).to(self.device)
            output = y_hat
            # print(y)
            # print(y_hat.squeeze(dim=1))

            if mode == 2:
                all_output[cnt:cnt + batch] = output.squeeze(dim=1)
                cnt += batch

            # accuracy
            if not seq:
                print("should not be here, accuracy")
                acc = torch.sum((y == output).float() * attn)
                total_acc += acc
                total_cnt += torch.sum(attn).item()
            else:
                threshold = 4  # Top 5% for 33, 8.4 (log), 4 (eq)
                threshold1 = 1
                threshold2 = 5
                accl = self.accuracy(y <= threshold, output.squeeze(
                    dim=1) <= threshold)  # accuracy of not acclaimed from whole sequence
                accr = self.accuracy(y > threshold,
                                     output.squeeze(dim=1) > threshold)  # accuracy of acclaimed from whole sequence
                # acc = accl + accr
                acc0 = self.accuracy(y <= threshold1, output.squeeze(dim=1) <= threshold1)
                acc1 = self.accuracy(torch.logical_and(y > threshold1, y <= threshold2),
                                     torch.logical_and(output.squeeze(dim=1) > threshold1,
                                                       output.squeeze(dim=1) <= threshold2))
                acc2 = self.accuracy(y > threshold2, output.squeeze(dim=1) > threshold2)
                acc = acc0 + acc1 + acc2
                accl = acc0
                accr = acc2
                acclf = self.accuracy_local(y <= threshold, output.squeeze(
                    dim=1) <= threshold)  # local accuracy given the true result is not acclaimed
                acclt = self.accuracy_local(y > threshold, output.squeeze(
                    dim=1) > threshold)  # local accuracy given the true result is acclaimed
                # print(y)
                # print(output)
                # print(self.accuracy(y <= threshold, output.squeeze() <= threshold))
                # print(self.accuracy(y > threshold, output.squeeze() > threshold))
                # exit()
                # print(acc)
                # print(y.shape[0])
                type1 = self.accuracy(y <= threshold, output.squeeze(dim=1) > threshold)
                type2 = self.accuracy(y > threshold, output.squeeze(dim=1) <= threshold)
                total_accl += accl
                local_accl += accl
                total_accr += accr
                local_accr += accr
                total_acc += acc
                local_acc += acc
                if acclf != -1:
                    total_acclf += acclf
                    total_lf_cnt += 1
                    local_acclf += acclf
                    local_lf_cnt += 1
                if acclt != -1:
                    total_acclt += acclt
                    total_lt_cnt += 1
                    local_acclt += acclt
                    local_lt_cnt += 1
                total_type1 += type1
                total_type2 += type2
                local_type1 += type1
                local_type2 += type2
                # total_cnt += y.shape[0]
                total_cnt += 1
                local_cnt += 1

            # calculate losses
            if not seq:
                y_hat = y_hat.permute(0, 2, 1)
            loss = self.reg_compute_loss(y_hat.squeeze().float(), y.float(), attn.float(), seq) / accum_grad
            total_loss += loss.item()
            local_loss += loss.item()

            # udpate only in train
            if mode == 0:
                loss.backward()
                if (count + 1) % accum_grad == 0:
                    self.optim.step()
                    self.model.zero_grad()
                    if local_lf_cnt == 0:
                        local_acclf = torch.tensor(-1)
                        local_lf_cnt = 1
                    if local_lt_cnt == 0:
                        local_acclt = torch.tensor(-1)
                        local_lt_cnt = 1
                    print("Loss", round(local_loss * accum_grad / local_cnt, 4), "Acc",
                          round(local_acc.item() / local_cnt, 4),
                          "Accl", round(local_accl.item() / local_cnt, 4), "Accr",
                          round(local_accr.item() / local_cnt, 4),
                          "Acclf", round(torch.tensor(local_acclf).item() / local_lf_cnt, 4), "Acclt",
                          round(torch.tensor(local_acclt).item() / local_lt_cnt, 4),
                          "Type1", round(local_type1.item() / total_cnt, 4), "Type2",
                          round(local_type2.item() / local_cnt, 4))
                    # with open('D:/UT/ECE324/MIDI-BERT/result/finetune/acclaimed_AcclaimedDS/full_log', 'a') as outfile:
                    #     outfile.write(
                    #         "Loss: {:06f} | Acc: {:06f} | Accl: {:06f} | Accr: {:06f} | Acclf: {:06f} | Acclt: {:06f} | Type1: {:06f} | Type2: {:06f} \r".format(
                    #             local_loss * accum_grad / local_cnt, local_acc.item() / local_cnt,
                    #             local_accl.item() / local_cnt, local_accr.item() / local_cnt,
                    #             local_acclf.item() / local_cnt, local_acclt.item() / local_cnt,
                    #             local_type1.item() / local_cnt, local_type2.item() / local_cnt))
                    local_acc = 0
                    local_accl = 0
                    local_accr = 0
                    local_acclf = 0
                    local_acclt = 0
                    local_type1 = 0
                    local_type2 = 0
                    local_cnt = 0
                    local_lf_cnt = 0
                    local_lt_cnt = 0
                    local_loss = 0

        if total_acclf == 0:
            total_acclf = torch.tensor([-1])
            total_lf_cnt = 1
        if total_acclt == 0:
            total_acclt = torch.tensor([-1])
            total_lt_cnt = 1
        # print(mode)
        if mode == 2:
            return round(total_loss * accum_grad / len(training_data), 4), round(total_acc.item() / total_cnt, 4), \
                   round(total_accl.item() / total_cnt, 4), round(total_accr.item() / total_cnt, 4), \
                   round(total_acclf.item() / total_lf_cnt, 4), round(total_acclt.item() / total_lt_cnt, 4), \
                   round(total_type1.item() / total_cnt, 4), round(total_type2.item() / total_cnt, 4), all_output

        return round(total_loss * accum_grad / len(training_data), 4), round(total_acc.item() / total_cnt, 4), \
               round(total_accl.item() / total_cnt, 4), round(total_accr.item() / total_cnt, 4), \
               round(total_acclf.item() / total_lf_cnt, 4), round(total_acclt.item() / total_lt_cnt, 4), \
               round(total_type1.item() / total_cnt, 4), round(total_type2.item() / total_cnt, 4)

    def save_checkpoint(self, epoch, train_acc, valid_acc,
                        valid_loss, train_loss, is_best, filename):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'optimizer': self.optim.state_dict()
        }
        torch.save(state, filename)

        best_mdl = filename.split('.')[0] + '_best.ckpt'

        if is_best:
            shutil.copyfile(filename, best_mdl)