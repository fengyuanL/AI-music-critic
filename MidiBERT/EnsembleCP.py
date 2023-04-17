import numpy as np
import pickle
from tqdm import tqdm
import data_creation.prepare_data.utils as utils
import json
import copy
import torch
import random
from MidiBERT.model import MidiBert
from MidiBERT.EnsembleFT import FinetuneTrainer
from MidiBERT.finetune_dataset import FinetuneDataset
from torch.utils.data import DataLoader
from transformers import BertConfig
from matplotlib import pyplot as plt
from scipy.stats import entropy
from queue import PriorityQueue
import heapq

Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "Padding": 8,
}

Emotion = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}


class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, 'rb'))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        self.pad_word = [self.event2word[etype]['%s <PAD>' % etype] for etype in self.event2word]

    def extract_events(self, input_path, task):
        note_items, tempo_items = utils.read_items(input_path)
        if len(note_items) == 0:  # if the midi contains nothing
            return None
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups, task)
        return events

    def padding(self, data, max_len, ans):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            if not ans:
                data.append(self.pad_word)
            else:
                data.append(0)

        return data
    
#     def calculate_entropy(self, hs):
#         total_H = 0
#         for vector in hs[0]:
#             unique, counts = np.unique(vector, return_counts=True)
#             frequency = np.asarray((unique, counts))
#             # print("frequency:", frequency)
#             pk = frequency[1]/sum(frequency[1])
#             # print("pk:", pk)
        
#             H = -np.sum(pk * np.log(pk)) / np.log(2)
#             # print(H)
#             total_H += H
#         return total_H

    def calculate_entropy(self, hs):
        #512vectors
        #a vector has 768 elements
        # Get entropy of each vector
        all_entropy = []
        heapq.heapify(all_entropy)
        for neuron in range(768):
            vector = hs[0][:,neuron]
            distribution = np.histogram(vector, bins=6, range=(-3,3), density=True, weights=None)[0]
            H=entropy(distribution,base=2)
            heapq.heappush(all_entropy, H)
        
        n_largest = heapq.nlargest(614, all_entropy)
        
        return sum(n_largest)/614

    def prepare_data(self, midi_paths, task, max_len):
        all_words, all_ys = [], []

        # with open('Data/Dataset/DataDictFullwithPop3_20.json') as json_file:
        #     DataDict = json.load(json_file)

        print("Loading Dictionary")
        with open('data_creation/prepare_data/dict/CP.pkl', 'rb') as f:
            e2w, w2e = pickle.load(f)

        print("\nBuilding BERT model")
        configuration = BertConfig(max_position_embeddings=512,
                                   position_embedding_type='relative_key_query',
                                   hidden_size=768)

        midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
        best_mdl = ''
        best_mdl = 'result/pretrain/1820/model_best.ckpt'
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        midibert.load_state_dict(checkpoint['state_dict'])

        index_layer = int(12) - 13

        case = -1
        output_dict = {}

        for path in tqdm(midi_paths):
            # extract events
            events = self.extract_events(path, task)
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            if task == "acclaimed":
                piecename = path[:path.rfind(", ")]
                piecename = piecename[piecename.rfind("/") + 1:]
                popularity = int(piecename[:piecename.find("_")])
                piecename = piecename[piecename.find("_") + 1:]

            if popularity != 0 and len(events) > 512 + 64:
                case = 1
                org_event = copy.deepcopy(events)
                local_words = []
                local_ys = []
                for up in range(0, 8):
                    events = copy.deepcopy(org_event)
                    events = events[(up * 64):]
                    # events to words
                    words, ys = [], []
                    for note_tuple in events:
                        nts, to_class = [], -1
                        for e in note_tuple:
                            e_text = '{} {}'.format(e.name, e.value)
                            nts.append(self.event2word[e.name][e_text])
                            if e.name == 'Pitch':
                                to_class = e.Type
                        words.append(nts)
                        if task == 'melody' or task == 'velocity':
                            ys.append(to_class + 1)

                    # slice to chunks so that max length = max_len (default: 512)
                    slice_words, slice_ys = [], []
                    for i in range(0, len(words), max_len):
                        slice_words.append(words[i:i + max_len])
                        if task == "composer":
                            name = path.split('/')[-2]
                            slice_ys.append(Composer[name])
                        elif task == "emotion":
                            name = path.split('/')[-1].split('_')[0]
                            slice_ys.append(Emotion[name])
                        elif task == "acclaimed":
                            piecename = path[:path.rfind(", ")]
                            piecename = piecename[piecename.rfind("/") + 1:]
                            popularity = piecename[:piecename.find("_")]
                            piecename = piecename[piecename.find("_") + 1 : ]
                            slice_ys.append(int(popularity))
                        else:
                            slice_ys.append(ys[i:i + max_len])

                    # padding or drop
                    # drop only when the task is 'composer' and the data length < max_len//2
                    if len(slice_words[-1]) < max_len:
                        if task == 'composer' and len(slice_words[-1]) < max_len // 2:
                            slice_words.pop()
                            slice_ys.pop()
                        else:
                            slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)

                    if (task == 'melody' or task == 'velocity') and len(slice_ys[-1]) < max_len:
                        slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

                    all_words = all_words + slice_words
                    all_ys = all_ys + slice_ys
                    local_words = local_words + slice_words
                    local_ys = local_ys + slice_ys
                    # we call the model here to evaluate each slice_words and compare with slice_ys
                trainset = FinetuneDataset(X=local_words, y=local_ys)
                train_loader = DataLoader(trainset, batch_size=1, num_workers=1, shuffle=False)
                trainer = FinetuneTrainer(midibert, train_loader, train_loader, train_loader, index_layer, 2e-5,
                                          2, 768, len(local_words), False, 0, None, True)
                test_loss, test_acc, test_accl, test_accr, test_acclf, test_acclt, test_type1, test_type2, output = trainer.test()
                # TODO: entropy
                entropy = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for k in local_words]
                # Forward BERT
                midibert.eval()
                with torch.no_grad():
                    for i in range(len(local_words)):
                        y = midibert(torch.tensor(local_words[i], device='cuda').unsqueeze(dim=0),
                                     torch.ones((1, 512)).to('cuda'))
                        # for the layers we want
                        for j in range(12):
                            hs = y['hidden_states'][j].cpu().detach().numpy()
                            entropy[i][j] = self.calculate_entropy(hs)
                    
                output_dict[piecename] = {}
                output_dict[piecename]["test_loss"] = test_loss
                output_dict[piecename]["test_acc"] = test_acc
                output_dict[piecename]["test_accl"] = test_accl
                output_dict[piecename]["test_accr"] = test_accr
                output_dict[piecename]["test_acclf"] = test_acclf
                output_dict[piecename]["test_acclt"] = test_acclt
                output_dict[piecename]["test_type1"] = test_type1
                output_dict[piecename]["test_type2"] = test_type2
                output_dict[piecename]["output"] = output.cpu().detach().tolist()
                output_dict[piecename]["pop"] = popularity
                output_dict[piecename]["Entropy"] = entropy
                output_dict[piecename]["up"] = up
                output_dict[piecename]["event_len"] = len(events)


            elif popularity != 0 and len(events) < 512 + 64:
                case = 2
                org_event = copy.deepcopy(events)
                local_words = []
                local_ys = []
                for up in range(0, 8):
                    events = copy.deepcopy(org_event)
                    # events to words
                    words, ys = [], []
                    for note_tuple in events:
                        nts, to_class = [], -1
                        for e in note_tuple:
                            e_text = '{} {}'.format(e.name, e.value)
                            nts.append(self.event2word[e.name][e_text])
                            if e.name == 'Pitch':
                                to_class = e.Type
                        words.append(nts)
                        if task == 'melody' or task == 'velocity':
                            ys.append(to_class + 1)

                    # slice to chunks so that max length = max_len (default: 512)
                    slice_words, slice_ys = [], []
                    for i in range(0, len(words), max_len):
                        slice_words.append(words[i:i + max_len])
                        if task == "composer":
                            name = path.split('/')[-2]
                            slice_ys.append(Composer[name])
                        elif task == "emotion":
                            name = path.split('/')[-1].split('_')[0]
                            slice_ys.append(Emotion[name])
                        elif task == "acclaimed":
                            piecename = path[:path.rfind(", ")]
                            piecename = piecename[piecename.rfind("/") + 1:]
                            popularity = piecename[:piecename.find("_")]
                            piecename = piecename[piecename.find("_") + 1:]
                            slice_ys.append(int(popularity))
                        else:
                            slice_ys.append(ys[i:i + max_len])

                    # padding or drop
                    # drop only when the task is 'composer' and the data length < max_len//2
                    if len(slice_words[-1]) < max_len:
                        if task == 'composer' and len(slice_words[-1]) < max_len // 2:
                            slice_words.pop()
                            slice_ys.pop()
                        else:
                            slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)

                    if (task == 'melody' or task == 'velocity') and len(slice_ys[-1]) < max_len:
                        slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

                    all_words = all_words + slice_words
                    all_ys = all_ys + slice_ys
                    local_words = local_words + slice_words
                    local_ys = local_ys + slice_ys
                    # we call the model here to evaluate each slice_words and compare with slice_ys
                trainset = FinetuneDataset(X=local_words, y=local_ys)
                train_loader = DataLoader(trainset, batch_size=1, num_workers=1, shuffle=False)
                trainer = FinetuneTrainer(midibert, train_loader, train_loader, train_loader, index_layer, 2e-5,
                                          2, 768, len(local_words), False, 0, None, True)
                test_loss, test_acc, test_accl, test_accr, test_acclf, test_acclt, test_type1, test_type2, output = trainer.test()
                # TODO: entropy
                entropy = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for k in local_words]
                # Forward BERT
                midibert.eval()
                with torch.no_grad():
                    for i in range(len(local_words)):
                        y = midibert(torch.tensor(local_words[i], device='cuda').unsqueeze(dim=0),
                                     torch.ones((1, 512)).to('cuda'))
                        # for the layers we want
                        for j in range(12):
                            hs = y['hidden_states'][j].cpu().detach().numpy()
                            entropy[i][j] = self.calculate_entropy(hs)
                
                output_dict[piecename] = {}
                output_dict[piecename]["test_loss"] = test_loss
                output_dict[piecename]["test_acc"] = test_acc
                output_dict[piecename]["test_accl"] = test_accl
                output_dict[piecename]["test_accr"] = test_accr
                output_dict[piecename]["test_acclf"] = test_acclf
                output_dict[piecename]["test_acclt"] = test_acclt
                output_dict[piecename]["test_type1"] = test_type1
                output_dict[piecename]["test_type2"] = test_type2
                output_dict[piecename]["output"] = output.cpu().detach().tolist()
                output_dict[piecename]["pop"] = popularity
                output_dict[piecename]["Entropy"] = entropy
                output_dict[piecename]["up"] = up
                output_dict[piecename]["event_len"] = len(events)

            if popularity == 0 and len(events) > 512 + 256:
                case = 3
                org_event = copy.deepcopy(events)
                local_words = []
                local_ys = []
                for up in range(0, 8):
                    events = copy.deepcopy(org_event)
                    events = events[(up * 64):]
                    # events to words
                    words, ys = [], []
                    for note_tuple in events:
                        nts, to_class = [], -1
                        for e in note_tuple:
                            e_text = '{} {}'.format(e.name, e.value)
                            nts.append(self.event2word[e.name][e_text])
                            if e.name == 'Pitch':
                                to_class = e.Type
                        words.append(nts)
                        if task == 'melody' or task == 'velocity':
                            ys.append(to_class + 1)

                    # slice to chunks so that max length = max_len (default: 512)
                    slice_words, slice_ys = [], []
                    for i in range(0, len(words), max_len):
                        slice_words.append(words[i:i + max_len])
                        if task == "composer":
                            name = path.split('/')[-2]
                            slice_ys.append(Composer[name])
                        elif task == "emotion":
                            name = path.split('/')[-1].split('_')[0]
                            slice_ys.append(Emotion[name])
                        elif task == "acclaimed":
                            piecename = path[:path.rfind(", ")]
                            piecename = piecename[piecename.rfind("/") + 1:]
                            popularity = piecename[:piecename.find("_")]
                            piecename = piecename[piecename.find("_") + 1:]
                            slice_ys.append(int(popularity))
                        else:
                            slice_ys.append(ys[i:i + max_len])

                    # padding or drop
                    # drop only when the task is 'composer' and the data length < max_len//2
                    if len(slice_words[-1]) < max_len:
                        if task == 'composer' and len(slice_words[-1]) < max_len // 2:
                            slice_words.pop()
                            slice_ys.pop()
                        else:
                            slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)

                    if (task == 'melody' or task == 'velocity') and len(slice_ys[-1]) < max_len:
                        slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

                    all_words = all_words + slice_words
                    all_ys = all_ys + slice_ys
                    local_words = local_words + slice_words
                    local_ys = local_ys + slice_ys
                    # we call the model here to evaluate each slice_words and compare with slice_ys
                trainset = FinetuneDataset(X=local_words, y=local_ys)
                train_loader = DataLoader(trainset, batch_size=1, num_workers=1, shuffle=False)
                trainer = FinetuneTrainer(midibert, train_loader, train_loader, train_loader, index_layer, 2e-5,
                                          2, 768, len(local_words), False, 0, None, True)
                test_loss, test_acc, test_accl, test_accr, test_acclf, test_acclt, test_type1, test_type2, output = trainer.test()
                # TODO: entropy
                entropy = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for k in local_words]
                # Forward BERT
                midibert.eval()
                with torch.no_grad():
                    for i in range(len(local_words)):
                        y = midibert(torch.tensor(local_words[i], device='cuda').unsqueeze(dim=0),
                                     torch.ones((1, 512)).to('cuda'))
                        # for the layers we want
                        for j in range(12):
                            hs = y['hidden_states'][j].cpu().detach().numpy()
                            entropy[i][j] = self.calculate_entropy(hs)

                output_dict[piecename] = {}
                output_dict[piecename]["test_loss"] = test_loss
                output_dict[piecename]["test_acc"] = test_acc
                output_dict[piecename]["test_accl"] = test_accl
                output_dict[piecename]["test_accr"] = test_accr
                output_dict[piecename]["test_acclf"] = test_acclf
                output_dict[piecename]["test_acclt"] = test_acclt
                output_dict[piecename]["test_type1"] = test_type1
                output_dict[piecename]["test_type2"] = test_type2
                output_dict[piecename]["output"] = output.cpu().detach().tolist()
                output_dict[piecename]["pop"] = popularity
                output_dict[piecename]["Entropy"] = entropy
                output_dict[piecename]["up"] = up
                output_dict[piecename]["event_len"] = len(events)

            elif popularity == 0 and len(events) < 512 + 256:
                case = 4
                org_event = copy.deepcopy(events)
                local_words = []
                local_ys = []
                for up in range(0, 8):
                    events = copy.deepcopy(org_event)
                    # events to words
                    words, ys = [], []
                    for note_tuple in events:
                        nts, to_class = [], -1
                        for e in note_tuple:
                            e_text = '{} {}'.format(e.name, e.value)
                            nts.append(self.event2word[e.name][e_text])
                            if e.name == 'Pitch':
                                to_class = e.Type
                        words.append(nts)
                        if task == 'melody' or task == 'velocity':
                            ys.append(to_class + 1)

                    # slice to chunks so that max length = max_len (default: 512)
                    slice_words, slice_ys = [], []
                    for i in range(0, len(words), max_len):
                        slice_words.append(words[i:i + max_len])
                        if task == "composer":
                            name = path.split('/')[-2]
                            slice_ys.append(Composer[name])
                        elif task == "emotion":
                            name = path.split('/')[-1].split('_')[0]
                            slice_ys.append(Emotion[name])
                        elif task == "acclaimed":
                            piecename = path[:path.rfind(", ")]
                            piecename = piecename[piecename.rfind("/") + 1:]
                            popularity = piecename[:piecename.find("_")]
                            slice_ys.append(int(popularity))
                        else:
                            slice_ys.append(ys[i:i + max_len])

                    # padding or drop
                    # drop only when the task is 'composer' and the data length < max_len//2
                    if len(slice_words[-1]) < max_len:
                        if task == 'composer' and len(slice_words[-1]) < max_len // 2:
                            slice_words.pop()
                            slice_ys.pop()
                        else:
                            slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)

                    if (task == 'melody' or task == 'velocity') and len(slice_ys[-1]) < max_len:
                        slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

                    all_words = all_words + slice_words
                    all_ys = all_ys + slice_ys
                    local_words = local_words + slice_words
                    local_ys = local_ys + slice_ys
                    # we call the model here to evaluate each slice_words and compare with slice_ys
                trainset = FinetuneDataset(X=local_words, y=local_ys)
                train_loader = DataLoader(trainset, batch_size=1, num_workers=1, shuffle=False)
                trainer = FinetuneTrainer(midibert, train_loader, train_loader, train_loader, index_layer, 2e-5,
                                          2, 768, len(local_words), False, 0, None, True)
                test_loss, test_acc, test_accl, test_accr, test_acclf, test_acclt, test_type1, test_type2, output = trainer.test()
                
                # TODO: entropy
                entropy = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for k in local_words]
                # Forward BERT
                midibert.eval()
                with torch.no_grad():
                    for i in range(len(local_words)):
                        y = midibert(torch.tensor(local_words[i], device='cuda').unsqueeze(dim=0),
                                     torch.ones((1, 512)).to('cuda'))
                        # for the layers we want
                        for j in range(12):
                            hs = y['hidden_states'][j].cpu().detach().numpy()
                            entropy[i][j] = self.calculate_entropy(hs)

                
                output_dict[piecename] = {}
                output_dict[piecename]["test_loss"] = test_loss
                output_dict[piecename]["test_acc"] = test_acc
                output_dict[piecename]["test_accl"] = test_accl
                output_dict[piecename]["test_accr"] = test_accr
                output_dict[piecename]["test_acclf"] = test_acclf
                output_dict[piecename]["test_acclt"] = test_acclt
                output_dict[piecename]["test_type1"] = test_type1
                output_dict[piecename]["test_type2"] = test_type2
                output_dict[piecename]["output"] = output.cpu().detach().tolist()
                output_dict[piecename]["pop"] = popularity
                output_dict[piecename]["Entropy"] = entropy
                output_dict[piecename]["up"] = up
                output_dict[piecename]["event_len"] = len(events)

        all_words = np.array(all_words)
        all_ys = np.array(all_ys)

        return all_words, all_ys, output_dict
