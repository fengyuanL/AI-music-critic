import os
from pathlib import Path
import glob
import pickle
import pathlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import math
import json
import random
from multiprocessing import Process
from transformers import BertConfig, BertModel, AdamW
import shutil
import tqdm
from torch.nn.utils import clip_grad_norm_
from MidiBERT.finetune_model import TokenClassification, SequenceClassification, SequenceRegression
from MidiBERT.EnsembleCP import *



class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# BERT model: similar approach to "felix"
class MidiBert(nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super().__init__()

        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []  # [3,18,88,66]
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=int)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=int)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # feed to bert
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        return y

    def get_rand_tok(self):
        c1, c2, c3, c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array(
            [random.choice(range(c1)), random.choice(range(c2)), random.choice(range(c3)), random.choice(range(c4))])


def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('-t', '--task', default='', choices=['melody', 'velocity', 'composer', 'emotion', 'acclaimed'])

    ### path ###
    parser.add_argument('--dict', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--dataset', type=str, choices=["pop909", "pop1k7", "ASAP", "pianist8", "emopia", "GMP", "GMP_mini", "GMP_1960", "GMP_Post1960", "GMP_Specify", "GMP_1910", "GMP_2023", "GMP_1860", "GMP_Post1860", "GMP_1820", "GMP_Post1820", "GMP_Post1910"])
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--input_file', type=str, default='')

    ### parameter ###
    parser.add_argument('--max_len', type=int, default=512)

    ### output ###
    parser.add_argument('--output_dir', default="Data/CP_data")
    parser.add_argument('--name', default="")  # will be saved as "{output_dir}/{name}.npy"

    args = parser.parse_args()

    if args.task == 'acclaimed' and (args.dataset != 'GMP' or args.dataset != 'GMP_mini' or args.dataset != 'GMP_1960' or args.dataset != 'GMP_Post1960' or args.dataset != 'GMP_Specify'):
        print('[error] acclaimed task is only supported for GMP (GiantMIDIPiano) dataset')
    elif args.task == 'melody' and args.dataset != 'pop909':
        print('[error] melody task is only supported for pop909 dataset')
        exit(1)
    elif args.task == 'composer' and args.dataset != 'pianist8':
        print('[error] composer task is only supported for pianist8 dataset')
        exit(1)
    elif args.task == 'emotion' and args.dataset != 'emopia':
        print('[error] emotion task is only supported for emopia dataset')
        exit(1)
    elif args.dataset == None or args.input_dir == None or args.input_file == None:
        print('[error] Please specify the input directory or dataset')
        exit(1)

    return args


def forward_bert(midibert, seq):
    midibert.eval()
    bertres = np.empty([len(seq), 13, 1, 512, 768])
    with torch.no_grad():
        for i in tqdm(range(len(seq))):
            y = midibert(torch.tensor(seq[i], device='cuda').unsqueeze(dim=0), torch.ones((1, 512)).to('cuda'))
            for j in range(0, 13):
                bertres[i][j] = y['hidden_states'][j].cpu().detach().numpy()
    return bertres


def extract(files, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files), "can't find any files in the folder (extract)"

    print(f'Number of {mode} files: {len(files)}')

    segments, ans, output_dict = model.prepare_data(files, args.task, int(args.max_len))

    with open(os.path.join('result', f'{args.dataset}_{mode}.json'), 'w') as convert_file:
        convert_file.write(json.dumps(output_dict))

    dataset = args.dataset if args.dataset != 'pianist8' else 'composer'

    if args.input_dir != '' or args.input_file != '':
        name = args.input_dir or args.input_file
        if args.name == '':
            args.name = Path(name).stem
        output_file = os.path.join(args.output_dir, f'{args.name}.npy')
    elif dataset == 'composer' or dataset == 'emopia' or dataset == 'pop909' or dataset == 'GMP'\
            or dataset == 'GMP_mini' or dataset == 'GMP_1960' or dataset == 'GMP_Post1960' or \
            dataset == 'GMP_Specify' or dataset == 'GMP_1910' or dataset == 'GMP_2023' or dataset == 'GMP_1860' \
            or dataset == 'GMP_Post1860' or dataset == 'GMP_Post1910' or dataset == 'GMP_1820' or dataset == 'GMP_Post1820':
        output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    elif dataset == 'pop1k7' or dataset == 'ASAP':
        output_file = os.path.join(args.output_dir, f'{dataset}.npy')

    # np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, NOT saved at {output_file}')

    if args.task != '':
        if args.task == 'melody' or args.task == 'velocity':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task[:3]}ans.npy')
        elif args.task == 'composer' or args.task == 'emotion':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_ans.npy')
        elif args.task == 'acclaimed':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_ans.npy')
        # np.save(ans_file, ans)
        print(f'Answer shape: {ans.shape}, NOT saved at {ans_file}')


def main():
    args = get_args()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # initialize model
    model = CP(dict=args.dict)

    if args.dataset == 'pop909':
        dataset = 'pop909_processed'
    elif args.dataset == 'emopia':
        dataset = 'EMOPIA_1.0'
    elif args.dataset == 'pianist8':
        dataset = 'joann8512-Pianist8-ab9f541'
    elif args.dataset == 'GMP':
        dataset = 'GMP'
    elif args.dataset == 'GMP_mini':
        dataset = 'GMP_mini'
    elif args.dataset == 'GMP_1960':
        dataset = 'GMP_1960'
    elif args.dataset == 'GMP_Post1960':
        dataset = 'GMP_Post1960'
    elif args.dataset == 'GMP_1910':
        dataset = 'GMP_1910'
    elif args.dataset == 'GMP_Specify':
        dataset = 'GMP_Specify'
    elif args.dataset == 'GMP_2023':
        dataset = 'GMP_2023'
    elif args.dataset == 'GMP_Post1860':
        dataset = 'GMP_Post1860'
    elif args.dataset == 'GMP_Post1910':
        dataset = 'GMP_Post1910'
    elif args.dataset == 'GMP_1820':
        dataset = 'GMP_1820'
    elif args.dataset == 'GMP_Post1820':
        dataset = 'GMP_Post1820'
    
    if args.dataset == 'pop909' or args.dataset == 'emopia':
        train_files = glob.glob(f'Data/Dataset/{dataset}/train/*.mid')
        valid_files = glob.glob(f'Data/Dataset/{dataset}/valid/*.mid')
        test_files = glob.glob(f'Data/Dataset/{dataset}/test/*.mid')

    elif args.dataset == 'pianist8':
        train_files = glob.glob(f'Data/Dataset/{dataset}/train/*/*.mid')
        valid_files = glob.glob(f'Data/Dataset/{dataset}/valid/*/*.mid')
        test_files = glob.glob(f'Data/Dataset/{dataset}/test/*/*.mid')

    elif args.dataset == 'pop1k7':
        files = glob.glob('Data/Dataset/dataset/midi_transcribed/*/*.midi')

    elif args.dataset == 'ASAP':
        files = pickle.load(open('Data/Dataset/ASAP_song.pkl', 'rb'))
        files = [f'Dataset/asap-dataset/{file}' for file in files]

    elif args.dataset == 'GMP':
        files = glob.glob(f'Data/Dataset/GMP/*.mid')
        train_files = files[:int(len(files) * 0.8)]
        valid_files = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        test_files = files[int(len(files) * 0.9) :]

    elif args.dataset == 'GMP_mini':
        files = glob.glob(f'Data/Dataset/GMP/*.mid')
        train_files = files[0:24]
        valid_files = files[24:27]
        test_files = files[27:30]

    elif args.dataset == 'GMP_1960':
        f = open("Data/Dataset/train1960.json")
        train_files = json.load(f)
        f = open("Data/Dataset/valid1960.json")
        valid_files = json.load(f)
        f = open("Data/Dataset/test1960.json")
        test_files = json.load(f)
        
    elif args.dataset == 'GMP_1910':
        f = open("Data/Dataset/train1960.json")
        train_files = json.load(f)
        f = open("Data/Dataset/valid1960.json")
        valid_files = json.load(f)
        f = open("Data/Dataset/test1960.json")
        test_files = json.load(f)

    elif args.dataset == 'GMP_2023':
        f = open("Data/Dataset/train2023.json")
        train_files = json.load(f)
        f = open("Data/Dataset/valid2023.json")
        valid_files = json.load(f)
        f = open("Data/Dataset/test2023.json")
        test_files = json.load(f)
        
    elif args.dataset == 'GMP_1860':
        f = open("Data/Dataset/train1860.json")
        train_files = json.load(f)
        f = open("Data/Dataset/valid1860.json")
        valid_files = json.load(f)
        f = open("Data/Dataset/test1860.json")
        test_files = json.load(f)
        
    elif args.dataset == 'GMP_Post1960':
        files = glob.glob(f'Data/Dataset/GMP_Post1960/*.mid')
        random.shuffle(files)
        valid_files = files[:int(len(files) * 0.5)]
        test_files = files[int(len(files) * 0.5) :]
        with open('Data/Dataset/validPost1960.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/testPost1960.json', 'w') as f:
            json.dump(test_files, f)
    
    elif args.dataset == 'GMP_Post1860':
        files = glob.glob(f'Data/Dataset/GMP_Post1860/*.mid')
        random.shuffle(files)
        valid_files = files[:int(len(files) * 0.5)]
        test_files = files[int(len(files) * 0.5) :]
        with open('Data/Dataset/validPost1860.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/testPost1860.json', 'w') as f:
            json.dump(test_files, f)
            
    elif args.dataset == 'GMP_Post1910':
        files = glob.glob(f'Data/Dataset/GMP_Post1910/*.mid')
        random.shuffle(files)
        valid_files = files[:int(len(files) * 0.5)]
        test_files = files[int(len(files) * 0.5) :]
        with open('Data/Dataset/validPost1910.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/testPost1910.json', 'w') as f:
            json.dump(test_files, f)

    elif args.dataset == 'GMP_1820':
        f = open("Data/Dataset/train1820.json")
        train_files = json.load(f)
        f = open("Data/Dataset/valid1820.json")
        valid_files = json.load(f)
        f = open("Data/Dataset/test1820.json")
        test_files = json.load(f)
        
    elif args.dataset == 'GMP_Post1820':
        files = glob.glob(f'Data/Dataset/GMP_Post1820/*.mid')
        random.shuffle(files)
        valid_files = files[:int(len(files) * 0.5)]
        test_files = files[int(len(files) * 0.5) :]
        with open('Data/Dataset/validPost1820.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/testPost1820.json', 'w') as f:
            json.dump(test_files, f)
    
    elif args.dataset == 'GMP_Specify':
        files = glob.glob(f'Data/Datasets/GMP_1910/*.mid')
        random.shuffle(files)
        train_files = files[:int(len(files) * 0.8)]
        valid_files = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        test_files = files[int(len(files) * 0.9) :]

    elif args.input_dir:
        files = glob.glob(f'{args.input_dir}/*.mid')

    elif args.input_file:
        files = [args.input_file]

    else:
        print('not supported')
        exit(1)

    print("Loading Dictionary")
    with open('data_creation/prepare_data/dict/CP.pkl', 'rb') as f:
    # with open('data_creation/prepare_data/dict/CP.pkl', 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("Loading Model")
    configuration = BertConfig(max_position_embeddings=512,
                               position_embedding_type='relative_key_query',
                               hidden_size=768)

    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)

    checkpoint = torch.load('result/pretrain/1960Large/model_best.ckpt')
    # checkpoint = torch.load('result/pretrain/1960/model_best.ckpt')
    midibert.load_state_dict(checkpoint['state_dict'])
    midibert.eval()
    midibert.to(device='cuda')


    if args.dataset in {'pop909', 'emopia', 'pianist8', 'GMP', 'GMP_mini', 'GMP_1960', 'GMP_Post1960', 'GMP_Specify', 'GMP_1910', 'GMP_2023', 'GMP_1860', 'GMP_Post1860', 'GMP_Post1910', 'GMP_1820', 'GMP_Post1820'}:
        valid_files = valid_files
        test_files = test_files
        p0 = Process(target=extract, args=(valid_files[int(len(valid_files) / 8 * 0) : int(len(valid_files) / 8 * 1)], args, model, 'valid0'))
        p1 = Process(target=extract, args=(valid_files[int(len(valid_files) / 8 * 1) : int(len(valid_files) / 8 * 2)], args, model, 'valid1'))
        p2 = Process(target=extract, args=(valid_files[int(len(valid_files) / 8 * 2) : int(len(valid_files) / 8 * 3)], args, model, 'valid2'))
        p3 = Process(target=extract, args=(valid_files[int(len(valid_files) / 8 * 3) : int(len(valid_files) / 8 * 4)], args, model, 'valid3'))
        p4 = Process(target=extract, args=(valid_files[int(len(valid_files) / 8 * 4) : int(len(valid_files) / 8 * 5)], args, model, 'valid4'))
        p5 = Process(target=extract, args=(valid_files[int(len(valid_files) / 8 * 5) : int(len(valid_files) / 8 * 6)], args, model, 'valid5'))
        p6 = Process(target=extract, args=(valid_files[int(len(valid_files) / 8 * 6) : int(len(valid_files) / 8 * 7)], args, model, 'valid6'))
        p7 = Process(target=extract, args=(valid_files[int(len(valid_files) / 8 * 7) : int(len(valid_files) / 8 * 8)], args, model, 'valid7'))
        p8 = Process(target=extract, args=(test_files[int(len(test_files) / 8 * 0) : int(len(test_files) / 8 * 1)], args, model, 'test0'))
        p9 = Process(target=extract, args=(test_files[int(len(test_files) / 8 * 1) : int(len(test_files) / 8 * 2)], args, model, 'test1'))
        p10 = Process(target=extract, args=(test_files[int(len(test_files) / 8 * 2) : int(len(test_files) / 8 * 3)], args, model, 'test2'))
        p11 = Process(target=extract, args=(test_files[int(len(test_files) / 8 * 3) : int(len(test_files) / 8 * 4)], args, model, 'test3'))
        p12 = Process(target=extract, args=(test_files[int(len(test_files) / 8 * 4) : int(len(test_files) / 8 * 5)], args, model, 'test4'))
        p13 = Process(target=extract, args=(test_files[int(len(test_files) / 8 * 5) : int(len(test_files) / 8 * 6)], args, model, 'test5'))
        p14 = Process(target=extract, args=(test_files[int(len(test_files) / 8 * 6) : int(len(test_files) / 8 * 7)], args, model, 'test6'))
        p15 = Process(target=extract, args=(test_files[int(len(test_files) / 8 * 7) : int(len(test_files) / 8 * 8)], args, model, 'test7'))
        p0.start()
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()
        p9.start()
        p10.start()
        p11.start()
        p12.start()
        p13.start()
        p14.start()
        p15.start()
        p0.join()
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        p8.join()
        p9.join()
        p10.join()
        p11.join()
        p12.join()
        p13.join()
        p14.join()
        p15.join()
    else:
        # in one single file
        extract(files, args, model)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()