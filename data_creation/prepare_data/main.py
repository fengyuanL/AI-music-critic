import os
from pathlib import Path
import glob
import pickle
import pathlib
import argparse
import torch
import json
import numpy as np
import random
from data_creation.prepare_data.model import *
from multiprocessing import Process


# GMP refers to giant midi piano

def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('-t', '--task', default='', choices=['melody', 'velocity', 'composer', 'emotion', 'acclaimed'])

    ### path ###
    parser.add_argument('--dict', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--dataset', type=str, choices=["pop909", "pop1k7", "ASAP", "pianist8", "emopia", "GMP", "GMP_mini", "GMP_1960", "GMP_Post1960", "GMP_Specify", 'GMP_1750', 'GMP_1820', 'GMP_1860', 'GMP_1910', 'GMP_2023', 'GMP_Post1960', 'GMP_Post1910'])
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--input_file', type=str, default='')

    ### parameter ###
    parser.add_argument('--max_len', type=int, default=512)

    ### output ###
    parser.add_argument('--output_dir', default="Data/CP_data")
    parser.add_argument('--name', default="")  # will be saved as "{output_dir}/{name}.npy"

    args = parser.parse_args()

    if args.task == 'acclaimed' and (args.dataset != 'GMP' or args.dataset != 'GMP_mini' or args.dataset != 'GMP_1960' or args.dataset != 'GMP_Post1960' or args.dataset != 'GMP_Specify'):
        print('[may be error] acclaimed task is only supported for GMP (GiantMIDIPiano) dataset')
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


def extract(files, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files), "can't find any files in the folder (extract)"

    print(f'Number of {mode} files: {len(files)}')

    segments, ans = model.prepare_data(files, args.task, int(args.max_len))

    dataset = args.dataset if args.dataset != 'pianist8' else 'composer'

    if args.input_dir != '' or args.input_file != '':
        name = args.input_dir or args.input_file
        if args.name == '':
            args.name = Path(name).stem
        output_file = os.path.join(args.output_dir, f'{args.name}.npy')
    elif dataset == 'composer' or dataset == 'emopia' or dataset == 'pop909' or dataset == 'GMP'\
            or dataset == 'GMP_mini' or dataset == 'GMP_1960' or dataset == 'GMP_Post1960' or \
            dataset == 'GMP_Specify' or dataset == 'GMP_1750' or dataset == 'GMP_1820' or \
            dataset == 'GMP_1860' or dataset == 'GMP_1910' or dataset == 'GMP_2023' or dataset == 'GMP_Post1910':
        output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')
    elif dataset == 'pop1k7' or dataset == 'ASAP':
        output_file = os.path.join(args.output_dir, f'{dataset}.npy')

    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    if args.task != '':
        if args.task == 'melody' or args.task == 'velocity':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task[:3]}ans.npy')
        elif args.task == 'composer' or args.task == 'emotion':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_ans.npy')
        elif args.task == 'acclaimed':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_ans.npy')
        np.save(ans_file, ans)
        print(f'Answer shape: {ans.shape}, saved at {ans_file}')


def main():
    # set seed
    seed = 324
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu
    np.random.seed(seed)
    random.seed(seed)
    
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
    elif args.dataset == 'GMP_Specify':
        dataset = 'GMP_Specify'
    elif args.dataset == 'GMP_1750':
        dataset = 'GMP_1750'
    elif args.dataset == 'GMP_1820':
        dataset = 'GMP_1820'
    elif args.dataset == 'GMP_1860':
        dataset = 'GMP_1860'
    elif args.dataset == 'GMP_1910':
        dataset = 'GMP_1910'
    elif args.dataset == 'GMP_2023':
        dataset = 'GMP_2023'
    elif args.dataset == 'GMP_Post1910':
        dataset = 'GMP_Post1910'
        

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
        files = glob.glob(f'Data/Dataset/GMP_1960/*.mid')
        random.shuffle(files)
        train_files = files[:int(len(files) * 0.8)]
        valid_files = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        test_files = files[int(len(files) * 0.9) :]
        with open('Data/Dataset/train1960.json', 'w') as f:
            json.dump(train_files, f)
        with open('Data/Dataset/valid1960.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/test1960.json', 'w') as f:
            json.dump(test_files, f)

    elif args.dataset == 'GMP_Post1960':
        files = glob.glob(f'Data/Dataset/GMP_Post1960/*.mid')
        random.shuffle(files)
        train_files = files[:int(len(files) * 0.8)]
        valid_files = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        test_files = files[int(len(files) * 0.9) :]
        with open('Data/Dataset/train1960.json', 'w') as f:
            json.dump(train_files, f)
        with open('Data/Dataset/valid1960.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/test1960.json', 'w') as f:
            json.dump(test_files, f)

    elif args.dataset == 'GMP_Specify':
        files = glob.glob(f'Data/Dataset/GMP_2023/*.mid')
        random.shuffle(files)
        train_files = files[:int(len(files) * 0.8)]
        valid_files = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        test_files = files[int(len(files) * 0.9) :]
    
    elif args.dataset == 'GMP_1750':
        files = glob.glob(f'Data/Dataset/GMP_1750/*.mid')
        random.shuffle(files)
        train_files = files[:int(len(files) * 0.8)]
        valid_files = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        test_files = files[int(len(files) * 0.9) :]
        with open('Data/Dataset/train1750.json', 'w') as f:
            json.dump(train_files, f)
        with open('Data/Dataset/valid1750.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/test1750.json', 'w') as f:
            json.dump(test_files, f)

    elif args.dataset == 'GMP_1820':
        files = glob.glob(f'Data/Dataset/GMP_1820/*.mid')
        random.shuffle(files)
        train_files = files[:int(len(files) * 0.8)]
        valid_files = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        test_files = files[int(len(files) * 0.9) :]
        with open('Data/Dataset/train1820.json', 'w') as f:
            json.dump(train_files, f)
        with open('Data/Dataset/valid1820.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/test1820.json', 'w') as f:
            json.dump(test_files, f)
            
            
    elif args.dataset == 'GMP_1860':
        files = glob.glob(f'Data/Dataset/GMP_1860/*.mid')
        random.shuffle(files)
        train_files = files[:int(len(files) * 0.8)]
        valid_files = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        test_files = files[int(len(files) * 0.9) :]
        with open('Data/Dataset/train1860.json', 'w') as f:
            json.dump(train_files, f)
        with open('Data/Dataset/valid1860.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/test1860.json', 'w') as f:
            json.dump(test_files, f)
            
    elif args.dataset == 'GMP_1910':
        files = glob.glob(f'Data/Dataset/GMP_1910/*.mid')
        random.shuffle(files)
        train_files = files[:int(len(files) * 0.8)]
        valid_files = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        test_files = files[int(len(files) * 0.9) :]
        with open('Data/Dataset/train1910.json', 'w') as f:
            json.dump(train_files, f)
        with open('Data/Dataset/valid1910.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/test1910.json', 'w') as f:
            json.dump(test_files, f)

    elif args.dataset == 'GMP_2023':
        files = glob.glob(f'Data/Dataset/GMP_2023/*.mid')
        random.shuffle(files)
        train_files = files[:int(len(files) * 0.8)]
        valid_files = files[int(len(files) * 0.8) : int(len(files) * 0.9)]
        test_files = files[int(len(files) * 0.9) :]
        with open('Data/Dataset/train2023.json', 'w') as f:
            json.dump(train_files, f)
        with open('Data/Dataset/valid2023.json', 'w') as f:
            json.dump(valid_files, f)
        with open('Data/Dataset/test2023.json', 'w') as f:
            json.dump(test_files, f)
            
    elif args.input_dir:
        files = glob.glob(f'{args.input_dir}/*.mid')

    elif args.input_file:
        files = [args.input_file]

    else:
        print('not supported')
        exit(1)

    if args.dataset in {'pop909', 'emopia', 'pianist8', 'GMP', 'GMP_mini', 'GMP_1960', 'GMP_Post1960', 'GMP_Specify', 'GMP_1750', 'GMP_1820', 'GMP_1860', 'GMP_1910', 'GMP_2023'}:
        p0 = Process(target=extract, args=(train_files[0:500], args, model, 'train0'))
        p1 = Process(target=extract, args=(train_files[500:1000], args, model, 'train1'))
        p2 = Process(target=extract, args=(train_files[1000:1500], args, model, 'train2'))
        p3 = Process(target=extract, args=(train_files[1500:2000], args, model, 'train3'))
        p4 = Process(target=extract, args=(train_files[2000:2500], args, model, 'train4'))
        p5 = Process(target=extract, args=(train_files[2500:3000], args, model, 'train5'))
        p6 = Process(target=extract, args=(train_files[3000:3500], args, model, 'train6'))
        p7 = Process(target=extract, args=(train_files[3500:4000], args, model, 'train7'))
        p8 = Process(target=extract, args=(train_files[4000:4500], args, model, 'train8'))
        p9 = Process(target=extract, args=(train_files[4500:5000], args, model, 'train9'))
        p10 = Process(target=extract, args=(train_files[5000:5500], args, model, 'train10'))
        p11 = Process(target=extract, args=(train_files[5500:], args, model, 'train11'))
        p12 = Process(target=extract, args=(valid_files, args, model, 'valid'))
        p13 = Process(target=extract, args=(test_files, args, model, 'test'))
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
    else:
        # in one single file
        extract(files, args, model)


if __name__ == '__main__':
    main()
