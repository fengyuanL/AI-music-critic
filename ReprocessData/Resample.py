import numpy as np
from transformers import BertConfig, BertModel
import torch
import torch.nn as nn
import math
import random
import pickle
from tqdm import tqdm


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


def forward_bert(midibert, seq):
    midibert.eval()
    bertres = np.empty([len(seq), 13, 1, 512, 768])
    with torch.no_grad():
        for i in tqdm(range(len(seq))):
            y = midibert(torch.tensor(seq[i], device='cuda').unsqueeze(dim=0), torch.ones((1, 512)).to('cuda'))
            for j in range(0, 13):
                bertres[i][j] = y['hidden_states'][j].cpu().detach().numpy()
    return bertres


def entropy_bert(midibert, seq):
    midibert.eval()
    bertres = np.empty([len(seq), 13, 1, 512, 768])
    with torch.no_grad():
        for i in tqdm(range(len(seq))):
            y = midibert(torch.tensor(seq[i], device='cuda').unsqueeze(dim=0), torch.ones((1, 512)).to('cuda'))
            for j in range(0, 13):
                hs = y['hidden_states'][j].cpu().detach().numpy()
                # calculate the entropy using the hidden state
    return bertres


def forward_bert(midibert, seq):
    midibert.eval()
    bertres = np.empty([len(seq), 13, 1, 512, 768])
    with torch.no_grad():
        for i in tqdm(range(len(seq))):
            y = midibert(torch.tensor(seq[i], device='cuda').unsqueeze(dim=0), torch.ones((1, 512)).to('cuda'))
            for j in range(0, 13):
                bertres[i][j] = y['hidden_states'][j].cpu().detach().numpy()
    return bertres

def downsample(seq, pop):
    # For train only
    # Get the overall popularity distribution, upsample the data with low distribution
    hist, _ = np.histogram(pop, bins=range(12))
    print(hist)

    zero_indices = np.where(pop == 0)[0]
    np.random.shuffle(zero_indices)
    num_left = np.random.randint(9000, 10000)

    delete_indices = zero_indices[num_left:]

    result_seq = np.delete(seq, delete_indices, axis=0)
    result_pop = np.delete(pop, delete_indices, axis=0)

    hist, _ = np.histogram(result_pop, bins=range(12))
    print(hist)
    return result_seq,result_pop

def rescale(seq, pop):
    # For train, valid, test
    # Doing log base 1.5133 for 63, 1.5157 for 64
    new_pop = np.around(np.emath.logn(1.5157, pop+1), 2)
    return new_pop

def upsample(seq, pop):
    hist, _ = np.histogram(pop, bins=range(12))
    print(hist)

    zero_indices = np.where(pop == 0)[0]

    non_zero_seq = np.delete(seq, zero_indices, axis=0)
    non_zero_pop = np.delete(pop, zero_indices, axis=0)

    result_seq = np.append(seq, non_zero_seq, axis = 0)
    result_pop = np.append(pop, non_zero_pop, axis = 0)

    count = 0

    while count <= 1:
        result_seq = np.append(result_seq, non_zero_seq, axis = 0)
        result_pop = np.append(result_pop, non_zero_pop, axis = 0)
        count += 1

    hist, _ = np.histogram(result_pop, bins=range(12))
    print(hist)
    return result_seq,result_pop

if __name__ == "__main__":
    print("Loading Preprocessed Dataset")
    seq = np.load('F:/CDF/MIDI-BERTFT/Data/CP_data/GMP_1960_org_test.npy')
    pop = np.load('F:/CDF/MIDI-BERTFT/Data/CP_data/GMP_1960_org_test_ans.npy')
    
    # seq = np.load('Data/CP_data/GMP_1960_org_train.npy')
    # pop = np.load('Data/CP_data/GMP_1960_org_train_ans.npy')
    print(seq.shape)
    print(pop.shape)

    # seq, pop = downsample(seq,pop)
    hist, _ = np.histogram(pop, bins=range(12))
    print(hist)

    # print("Loading Dictionary")
    # with open('F:/CDF/MIDI-BERTFT/data_creation/prepare_data/dict/CP.pkl', 'rb') as f:
    # with open('data_creation/prepare_data/dict/CP.pkl', 'rb') as f:
        # e2w, w2e = pickle.load(f)

    # print("Loading Model")
    # configuration = BertConfig(max_position_embeddings=512,
                               # position_embedding_type='relative_key_query',
                               # hidden_size=768)

    # midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)

    # checkpoint = torch.load('F:/CDF/MIDI-BERTFT/result/pretrain/1960/model_best.ckpt')
    # checkpoint = torch.load('result/pretrain/1960/model_best.ckpt')
    # midibert.load_state_dict(checkpoint['state_dict'])
    # midibert.eval()
    # midibert.to(device='cuda')

    # midibert(torch.tensor(seq[0], device='cuda').unsqueeze(dim=0), torch.ones((1, 512)).to('cuda'))

    print('Reprocess data')
    # bertres = forward_bert(midibert, seq[0:10])
    pop_10 = rescale(seq, pop)
    # seq_d, pop_d = downsample(seq, pop_10)
    seq_d, pop_d = upsample(seq, pop_10)

    print('Saving data')
    # np.save('F:/CDF/MIDI-BERTFT/Data/CP_data/GMP_1960_reproc_train.npy', bertres)
    # np.save('F:/CDF/MIDI-BERTFT/Data/CP_data/GMP_1960_test_ans.npy', pop_10)
    hist, _ = np.histogram(pop_d, bins=range(12))
    print(hist)
    np.save('F:/CDF/MIDI-BERTFT/Data/CP_data/GMP_1960_test.npy', seq_d)
    np.save('F:/CDF/MIDI-BERTFT/Data/CP_data/GMP_1960_test_ans.npy', pop_d)
