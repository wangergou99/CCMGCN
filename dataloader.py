import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
import numpy as np


class IEMOCAPDataset(Dataset):

    def __init__(self, split ):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open('IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        #self.labels = self.emotion_labels
        self.trainIds = self.trainIds+self.validIds


        for key, value in self.roberta1.items():
            if key in self.roberta2:
                self.roberta1[key] = np.sum([self.roberta2[key], self.roberta3[key],self.roberta4[key],value], axis=0)
                #self.roberta1[key] = 0.25*self.roberta1[key]
        #torch_data = {}
        # for i,value in self.roberta1.items():
        #     temp = torch.from_numpy(value)


        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if split == 'train' else self.testVid)]

        self.videoText=self.roberta1
        self.len = len(self.keys)
        k = self.len
        # k = int(0.1* k)  # 设置百分比，用来除去文本模态的数据
        # j = 0
        # for i in self.videoText:
        #     j += 1
        #     if j == k:
        #         break
        #         # return self.roberta1
        #     if j <= k:
        #         a = self.videoText[i]
        #         b=a.shape[0]
        #         m = 0
        #         for l in a:
        #             a[m] = a[m] * 0
        #             m = m + 1
        #             if m==k:
        #                 break
        #         # a = a[:] * 0.00001
        #         self.videoText[i] = a
        #     continue


        # n=0
        # j = 0
        # start = 0
        # k=0
        # for b in dia_len:
        #     temp_ = l[start:start + dia_len[n], :]
        #     temp_mean = torch.mean(temp_, dim=0)
        #     for k in l[start:start + dia_len[n], :]:
        #         k = l[j]
        #         c = torch.sum(k)
        #         if c == 0:
        #
        #             l[j] = temp_mean
        #         j+=1
        #     start += dia_len[n]
        #     n+=1


        # k = self.len
        # k = 0.3*k
        # j = 0
        # for i in self.roberta1:
        #     j += 1
        #     if j==k:
        #         break
        #         #return self.roberta1
        #     if j <=k:
        #         a = self.roberta1[i]
        #         a = a*0
        #         self.roberta1[i] = a
        #     continue

        # i = 'Ses03F_impro06'
        # a = self.roberta1[i]
        # a = a*0
        # self.roberta1[i] = a

        # j = 0
        # for i in self.roberta1:
        #     j += 1
        #     if j==2:
        #         return self.roberta1
        #     if j <=2:
        #         a = self.roberta1[i]
        #         a = a*0
        #         self.roberta1[i] = a
        #     continue





    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in \
                                  self.videoSpeakers[vid]]), \
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class MELDDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):
        
        self.Speakers, self.Features, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return  torch.FloatTensor(self.Features[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.EmotionLabels[conv])), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                conv

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]


class DailyDialogueDataset2(Dataset):

    def __init__(self, split, path):

        self.Speakers, self.Features, _, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]

        return torch.FloatTensor(self.Features[conv]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.Speakers[conv]]), \
               torch.FloatTensor([1] * len(self.EmotionLabels[conv])), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               conv

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]
