import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
#from torch_geometric.nn import RGCNConv, GraphConv
from torch.nn.parameter import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
from sklearn.metrics import accuracy_score
from model_GCN import GCNII_lyc
import ipdb
from tqdm import tqdm
import time

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class MM_GCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False):
        super(MM_GCN, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph

        self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden, nclass=nclass,
                               dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
                               return_feature=return_feature, use_residue=use_residue)
        self.a_fc = nn.Linear(a_dim, n_dim)
        self.v_fc = nn.Linear(v_dim, n_dim)
        self.l_fc = nn.Linear(l_dim, n_dim)
        if self.use_residue:
            self.feature_fc = nn.Linear(n_dim*3+nhidden*3, nhidden)
        else:
            self.feature_fc = nn.Linear(nhidden * 3, nhidden)
        self.final_fc = nn.Linear(nhidden, nclass)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.a_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.v_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.l_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal


    def forward(self, a, v, l, dia_len, qmask):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        if self.use_modal:
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])


        adj = self.create_big_adj(a, v, l, dia_len, self.modals)

        if len(self.modals) == 3:
            features = torch.cat([a, v, l], dim=0).cuda()
        elif 'a' in self.modals and 'v' in self. modals:
            features = torch.cat([a, v], dim=0).cuda()
        elif 'a' in self.modals and 'l' in self.modals:
            features = torch.cat([a, l], dim=0).cuda()
        elif 'v' in self.modals and 'l' in self.modals:
            features = torch.cat([v, l], dim=0).cuda()
        else:
            return NotImplementedError
        features = self.graph_net(features, None, qmask, adj)
        all_length = l.shape[0] if len(l)!=0 else a.shape[0] if len(a) != 0 else v.shape[0]
        if len(self.modals) == 3:
            features = torch.cat([features[:all_length], features[all_length:all_length * 2], features[all_length * 2:all_length * 3]], dim=-1)
        else:
            features = torch.cat([features[:all_length], features[all_length:all_length * 2]], dim=-1)
        if self.return_feature:
            return features
        else:
            return F.softmax(self.final_fc(features), dim=-1)

    def create_big_adj(self, a, v, l, dia_len, modals):
        modal_num = len(modals)
        all_length = l.shape[0] if len(l)!=0 else a.shape[0] if len(a) != 0 else v.shape[0]
        adj = torch.zeros((modal_num*all_length, modal_num*all_length)).cuda()
        if len(modals) == 3:
            features = [a, v, l]
        elif 'a' in modals and 'v' in modals:
            features = [a, v]
        elif 'a' in modals and 'l' in modals:
            features = [a, l]
        elif 'v' in modals and 'l' in modals:
            features = [v, l]
        else:
            return NotImplementedError
        start = 0
        for i in range(len(dia_len)):
            sub_adjs = []
            for j, x in enumerate(features):
                if j < 0:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                else:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i]))
                    temp = x[start:start + dia_len[i]]
                    vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
                    norm_temp = (temp.permute(1, 0) / vec_length)
                    cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)  # seq, seq
                    cos_sim_matrix = cos_sim_matrix * 0.99999
                    sim_matrix = 1 - torch.acos(cos_sim_matrix)/np.pi
                    sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                sub_adjs.append(sub_adj)
            dia_idx = np.array(np.diag_indices(dia_len[i]))
            for m in range(modal_num):
                for n in range(modal_num):
                    m_start = start + all_length*m
                    n_start = start + all_length*n
                    if m == n:
                        adj[m_start:m_start+dia_len[i], n_start:n_start+dia_len[i]] = sub_adjs[m]
                    else:
                        modal1 = features[m][start:start+dia_len[i]] #length, dim
                        modal2 = features[n][start:start+dia_len[i]]
                        normed_modal1 = modal1.permute(1, 0) / torch.sqrt(torch.sum(modal1.mul(modal1), dim=1)) #dim, length
                        normed_modal2 = modal2.permute(1, 0) / torch.sqrt(torch.sum(modal2.mul(modal2), dim=1)) #dim, length
                        dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1) #length
                        dia_cos_sim = dia_cos_sim * 0.99999
                        dia_sim = 1 - torch.acos(dia_cos_sim)/np.pi
                        idx =dia_idx.copy()
                        idx[0] += m_start
                        idx[1] += n_start
                        adj[idx] = dia_sim

            start += dia_len[i]
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D)

        return adj
def cos_sim_features(modal1,modal2):
    # output_no = torch.matmul(modal1,modal2)
    # cos_sim_matrix = torch.sum(output_no, dim=0)  # seq, seq

    cos_sim_matrix = nn.functional.cosine_similarity(modal1,modal2,dim=0)
    cos_sim_matrix = cos_sim_matrix * 0.99999

    #dia_sim =cos_sim_matrix

    dia_sim = 1 - torch.acos(cos_sim_matrix) / np.pi
    #dia_sim = torch.acos(cos_sim_matrix) / np.pi

    # normed_modal1 = modal1.permute(1, 0) / torch.sqrt(torch.sum(modal1.mul(modal1), dim=1))  # dim, length
    # normed_modal2 = modal2.permute(1, 0) / torch.sqrt(torch.sum(modal2.mul(modal2), dim=1)) #dim, length
    # dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1) #length
    # dia_sim = 1 - torch.acos(dia_cos_sim)/np.pi
    # dia_sim = dia_sim*0.99999
    return dia_sim
def cos_sim_features2(modal1,modal2):
    # output_no = torch.matmul(modal1,modal2)
    # cos_sim_matrix = torch.sum(output_no, dim=0)  # seq, seq

    cos_sim_matrix = nn.functional.cosine_similarity(modal1,modal2,dim=0)
    cos_sim_matrix = cos_sim_matrix * 0.99999

    #dia_sim =cos_sim_matrix

    #dia_sim = 1 - torch.acos(cos_sim_matrix) / np.pi
    dia_sim = torch.acos(cos_sim_matrix) / np.pi

    return dia_sim
class MM_GCN2(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False):
        super(MM_GCN2, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph

        self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden, nclass=nclass,
                               dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
                               return_feature=return_feature, use_residue=use_residue)
        self.a_fc = nn.Linear(a_dim, n_dim)
        self.v_fc = nn.Linear(v_dim, n_dim)
        self.l_fc = nn.Linear(l_dim, n_dim)
        if self.use_residue:
            self.feature_fc = nn.Linear(n_dim*3+nhidden*3, nhidden)
        else:
            self.feature_fc = nn.Linear(nhidden * 3, nhidden)
        self.final_fc = nn.Linear(nhidden, nclass)
        self.w_k = nn.Linear(200,400)
        self.w_q = nn.Linear(200,400)
        self.weight = nn.Parameter(torch.Tensor(200 * 2))
        self.weight2 = nn.Parameter(torch.Tensor(900 , 3))
        self.proj = nn.Linear(400,100)
        self.tran = nn.Linear(300, 900)
        self.tran200 = nn.Linear(1,200)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.a_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.v_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.l_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.dropout1 = nn.Dropout(0.359)#0.359
        self.gate = nn.Linear(400,200)
        self.gate1 = nn.Linear(300, 200)
        self.gate3 = nn.Linear(900, 3)
        self.gate2 = nn.Linear(200, 300)
        self.gate4 = nn.Linear(600, 300)
        # self.gate2= nn.Linear(600,300)
        self.cor = nn.Linear(300,200).cuda()




    def forward(self, a, v, l, dia_len, qmask):
        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)
        # lenth = sum(dia_len)
        # bb = 3*lenth
        #self.weight2 = nn.Parameter(torch.Tensor(300 , 3))

        if self.use_speaker:
            if 'v' in self.modals:
                #l = 1.3*l
                #v =0.2*v
                #v -= 0.5*spk_emb_vector
                l += spk_emb_vector
                #a += spk_emb_vector
                #v += spk_emb_vector

        if self.use_modal:
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        ### 设置每段对话的前百分比话语特征为0
        lenth = sum(dia_len)
        non_len= []
        for i in dia_len:
            non_len.append(int(i * 0.5))    ##在此设置百分比


        j = 0
        start = 0
        for i in dia_len:

            g = l[start:start+non_len[j], :]
            l[start:start+non_len[j], :] = 0.0001 * g
            start = start+dia_len[j]
            j += 1
            if start==lenth:
                break

            # b = l[10]
            # b = l[70]
            # b = l[78]
        #均值修复层
        n=0
        j = 0
        start = 0
        k=0
        for b in dia_len:
            temp_ = l[start:start + dia_len[n], :]
            temp_mean = torch.mean(temp_, dim=0)
            for k in l[start:start + dia_len[n], :]:
                k = l[j]
                c = torch.sum(k)
                if torch.abs(c) < 0.01:

                    l[j] = temp_mean
                j+=1
            start += dia_len[n]
            n+=1









        #a1 = sum(dia_len)
        # # features_a_all = torch.cat(adjaa,adjav,adjal)
        # def attention_linear(text, converted_vis_embed_map, vis_embed_map):
        #     # text: batch_size, hidden_dim; converted_vis_embed_map: batch_size, keys_number,embed_size; vis_embed_map: batch_size, keys_number, 2048
        #     # keys_size = converted_vis_embed_map.size(1)
        #     # text = text.unsqueeze(1).expand(-1, keys_size, -1)  # batch_size, keys_number,hidden_dim
        #     attention_inputs = torch.tanh(text + converted_vis_embed_map)
        #     # attention_inputs = F.dropout( attention_inputs )
        #     # att_weights = self.madality_attetion(attention_inputs)  # batch_size, keys_number
        #     att_weights = F.softmax(attention_inputs, dim=-1)  # .view(-1, 1,200)  # batch_size, 1, keys_number
        #     # vis_embed_map = vis_embed_map.unsqueeze(1).permute(0,2,1)
        #
        #     att_vector = torch.mul(att_weights, vis_embed_map)  # batch_size, 2048
        #     return att_vector, att_weights
        #
        # # apply entity-based attention mechanism to obtain different image representations
        # # vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
        # # converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, embed
        # # converted_aspect = self.aspect2text(avg_aspect)
        #
        # # att_vector: batch_size, 2048
        # att_vector, att_weights = attention_linear(a, l, l)
        # converted_att_vis_embed = torch.tanh(att_vector)  # att_vector: batch_size, hidden_dim
        # merge_representation = torch.cat((a, att_vector), dim=-1)
        # gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, hidden_dim
        # a = torch.mul(gate_value, converted_att_vis_embed)
        #
        # att_vector, att_weights = attention_linear(v, l, l)
        # converted_att_vis_embed = torch.tanh(att_vector)  # att_vector: batch_size, hidden_dim
        # merge_representation = torch.cat((v, att_vector), dim=-1)
        # gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, hidden_dim
        # v = torch.mul(gate_value, converted_att_vis_embed)
        #
        # # att_vector, att_weights = attention_linear(l, a, a)
        # # converted_att_vis_embed = torch.tanh(att_vector)  # att_vector: batch_size, hidden_dim
        # # merge_representation = torch.cat((l, att_vector), dim=-1)
        # # gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, hidden_dim
        # # l = torch.mul(gate_value, converted_att_vis_embed)
        # a = 10 * a
        # # l = 1000 * l
        # v = 10 * v
        # # a= torch.tanh(a)
        # # l = torch.sigmoid(l)
        # # v = torch.sigmoid(v)
        # 门控机制代码
        # merge_representation1 = torch.cat((l, v), dim=-1)
        # merge_representation2 = torch.cat((l, a), dim=-1)
        # merge_representation3 = torch.cat((l, l), dim=-1)
        #
        # gate_value1 = torch.sigmoid(self.gate(merge_representation1))
        # gate_value2 = torch.sigmoid(self.gate(merge_representation2))
        # gate_value3 = torch.sigmoid(self.gate(merge_representation3))
        #
        # l = torch.mul(gate_value3, l)
        # v= torch.mul(gate_value1, v)
        # a = torch.mul(gate_value2, a)

        #features_all = torch.cat([gated1_converted_embed,gated2_converted_embed,gated3_converted_embed],dim=-1)
        #以上

        adjaa,adjvv,adjll,adjav,adjal,adjva,adjvl,adjla,adjlv = self.create_big_adj(a, v, l, dia_len, self.modals)
        # testa = torch.sum(adjaa)
        # testa = testa.cpu().detach().numpy()
        # isnan = np.isnan(testa)
        # if isnan:
        #     raise Exception

        # features_avl = a+v+l
        # features_avl = torch.cat([features_avl],dim=0).cuda()
        #
        # features_early = adjaa+adjvv+adjll+adjav+adjal+adjva+adjvl+adjla+adjlv
        # features_early = self.graph_net(features_avl, None, qmask, features_early)
        # features_early = self.tran(features_early)
        # return features_early

        #以下是同模态下构不同dialog的图
        num_all = sum(dia_len)
        features_other_a1 = a.unsqueeze(1).permute(2, 1, 0)
        features_other_a2 = a.unsqueeze(2).permute(1, 0, 2)
        output_a1 = cos_sim_features(features_other_a1, features_other_a2)
        # output_a = F.cosine_similarity(features_other_a1, features_other_a2, dim=0)
        features_other_v1 = v.unsqueeze(1).permute(2, 1, 0)
        features_other_v2 = v.unsqueeze(2).permute(1, 0, 2)
        output_v1 = cos_sim_features(features_other_v1, features_other_v2)
        # output_v = F.cosine_similarity(features_other_v1, features_other_v2, dim=0)
        features_other_l1 = l.unsqueeze(1).permute(2, 1, 0)
        features_other_l2 = l.unsqueeze(2).permute(1, 0, 2)
        output_l1 = cos_sim_features(features_other_l1, features_other_l2)
        # output_l = F.cosine_similarity(features_other_l1, features_other_l2, dim=0)

        zero = torch.zeros(num_all, num_all).cuda()
        one = torch.ones(num_all, num_all).cuda()
        num_alpha = 0.86  # 0.86+0.0011:58.88   0.91+0.0012:58.7

        output_a = torch.where(output_a1 > num_alpha, one, zero)
        output_v = torch.where(output_v1 > num_alpha, one, zero)
        output_l = torch.where(output_l1 > num_alpha, one, zero)

        start_1 = 0
        for i in range(len(dia_len)):
            output_a[start_1:start_1 + dia_len[i], start_1:start_1 + dia_len[i]] = 0
            output_v[start_1:start_1 + dia_len[i], start_1:start_1 + dia_len[i]] = 0
            output_l[start_1:start_1 + dia_len[i], start_1:start_1 + dia_len[i]] = 0
            start_1 += dia_len[i]

        testa = torch.sum(output_a)
        testav = torch.sum(output_v)
        testal = torch.sum(output_l)

        adj_a = torch.zeros(num_all, num_all).cuda()
        adj_v = torch.zeros(num_all, num_all).cuda()
        adj_l = torch.zeros(num_all, num_all).cuda()
        adj_a[:, :] = adjaa
        adj_v[:, :] = adjvv
        adj_l[:, :] = adjll
        test_num = 0.0001  # 0.0001

        adj_a = adj_a + test_num * output_a
        adj_v = adj_v + test_num * output_v
        adj_l = adj_l + test_num * output_l
        adjaa = adj_a
        adjvv = adj_v
        adjll = adj_l
        #以上


        #
        tran = nn.Linear(num_all, 200).cuda()
        output_a2 =tran(output_a1)
        output_l2 = tran(output_l1)
        output_v2 = tran(output_v1)

        avg_out_a = torch.mean(output_a2,dim=0)
        avg_out_l = torch.mean(output_l2, dim=0)
        avg_out_v = torch.mean(output_v2, dim=0)

        var_out_a = torch.var(output_a2,dim=0)
        var_out_l = torch.var(output_l2, dim=0)
        var_out_v = torch.var(output_v2, dim=0)

        output_a3 = torch.div((output_a2 - avg_out_a),var_out_a)
        output_l3 = torch.div((output_l2 - avg_out_l), var_out_l)
        output_v3 = torch.div((output_v2 - avg_out_v), var_out_v)

        output_4 = torch.minimum(output_a3,output_v3)
        output_5 = torch.minimum(output_4 , output_l3)

        # def cal_distance(x, y):
        #     return torch.sum((x - y) ** 2) ** 0.5
        #
        # # iteration func
        # def KNN_by_iter(train_x, train_y, test_x, test_y, k):
        #     k=6
        #
        #     since = time.time()
        #
        #     # cal distance
        #     res = []
        #     for x in tqdm(test_x):
        #         dists = []
        #         for y in train_x:
        #             dists.append(cal_distance(x, y).view(1))
        #
        #         idxs = torch.cat(dists).argsort()[:k]
        #         train_y  = train_y.cpu().detach().numpy()
        #         train_y  = train_y.flatten()
        #         train_y = train_y.astype('float64')
        #         res.append(np.bincount(np.array([train_y[idx] for idx in idxs])).argmax())
        #
        #     # print(res[:10])
        #     print("acc", accuracy_score(test_y, res))
        #
        #     time_elapsed = time.time() - since
        #     print('KNN iter training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))







        # #该做法是将一个模态的邻接矩阵进行三次卷积，理解为串行图卷积
        #
        #
        #
        #
        #
        # features_a = torch.cat([a], dim=0).cuda()
        # features_aa = self.graph_net(features_a, None, qmask, adjaa)
        # features_aa = self.cor(features_aa)
        #
        # features_av = self.graph_net(features_aa, None, qmask, adjav)
        # features_av = self.cor(features_av)
        #
        # features_a_Fin = self.graph_net(features_av, None, qmask, adjal)
        #
        #
        # features_v = torch.cat([v], dim=0).cuda()
        # features_vv = self.graph_net(features_v, None, qmask, adjvv)
        # features_vv = self.cor(features_vv)
        #
        # features_va = self.graph_net(features_vv, None, qmask, adjva)
        # features_va = self.cor(features_va)
        #
        # features_v_Fin = self.graph_net(features_va, None, qmask, adjvl)
        #
        #
        # features_l = torch.cat([l], dim=0).cuda()
        # features_ll = self.graph_net(features_l, None, qmask, adjll)
        # features_ll = self.cor(features_ll)
        #
        # features_la = self.graph_net(features_ll, None, qmask, adjla)
        # features_la = self.cor(features_la)
        #
        # features_l_Fin = self.graph_net(features_la, None, qmask, adjlv)
        #
        #
        # features_aaa = features_a_Fin
        # features_vvv = features_v_Fin
        # features_lll = features_l_Fin



        #并行的图卷积（一致性约束的做法）
        features_a = torch.cat([a], dim=0).cuda()
        features_aa = self.graph_net(features_a, None, qmask, adjaa)

        features_av = self.graph_net(features_a, None, qmask, adjav)

        features_al = self.graph_net(features_a, None, qmask, adjal)

        features_v = torch.cat([v], dim=0).cuda()
        features_vv = self.graph_net(features_v, None, qmask, adjvv)

        features_va = self.graph_net(features_v, None, qmask, adjva)

        features_vl = self.graph_net(features_v, None, qmask, adjvl)

        features_l = torch.cat([l], dim=0).cuda()
        features_ll = self.graph_net(features_l, None, qmask, adjll)

        features_la = self.graph_net(features_l, None, qmask, adjla)

        features_lv = self.graph_net(features_l, None, qmask, adjlv)

        features_aaa = features_aa + features_av + features_al
        features_vvv = features_vv + features_va + features_vl
        features_lll = features_ll + features_la + features_lv

        ##门控机制代码1
        # merge_representation1 = torch.cat((features_lll, features_vvv), dim=-1)
        # merge_representation2 = torch.cat((features_lll, features_aaa), dim=-1)
        # merge_representation3 = torch.cat((features_lll, features_lll), dim=-1)

        # gate_value1 = torch.sigmoid(self.gate1(features_lll))
        # gate_value2 = torch.sigmoid(self.gate1(features_aaa))
        # gate_value3 = torch.sigmoid(self.gate1(features_vvv))
        #
        # gate_value4 = torch.sigmoid(self.gate2(gate_value1))
        # gate_value5 = torch.sigmoid(self.gate2(gate_value2))
        # gate_value6 = torch.sigmoid(self.gate2(gate_value3))
        #
        # gate_value7 = torch.sigmoid(torch.mul(gate_value4,features_lll))
        # gate_value8 = torch.sigmoid(torch.mul(gate_value5, features_aaa))
        # gate_value9 =torch.sigmoid(torch.mul(gate_value6, features_vvv))

        #gate_value10=torch.mean(gate_value7,dim=-1)

        #gate_value10 = gate_value10.unsqueeze(0)
        #gate_value10 = torch.transpose(gate_value10,1,0)
        # features_lll =torch.tanh(gate_value7*features_lll)
        # features_aaa =torch.tanh(gate_value8*features_aaa)
        # features_vvv =torch.tanh(gate_value9*features_vvv)


        # features_vvv = torch.mul(gate_value3, features_vvv)
        # features_lll= torch.mul(gate_value1, features_lll)
        # features_aaa = torch.mul(gate_value2, features_aaa)
        #以上

        # 第一种门控
        #features_a1 = self.gate1(features_aaa)
        #features_v1 = self.gate1(features_vvv)
        #features_l1 = self.gate1(features_lll)

        #features_a2= self.gate2(features_a1)
        #features_v2 = self.gate2(features_v1)
        #features_l2 = self.gate2(features_l1)

        #features_aaa = torch.tanh(torch.mul(features_a2,features_aaa))
        #features_vvv = torch.tanh(torch.mul(features_v2,features_vvv))
        #features_lll = torch.tanh(torch.mul(features_l2,features_lll))



        #第二种门控
        # features_all = torch.cat((features_aaa,features_vvv,features_lll),dim=-1)
        # lenth = sum(dia_len)
        # #bb = 3*lenth
        # #threenums = torch.matmul(features_all,self.weight2)
        # #threenums = threenums+0.00001
        # threenums=self.gate3(features_all)
        # a_nums = threenums[:,0]
        # v_nums = threenums[:, 1]
        # l_nums = threenums[:, 2]
        #
        # #qq=a_nums.size()
        # a_nums = a_nums.unsqueeze(1).expand(lenth,300)
        # v_nums = v_nums.unsqueeze(1).expand(lenth, 300)
        # l_nums = l_nums.unsqueeze(1).expand(lenth, 300)
        # features_aaa=torch.mul(a_nums,features_aaa)
        # features_vvv= torch.mul(v_nums, features_vvv)
        # features_lll = torch.mul(l_nums, features_lll)
        #
        # features_aaa = torch.tanh(features_aaa)
        # features_vvv = torch.tanh( features_vvv)
        # features_lll = torch.tanh(features_lll)


        # #第三种门控  66.57。。。。66.92
        features_av = torch.cat((features_aaa, features_vvv), dim=-1)
        features_al = torch.cat((features_aaa, features_lll), dim=-1)
        features_lv = torch.cat((features_lll, features_vvv), dim=-1)
        features_av2 = self.gate4(features_av)
        features_al2 = self.gate4(features_al)
        features_lv2 = self.gate4(features_lv)
        features_av3 = torch.cat((features_aaa, features_av2), dim=-1)
        features_al3 = torch.cat((features_lll, features_al2), dim=-1)
        features_lv3 = torch.cat((features_vvv, features_lv2), dim=-1)
        features_aaa = torch.tanh(self.gate4(features_av3))
        features_lll = torch.tanh(self.gate4(features_al3))
        features_vvv = torch.tanh(self.gate4(features_lv3) ) #加激活函数试试！！








        #features_fin = KNN_by_iter(features_aaa,features_vvv,features_lll,features_aaa,6)


        # # 定义KNN函数
        # def KNN(train_x, train_y, test_x, test_y, k):
        #     # 获取当前时间
        #     since = time.time()
        #     # 可以将m,n理解为求其数据个数，属于torch.tensor类
        #     m = test_x.size(0)
        #     n = train_x.size(0)
        #
        #     # 计算欧几里得距离矩阵，矩阵维度为m*n；
        #     print("计算距离矩阵")
        #
        #     # test,train本身维度是m*1, **2为对每个元素平方，sum(dim=1，对行求和；keepdim =True时保持二维，
        #     # 而False对应一维，expand是改变维度，使其满足 m * n)
        #     xx = (test_x ** 2).sum(dim=1, keepdim=True).expand(m, n)
        #     # 最后增添了转置操作
        #     yy = (train_x ** 2).sum(dim=1, keepdim=True).expand(n, m).transpose(0, 1)
        #     # 计算近邻距离公式
        #     dist_mat = xx + yy - 2 * test_x.matmul(train_x.transpose(0, 1))
        #     # 对距离进行排序
        #     mink_idxs = dist_mat.argsort(dim=-1)
        #     # 定义一个空列表
        #     res = []
        #     for idxs in mink_idxs:
        #         # voting
        #         # 代码下方会附上解释np.bincount()函数的博客
        #         res.append(np.bincount(np.array([train_y[idx] for idx in idxs[:k]])).argmax())
        #
        #     assert len(res) == len(test_y)
        #     print("acc", accuracy_score(test_y, res))
        #     # 计算运行时长
        #     time_elapsed = time.time() - since
        #     print('KNN mat training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #
        # # 欧几里得距离计算公式
        # def cal_distance(x, y):
        #     return torch.sum((x - y) ** 2) ** 0.5
        #
        # # KNN的迭代函数
        # def KNN_by_iter(train_x, train_y, test_x, test_y, k):
        #     since = time.time()
        #
        #     # 计算距离
        #     res = []
        #     for x in tqdm(test_x):
        #         dists = []
        #         for y in train_x:
        #             dists.append(cal_distance(x, y).view(1))
        #         # torch.cat()用来拼接tensor
        #         idxs = torch.cat(dists).argsort()[:k]
        #         res.append(np.bincount(np.array([train_y[idx] for idx in idxs])).argmax())
        #
        #     # print(res[:10])
        #     print("acc", accuracy_score(test_y, res))
        #
        #     time_elapsed = time.time() - since
        #     print('KNN iter training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #
        # if __name__ == "__main__":
        #     # 加载数据集（下载数据集）
        #     # train_dataset = datasets.MNIST(root="./data", download=True, transform=transforms.ToTensor(), train=True)
        #     # test_dataset = datasets.MNIST(root="./data", download=True, transform=transforms.ToTensor(), train=False)
        #
        #     train_dataset=features_aaa
        #     test_dataset=features_vvv
        #
        #     # 组织训练，测试数据
        #     train_x = []
        #     train_y = []
        #     for i in range(len(train_dataset)):
        #         img, target = train_dataset[i]
        #         train_x.append(img.view(-1))
        #         train_y.append(target)
        #
        #         if i > 5000:
        #             break
        #
        #     # print(set(train_y))
        #
        #     test_x = []
        #     test_y = []
        #     for i in range(len(test_dataset)):
        #         img, target = test_dataset[i]
        #         test_x.append(img.view(-1))
        #         test_y.append(target)
        #
        #         if i > 200:
        #             break
        #
        #     print("classes:", set(train_y))
        #     features_fin = KNN_by_iter(features_aaa, features_vvv, features_lll, features_aaa, 6)
        #     KNN(torch.stack(train_x), train_y, torch.stack(test_x), test_y, 7)
        #     KNN_by_iter(torch.stack(train_x), train_y, torch.stack(test_x), test_y, 7)


        one = features_vvv
        two = features_lll
        three = features_aaa

        b1 = one -two  #V-L
        b2 = one -three
        b3 = two - three
        one = one.permute(1,0)
        two = two.permute(1, 0)
        three = three.permute(1, 0)
        modalbias1 = cos_sim_features2(one,two)
        modalbias2 = cos_sim_features2(one, three)
        modalbias3 = cos_sim_features2(two, three)
        modalbias = 0.8*modalbias1+0.1*modalbias3+0.1*modalbias2#1,1,1:6858，0.8,0.1,0.1:68.94
        #num_all = sum(dia_len)
        #for i in np.arange(len(num_all)):

        # #欧氏距离
        # a1 = torch.sum(torch.pow(b1, 2),dim=1)
        # modalbias4 = torch.sqrt(a1)

        #modalbias = torch.cosine_similarity(one,two)

        # normed_modal1 = one.permute(1, 0) / torch.sqrt(torch.sum(one.mul(one), dim=1))  # dim, length
        # normed_modal2 = two.permute(1, 0) / torch.sqrt(torch.sum(two.mul(two), dim=1)) #dim, length
        # dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1) #length
        # dia_sim = 1 - torch.acos(dia_cos_sim)/np.pi
        acosbias = modalbias*0.99999

        # normed_modal1 = one.permute(1, 0) / torch.sqrt(torch.sum(one.mul(one), dim=1))  # dim, length
        # normed_modal2 = two.permute(1, 0) / torch.sqrt(torch.sum(two.mul(two), dim=1))  # dim, length
        # dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1)  # length
        # dia_sim = 1 - torch.acos(dia_cos_sim) / np.pi
        # acosbias = dia_sim * 0.99999


        # features_aaa = torch.mul(features_aa , features_av)
        # features_aaa = torch.mul(features_aaa, features_al)
        # features_aaa = 100*features_aaa
        # features_vvv = torch.mul(features_vv , features_va)
        # features_vvv = torch.mul(features_vvv, features_vl)
        # features_vvv = 100*features_vvv
        # features_lll = torch.mul(features_ll , features_la)
        # features_lll = torch.mul(features_lll, features_lv)
        # features_lll = 100*features_lll




        # def attention_linear(text, converted_vis_embed_map, vis_embed_map):
        #
        #     attention_inputs = torch.tanh(text + converted_vis_embed_map)
        #     attention_inputs = F.dropout( attention_inputs )
        #     # att_weights = self.madality_attetion(attention_inputs)  # batch_size, keys_number
        #     att_weights = F.softmax(attention_inputs, dim=-1)  # .view(-1, 1,200)  # batch_size, 1, keys_number
        #     att_vector = torch.mul(att_weights, vis_embed_map)  # batch_size, 2048
        #     return att_vector, att_weights
        #
        # # apply entity-based attention mechanism to obtain different image representations
        # # vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
        # # converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, embed
        # # converted_aspect = self.aspect2text(avg_aspect)
        #
        # # att_vector: batch_size, 2048
        # att_vector, att_weights = attention_linear(features_aaa, features_lll, features_lll)
        # converted_att_vis_embed = torch.tanh(att_vector)  # att_vector: batch_size, hidden_dim
        # merge_representation = torch.cat((features_aaa, att_vector), dim=-1)
        # gate_value = torch.sigmoid(self.gate2(merge_representation))  # batch_size, hidden_dim
        # features_aaa = torch.mul(gate_value, converted_att_vis_embed)
        #
        # att_vector, att_weights = attention_linear(features_vvv,  features_lll, features_lll)
        # converted_att_vis_embed = torch.tanh(att_vector)  # att_vector: batch_size, hidden_dim
        # merge_representation = torch.cat((features_vvv, att_vector), dim=-1)
        # gate_value = torch.sigmoid(self.gate2(merge_representation))  # batch_size, hidden_dim
        # features_vvv = torch.mul(gate_value, converted_att_vis_embed)
        #
        # # att_vector, att_weights = attention_linear(features_lll, features_aaa, features_aaa)
        # # converted_att_vis_embed = torch.tanh(att_vector)  # att_vector: batch_size, hidden_dim
        # # merge_representation = torch.cat((features_lll, att_vector), dim=-1)
        # # gate_value = torch.sigmoid(self.gate2(merge_representation))  # batch_size, hidden_dim
        # # features_lll = torch.mul(gate_value, converted_att_vis_embed)
        #
        # #features_lll = 100 * features_lll
        # features_aaa = 80*features_aaa
        # features_vvv = 80*features_vvv

        #




        # # 门控机制代码
        # merge_representation1 = torch.cat((features_aaa, features_vvv), dim=-1)
        # merge_representation2 = torch.cat((features_vvv, features_lll), dim=-1)
        # merge_representation3 = torch.cat((features_lll, features_aaa), dim=-1)
        #
        # gate_value1 = torch.sigmoid(self.gate(merge_representation1))
        # gate_value2 = torch.sigmoid(self.gate(merge_representation2))
        # gate_value3 = torch.sigmoid(self.gate(merge_representation3))
        #
        # gated1_converted_embed = torch.mul(gate_value1, features_aaa)
        # gated2_converted_embed = torch.mul(gate_value2, features_vvv)
        # gated3_converted_embed = torch.mul(gate_value3, features_lll)
        #
        # #features_all = torch.cat([gated1_converted_embed,gated2_converted_embed,gated3_converted_embed],dim=-1)
        # #以上

        features_all = torch.cat([features_aaa,features_vvv,features_lll],dim=-1)






        return features_all,modalbias,acosbias








    def create_big_adj(self, a, v, l, dia_len, modals):

        modal_num = len(modals)
        all_length = l.shape[0] if len(l)!=0 else a.shape[0] if len(a) != 0 else v.shape[0]
        adjaa = torch.zeros((all_length, all_length)).cuda()
        adjvv = torch.zeros((all_length, all_length)).cuda()
        adjll = torch.zeros((all_length, all_length)).cuda()
        adjav = torch.zeros((all_length, all_length)).cuda()
        adjal = torch.zeros((all_length, all_length)).cuda()
        adjva = torch.zeros((all_length, all_length)).cuda()
        adjvl = torch.zeros((all_length, all_length)).cuda()
        adjla = torch.zeros((all_length, all_length)).cuda()
        adjlv = torch.zeros((all_length, all_length)).cuda()
        if len(modals) == 3:
            features = [a, v, l]
        elif 'a' in modals and 'v' in modals:
            features = [a, v]
        elif 'a' in modals and 'l' in modals:
            features = [a, l]
        elif 'v' in modals and 'l' in modals:
            features = [v, l]
        else:
            return NotImplementedError
        start = 0
        start1 =0
        start2 =0
        start3 =0
        idx1 =0
        idx2 =0
        idx3 =0
        idx4 =0
        idx5 =0
        idx6 =0


        for i in range(len(dia_len)):
            sub_adjs = []
            for j, x in enumerate(features):
                if j < 0:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                else:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i]))
                    temp = x[start:start + dia_len[i]]
                    vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
                    norm_temp = (temp.permute(1, 0) / vec_length)
                    #以下是原文代码
                    output_no = torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1))
                    cos_sim_matrix = torch.sum(output_no, dim=0)  # seq, seq
                    cos_sim_matrix = cos_sim_matrix * 0.99999
                    sim_matrix = 1 - torch.acos(cos_sim_matrix)/np.pi
                    # testa = torch.sum(sim_matrix)
                    # testa = testa.cpu().detach().numpy()
                    # isnan = np.isnan(testa)
                    # if isnan:
                    #     raise Exception

                    sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                    #以上

                    # #所有矩阵仅对角
                    # norm_temp = norm_temp.permute(1,0)
                    # normed_modal1 = torch.unsqueeze(norm_temp, dim=1)
                    # normed_modal2 = torch.unsqueeze(norm_temp, dim=1)
                    # mb_size = normed_modal2.shape[0]
                    # k_len = normed_modal2.shape[1]
                    # q_len = normed_modal1.shape[1]
                    #
                    # kx = self.w_k(normed_modal2).view(mb_size, k_len, 2, 200)
                    # kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, 200)
                    # qx = self.w_q(normed_modal1).view(mb_size, q_len, 2, 200)
                    # qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, 200)
                    #
                    # kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
                    # qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
                    # # kq = torch.cat((kxx, qxx), dim=-1)
                    # # score = torch.tanh(torch.sum(kq, dim=-1))
                    #
                    # kq = torch.cat((kxx, qxx), dim=-1)
                    # score = torch.tanh(torch.matmul(kq, self.weight))
                    #
                    # # score = torch.sum(kxx.mul(qxx),dim=-1)
                    # # score = torch.tanh(score)
                    #
                    # score = F.softmax(score, dim=0)
                    # output = torch.bmm(score, kx)
                    # output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
                    # output = self.proj(output)
                    # dia_sim = self.dropout1(output)
                    # dia_sim = dia_sim.squeeze(1)
                    # dia_sim = torch.sum(dia_sim, dim=1)
                    # sub_adj = torch.diag_embed(dia_sim)
                    # #以上
                sub_adjs.append(sub_adj)
            dia_idx = np.array(np.diag_indices(dia_len[i]))
            for m in range(modal_num):
                for n in range(modal_num):
                    m_start = start + all_length*m
                    n_start = start + all_length*n
                    if m == n:
                        if m==0:
                            adjaa[start1:start1+dia_len[i], start1:start1+dia_len[i]] = sub_adjs[m]
                            start1+=dia_len[i]
                        if m==1:
                            adjvv[start2:start2+ dia_len[i],start2:start2+ dia_len[i]] = sub_adjs[m]
                            start2 += dia_len[i]
                        if m==2:
                            adjll[start3: start3+dia_len[i],start3: start3+dia_len[i]] = sub_adjs[m]
                            start3 += dia_len[i]
                    else:
                        modal1 = features[m][start:start+dia_len[i]] #length, dim
                        modal2 = features[n][start:start+dia_len[i]]
                        testaa = torch.sum(modal1,dim=-1)

                        testa =torch.sum(testaa)
                        testa = testa.cpu().detach().numpy()
                        isnan = np.isnan(testa)
                        # if isnan:
                        #     raise Exception

                        # for j, x in enumerate(features):
                        #     if j < 0:
                        #         sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                        #     else:
                        #         sub_adj = torch.zeros((dia_len[i], dia_len[i]))
                        #         temp = x[start:start + dia_len[i]]
                        #         vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
                        #         norm_temp = (temp.permute(1, 0) / vec_length)
                        #         cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)),
                        #                                    dim=0)  # seq, seq
                        #         cos_sim_matrix = cos_sim_matrix * 0.99999
                        #         sim_matrix = 1 - torch.acos(cos_sim_matrix) / np.pi
                        #         sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix

                        # 以下是原文代码
                        normed_modal1 = modal1.permute(1, 0) / torch.sqrt(torch.sum(modal1.mul(modal1), dim=1)) #dim, length
                        normed_modal2 = modal2.permute(1, 0) / torch.sqrt(torch.sum(modal2.mul(modal2), dim=1)) #dim, length
                        dia_cos_sim = torch.matmul(normed_modal1.unsqueeze(2),normed_modal2.unsqueeze(1))
                        dia_cos_sim = torch.sum(dia_cos_sim, dim=0) #length
                        dia_sim = 1 - torch.acos(dia_cos_sim)/np.pi
                        dia_sim = dia_sim*0.99999
                        # 以上

                        # # 模态之间的新构图法
                        # dia_sim_diag2 = torch.zeros(dia_len[i],dia_len[i])
                        # dia_sim_diag2[:dia_len[i], :dia_len[i]] = dia_sim


                        # #att
                        # normed_modal1 = torch.unsqueeze(modal1, dim=1)
                        # normed_modal2 = torch.unsqueeze(modal2, dim=1)
                        # mb_size = normed_modal2.shape[0]
                        # k_len = normed_modal2.shape[1]
                        # q_len = normed_modal1.shape[1]
                        #
                        # kx = self.w_k(normed_modal2).view(mb_size, k_len, 2, 200)  #view改变tensor的形状
                        # kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, 200)
                        # qx = self.w_q(normed_modal1).view(mb_size, q_len, 2, 200)
                        # qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, 200)
                        #
                        # kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
                        # qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
                        # kq = torch.cat((kxx, qxx), dim=-1)
                        # weight_kq = torch.matmul(kq, self.weight)
                        # score = torch.tanh(weight_kq)
                        # score = F.softmax(score, dim=0)
                        # #output = self.tran200(score)
                        # #output = 0.01*output  #67.47
                        # output = torch.bmm(score, kx)
                        # #number_b = kq.shape[0]
                        # #output = kx/number_b
                        # output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
                        # output = self.proj(output)
                        # dia_sim = self.dropout1(output)
                        # dia_sim = dia_sim.squeeze(1)
                        # dia_sim = torch.sum(dia_sim,dim=1)
                        # #以上
                        dia_sim_diag = 0.01*dia_sim
                        #dia_sim_diag = torch.diag_embed(dia_sim)



                        idx =dia_idx.copy()
                        idx[0] += m_start
                        idx[1] += n_start
                        if m == 0 and n == 1:
                            adjav[idx1:idx1+dia_len[i],idx1:idx1+dia_len[i]]= dia_sim_diag
                            idx1 += dia_len[i]
                        if m == 0 and n == 2:
                            adjal[idx2:idx2+dia_len[i],idx2:idx2+dia_len[i]] = dia_sim_diag
                            idx2 += dia_len[i]
                        if m == 1 and n == 0:
                            adjva[idx3:idx3+dia_len[i],idx3:idx3+dia_len[i]]= dia_sim_diag
                            idx3 += dia_len[i]
                        if m == 1 and n == 2:
                            adjvl[idx4:idx4+dia_len[i],idx4:idx4+dia_len[i]]= dia_sim_diag
                            idx4 += dia_len[i]
                        if m == 2 and n == 0:
                            adjla[idx5:idx5+dia_len[i],idx5:idx5+dia_len[i]]= dia_sim_diag
                            idx5 += dia_len[i]
                        if m == 2 and n == 1:
                            adjlv[idx6:idx6+dia_len[i],idx6:idx6+dia_len[i]] = dia_sim_diag
                            idx6 += dia_len[i]



            start += dia_len[i]

        daa = adjaa.sum(1)
        dvv = adjvv.sum(1)
        dll = adjll.sum(1)
        dav = adjav.sum(1)
        dal = adjal.sum(1)
        dva = adjva.sum(1)
        dvl = adjvl.sum(1)
        dla = adjla.sum(1)
        dlv = adjlv.sum(1)
        da = daa+dav+dal
        dv = dva+dvv+dvl
        dl = dla+dlv+dll

        Daa = torch.diag(torch.pow(da, -0.5))
        Dvv = torch.diag(torch.pow(dv, -0.5))
        Dll = torch.diag(torch.pow(dl, -0.5))
        #仅对角
        # daa = torch.pow(daa, 2)
        # Daa = torch.diag(torch.pow(daa, -0.25))
        # dvv = torch.pow(dvv, 2)
        # Dvv = torch.diag(torch.pow(dvv, -0.25))
        # dll = torch.pow(dll, 2)
        # Dll = torch.diag(torch.pow(dll, -0.25))
        #以上
        dav = torch.pow(dav, 2)
        Dav = torch.diag(torch.pow(dav,-0.25))
        dal = torch.pow(dal, 2)
        Dal = torch.diag(torch.pow(dal,-0.25))
        dva = torch.pow(dva, 2)
        Dva = torch.diag(torch.pow(dva,-0.25))
        dvl = torch.pow(dvl, 2)
        Dvl = torch.diag(torch.pow(dvl,-0.25))
        dla = torch.pow(dla, 2)
        Dla = torch.diag(torch.pow(dla,-0.25))
        dvl = torch.pow(dlv, 2)
        Dlv = torch.diag(torch.pow(dvl,-0.25))
        testa = 1
        # Daa = (Daa+Dav+Dal)
        # Dvv = (Dva+Dvv+Dvl)
        # Dll = (Dla+Dlv+Dlv)
        adjaa = Daa.mm(adjaa).mm(Daa)
        #adjaa = testa*adjaa
        adjvv = Dvv.mm(adjvv).mm(Dvv)
        #adjvv = testa*adjvv
        adjll = Dll.mm(adjll).mm(Dll)
        #adjll = testa*adjll
        # adjav = Dav.mm(adjav).mm(Dav)
        # adjal = Dal.mm(adjal).mm(Dal)
        # adjva = Dva.mm(adjva).mm(Dva)
        # adjvl = Dvl.mm(adjvl).mm(Dvl)
        # adjla = Dla.mm(adjla).mm(Dla)
        # adjlv = Dlv.mm(adjlv).mm(Dlv)
        # testb = 5
        # adjaa = testb * adjaa
        # adjvv = testb * adjvv
        # adjll = testb * adjll
        # adjav = testb * adjav
        # adjal = testb * adjal
        # adjva = testb * adjva
        # adjvl = testb * adjvl
        # adjla = testb * adjla
        # adjlv = testb * adjlv


        return adjaa,adjvv,adjll,adjav,adjal,adjva,adjvl,adjla,adjlv
