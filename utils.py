import itertools
import numpy as np
import torch, pickle
from transformers.pipelines import FillMaskPipeline
from transformers import BertForMaskedLM


class Glove(torch.nn.Module):
    def __init__(self, weight, word2id, id2word):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(weight)
        self.word2id = word2id
        self.id2word = id2word

    def forward(self, x):
        """x是一个list, 返回两两相似度的值"""
        res = {}
        embed = self.embedding(x)
        for i, j in itertools.combinations(range(embed.shape[1]), 2):
            res[(i, j)] = torch.cosine_similarity(embed[:, i], embed[:, j], dim=1)
        return res

    def encoding(self, triples):
        # 输入batch*n的字符 或者 id
        input = []
        for words in triples:
            #             input.append([self.word2id[word] for word in words])
            input.append([self.word2id[word.split(' ')[-1]] for word in words])

        return torch.from_numpy(np.array(input))
class POSVocab():
    """词表类 包含形容词词表和名词词表"""
    def __init__(self, tokenizer, adj_file, noun_file):
        self.tokenizer = tokenizer

        self.vocab_file = {}
        self.vocab_file['adj'] = adj_file
        self.vocab_file['noun'] = noun_file

        self.words = {}
        with open(self.vocab_file['adj'], 'rb') as f:
            self.words['adj'] = pickle.load(f)

        with open(self.vocab_file['noun'], 'rb') as f:
            self.words['noun'] = pickle.load(f)

        self.mask = {}
        self.mask['adj'] = self.getVocabMask(tokenizer, self.vocab_file['adj'])
        self.mask['noun'] = self.getVocabMask(tokenizer, self.vocab_file['noun'])

    def getVocabMask(self, tokenizer, vocab_file):
        """返回一个mask矩阵, mask掉那些不在词表里的单词"""
        with open(vocab_file, 'rb') as f:
            words = set(pickle.load(f))

        vocab = tokenizer.vocab
        mask = [False] * len(vocab)
        for word, i in vocab.items():
            if word in words:
                mask[i] = True

        return torch.tensor(mask)


class MetaphorGenerator(FillMaskPipeline):
    def __init__(self, model, tokenizer, pos_vocab, device):
        super().__init__(model, tokenizer)
        self.mask_token = tokenizer.mask_token
        self.pos_vocab = pos_vocab
        self.device = device if device else torch.device("cpu")
        self.model.to(self.device)

    def generate(self, eles, score_fnt, pattern_fnt, top_k=15):
        sorted_scores, origin_scores, ids, topk_tokens = [], [], [], []
        for ele in eles:
            # 要预测的元素
            if ele[1] is None:
                mode = 'noun'
            elif ele[2] is None:
                mode = 'adj'

            # 构造模板, 前向传播
            sentence, weight = pattern_fnt(ele, self.tokenizer.mask_token)
            inputs = self._parse_and_tokenize(sentence)
            outputs = self._forward(inputs, return_tensors=True)

            # 取出[MASK]位置的logit
            masked_index = torch.nonzero(inputs["input_ids"] == self.tokenizer.mask_token_id, as_tuple=True)
            logits = outputs[masked_index[0], masked_index[1], :]

            # 用词表筛选一下
            if self.pos_vocab is not None:
                logits.masked_fill_(~self.pos_vocab.mask[mode], -1e9)

            # 计算分数, 返回分数
            score = score_fnt(logits, ele=ele, pos_vocab=self.pos_vocab, weight=weight, mode=mode)
            score_sorted, id = score.topk(top_k)
            # TODO 子词这里会预测错误
            token = self.tokenizer.decode(id).split(' ')

            sorted_scores.append(score_sorted)
            origin_scores.append(score)
            ids.append(id)
            topk_tokens.append(token)

        return origin_scores, topk_tokens



from utils import POSVocab, MetaphorGenerator
from transformers import BertForMaskedLM, BertTokenizer
import torch


class PSGenerator(MetaphorGenerator):
    """pattern search方法的比喻生成器"""
    def __init__(self, bert_path='.\model\Bert_ant', glove_path='.\model\glove'):
        model = BertForMaskedLM.from_pretrained(bert_path)
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        vocabs = POSVocab(tokenizer, adj_file='vocab/adj_4800', noun_file="vocab/noun_concrete")
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        self.glove = torch.load(glove_path)
        super().__init__(model, tokenizer, vocabs, device)

    def SI_pattern(self, ele, mask_token):
        tenor = ele[0]
        vehile = ele[1] if ele[1] else mask_token
        attr = ele[2] if ele[2] else mask_token
        sents = [
            f"{vehile} is very {attr}.",
            f"The {tenor} is as {attr} as {vehile}.",
        ]

        return sents, None

    def SI_score(self, logit, **kwargs):
        # 取出词表中的id的logit
        pos_vocab = kwargs['pos_vocab']
        mode = kwargs['mode']
        idx = torch.nonzero(pos_vocab.mask['adj'], as_tuple=True)
        tmp = logit[:, idx[0]]
        # softmax
        tmp = torch.softmax(tmp, dim=1)
        logit[:, idx[0]] = torch.log(tmp)
        return torch.mean(logit, dim=0)


    def SI(self, tenor, vehicle, topk=15):
        input = [[tenor, vehicle, None]]
        return self.generate(input, self.SI_score, self.SI_pattern, topk)[1][0]

    def SG_pattern(self, ele, mask_token):
        tenor = ele[0]
        vehicle = ele[1] if ele[1] else mask_token
        attr = ele[2] if ele[2] else mask_token
        sents = [
            f"the {attr} {vehicle}.",
            f"{tenor} is like {vehicle}, because they are both {attr}.",
        ]
        return sents, None

    def SG_score(self, logit, **kwargs):
        # 喻体的logit
        pos_vocab = kwargs['pos_vocab']
        mode = kwargs['mode']
        # 获取本体与所有名词之间的余弦距离
        tenor = kwargs['ele'][0]
        noun_list = pos_vocab.words[mode]

        inputs = []
        for noun in noun_list:
            if noun in self.glove.id2word:
                inputs.append([tenor, noun])

        # 过滤与本体相近的词, 剩下所有的词都是可能的候选词
        sim = self.glove(self.glove.encoding(inputs))[(0,1)] # todo 一个个求太慢 弄成缓存; 如果vehicle不在glove里, 可能报错
        mask = [False] * len(pos_vocab.tokenizer.vocab)
        for i in range(len(inputs)):
            if sim[i]<=0.5:
                word = inputs[i][1]
                mask[self.tokenizer.vocab[word]] = True
        mask = torch.tensor(mask)
        logit.masked_fill_(~mask, -1e9)

        # 取出候选词的logit, 计算每个候选词概率
        idx = torch.nonzero(mask, as_tuple=True)
        tmp = logit[:, idx[0]]
        tmp = torch.softmax(tmp, dim=1)
        tmp = torch.log(tmp)
        logit[:, idx[0]] = tmp
        return torch.sum(logit, dim=0)

    def SG(self, tenor, attr, topk=15):
        input = [[tenor, None, attr]]
        return self.generate(input, self.SG_score, self.SG_pattern, topk)[1]

