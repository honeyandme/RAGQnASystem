import os
import random
#自动机 用于字符串匹配
import ahocorasick
import re
from tqdm import tqdm
import sys
class Build_Ner_data():
    """
        这是一个ner数据生成类。主要作用是将data文件夹下的medical.json文件中的文本打上标签。
        这里有四类标签["食物","药品商","治疗方法","药品"]，每种标签所对应的实体在data文件夹下的f'{type}.txt'中
        这里将每种实体导入到ahocorasick中，对每个文本进行模式匹配。
        """
    def __init__(self):
        self.idx2type=idx2type = ["疾病","疾病症状","检查项目","科目","食物","药品商","治疗方法","药品"]
        self.type2idx=type2idx = {"疾病":0,"疾病症状":1,"检查项目":2,"科目":3,"食物":4,"药品商":5,"治疗方法":6,"药品":7}
        self.max_len = 30
        self.p = ['，', '。' , '！' , '；' , '：' , ',' ,'.','?','!',';']
        self.ahos = [ahocorasick.Automaton() for i in range(len(idx2type))]

        for type in idx2type:
            with open(os.path.join('data','ent_aug',f'{type}.txt'),encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                if len(en)>=2:
                    self.ahos[type2idx[type]].add_word(en,en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()
    def split_text(self,text):
        """
        将长文本随机分割为短文本

        :param arg1: 长文本
        :return: 返回一个list,代表分割后的短文本
        :rtype: list
        """
        text = text.replace('\n',',')
        pattern = r'([，。！；：,.?!;])(?=.)|[？,]'

        sentences = []

        for s in re.split(pattern, text):
            if s and len(s)>0:
                sentences.append(s)

        sentences_text = [x for x in sentences if x not in self.p]
        sentences_Punctuation = [x for x in sentences[1::2] if x in self.p]
        split_text = []
        now_text = ''

        #随机长度,有15%的概率生成短文本 10%的概率生成长文本
        for i in range(len(sentences_text)):
            if (len(now_text)> self.max_len and random.random()<0.9 or random.random()<0.15) and len(now_text)>0:
                split_text.append(now_text)
                now_text = sentences_text[i]
                if i < len(sentences_Punctuation):
                    now_text += sentences_Punctuation[i]
            else:
                now_text += sentences_text[i]
                if i < len(sentences_Punctuation):
                    now_text+=sentences_Punctuation[i]
        if len(now_text)>0:
            split_text.append(now_text)

        #随机选取30%的数据,把末尾标点改为。
        for i in range(len(split_text)):
            if random.random()<0.3:
                if(split_text[i][-1] in self.p):
                    split_text[i] = split_text[i][:-1]+'。'
                else:
                    split_text[i] = split_text[i]+'。'
        return split_text
    def make_text_label(self,text):
        """
        通过ahocorasick类对文本进行识别，创造出文本的ner标签

        :param arg1: 文本
        :return: 返回一个list,代表标签
        :rtype: list
        """
        label = ['O']*len(text)
        flag = 0
        mp = {}
        for type in self.idx2type:
            li = list(self.ahos[self.type2idx[type]].iter(text))
            if len(li)==0:
                continue
            li = sorted(li,key=lambda x:len(x[1]),reverse=True)
            for en in li:
                ed,name = en
                st = ed-len(name)+1
                if st in mp or ed in mp:
                    continue
                label[st:ed+1] = ['B-'+type] + ['I-'+type]*(ed-st)
                flag = flag+1
                for i in range(st,ed+1):
                    mp[i] = 1
        return label,flag

#将文本和对应的标签写入ner_data2.txt
def build_file(all_text,all_label):
    with open(os.path.join('data','ner_data_aug.txt'),"w",encoding="utf-8") as f:
        for text, label in zip(all_text, all_label):
            for t, l in zip(text, label):
                f.write(f'{t} {l}\n')
            f.write('\n')
if __name__ == "__main__":
    print('hehe')
    with open(os.path.join('data','medical.json'),'r',encoding='utf-8') as f:
        all_data = f.read().split('\n')
    build_ner_data = Build_Ner_data()

    all_text,all_label = [],[]

    for data in tqdm(all_data):
        if len(data)<2:
            continue
        data = eval(data)
        data_text = [data.get("desc",""),data.get("prevent", ""),data.get("cause", "")]

        data_text_split = []
        for text in data_text:
            if len(text)==0:
                continue
            text_split = build_ner_data.split_text(text)
            for tmp in text_split:
                if len(tmp)>0:
                    data_text_split.append(tmp)
        for text in data_text_split:
            if len(text)==0:
                continue
            label,flag = build_ner_data.make_text_label(text)
            if flag>=1:
                assert (len(text) == len(label))
                all_text.append(text)
                all_label.append(label)

    build_file(all_text,all_label)

