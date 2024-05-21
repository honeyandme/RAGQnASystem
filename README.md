# 基于RAG与大模型技术的医疗问答系统

本项目使用的数据集来源于[Open-KG](http://data.openkg.cn/dataset/disease-information)，参考了[RAGOnMedicalKG](https://github.com/liuhuanyong/RAGOnMedicalKG)、[QASystemOnMedicalKG](https://github.com/liuhuanyong/QASystemOnMedicalKG)

## 介绍

本项目整体流程：

<img src="img/all.png" style="zoom:100%;" />


本项目设计了一个基于 RAG 与大模型技术的医疗问答系统，利用 DiseaseKG 数据集与 Neo4j 构建知识图谱，结合 BERT 的命名实体识别和 34b 大模型的意图识别，通过精确的知识检索和问答生成，提升系统在医疗咨询中的性能，解决大模型在医疗领域应用的可靠性问题。

RAG技术：

<img src="img/RAG.png" style="zoom:100%;" />



本项目采用知识图谱实现RAG，如果您想用向量数据库实现RAG技术，请移步[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)：

<img src="img/langchain+chatglm.png" style="zoom:50%;" />

本项目主要贡献：

(1) 传统的 RAG 技术通常是利用向量数据库实现的。区别于传统的 RAG 实现方 式，本项目采用了知识图谱，为大模型提供了更加精确的外部信息。

(2) 本项目构建了一个医疗领域的知识图谱，并采用大语言模型优化了知识图谱数 据集文件的实体信息，使得构建出的知识图谱更加准确与科学。

(3) 本项目通过规则匹配的方式构建了一个实体识别数据集（NER），得益于（2） 在实体名字上的优化，我们的模型可以轻松的在构建的数据集上表现出极高的性能。

(4) 本项目针对实体识别任务提出并实施了三种数据增强策略：实体替换、实体掩 码和实体拼接， 提升了 RoBERTa 模型的性能。 在测试集上，这些数据增强措施使得 RoBERTa 模型的 F1 分数从原来的 96.77%提升至 97.40%。

(5) 为了避免数据标注所造成的人工成本，本项目直接设计 Prompt，结合上下文学习与思维链技术，采用大语言模型对用户的提问进行意图识别。这种方法在减少人工成本的基础上保证了意图识别过程的准确度。

(6) 本项目使用 Streamlit 框架对上述模型进行部署，实现了高度封装。我们的界面 涵盖了注册与登录、大语言模型的选择、创建多个聊天窗口等多项功能。

## :fire:To do

- [x] 增加界面的功能(2024.5.21)：增加了登陆、注册界面(含用户、管理员2个身份)，大模型选择按钮(可选千问和llama)、多窗口对话功能等。
- [ ] NL2Cyhper
- [ ] 更多优化...

## Python环境配置

一个例子:

```
git clone https://github.com/honeyandme/RAGQnASystem.git
cd RAGQnASystem
conda create -n RAGQnASystem python=3.10
conda activate RAGQnASystem
pip install -r requirements.txt
```

## 构建知识图谱

首先需要安装Neo4j，[官方网站](https://neo4j.com/deployment-center/#community)。本项目使用的版本是neo4j-community-5.18.1，需要依赖jdk17。

安装并运行Neo4j后，我们需要根据```data/medical_new_2.json```数据集创建一个知识图谱。

```
python build_up_graph.py --website YourWebSite --user YourUserName --password YourPassWord --dbname YourDBName
```

其中，```--website```代表你的Neo4j网址，```--user```代表你的数据库用户名，```--password```代表你的数据库密码，```--dbname```代表你的数据库名字。

示例:

```
python build_up_graph.py --website http://localhost:7474 --user neo4j --password YourPassWord --dbname neo4j
```

运行```build_up_graph.py```后，会自动在```data```文件夹下创建```ent_aug```文件夹和```rel_aug.txt```文件，分别存放所有实体和关系。



下表展示了```medical_new_2.json```中的关键信息，更多详细信息请点击[这里](https://github.com/nuolade/disease-kb)查看：

知识图谱实体类型（8类实体）：

| 实体类型   | 中文含义 | 实体数量 | 举例               |
| ---------- | -------- | -------- | ------------------ |
| Disease    | 疾病     | 8808     | 急性肺脓肿         |
| Drug       | 药品     | 3828     | 布林佐胺滴眼液     |
| Food       | 食物     | 4870     | 芝麻               |
| Check      | 检查项目 | 3353     | 胸部CT检查         |
| Department | 科目     | 54       | 内科               |
| Producer   | 在售药品 | 17,201   | 青阳醋酸地塞米松片 |
| Symptom    | 疾病症状 | 5,998    | 乏力               |
| Cure       | 治疗方法 | 544      | 抗生素药物治疗     |
| Total      | 总计     | 44,656   | 约4.4万实体量级    |

疾病实体属性类型（7类属性）：

| 属性类型      | 中文含义     | 举例                          |
| ------------- | ------------ | ----------------------------- |
| name          | 疾病名称     | 成人呼吸窘迫综合征            |
| desc          | 疾病简介     | 成人呼吸窘迫综合征简称ARDS... |
| cause         | 疾病病因     | 化脓性感染可使细菌毒素...     |
| prevent       | 预防措施     | 对高危的患者应严密观察...     |
| cure_lasttime | 治疗周期     | 2-4月                         |
| cured_prob    | 治愈概率     | 85%                           |
| easy_get      | 疾病易感人群 | 无特定的人群                  |

知识图谱关系类型（11类关系）：

| 实体关系类型   | 中文含义     | 关系数量 | 举例                                     |
| -------------- | ------------ | -------- | ---------------------------------------- |
| belongs_to     | 属于         | 8,843    | <内科,属于, 呼吸内科>                    |
| common_drug    | 疾病常用药品 | 14,647   | <成人呼吸窘迫综合征,常用, 人血白蛋白>    |
| do_eat         | 疾病宜吃食物 | 22,230   | <成人呼吸窘迫综合征,宜吃,莲子>           |
| drugs_of       | 药品在售药品 | 17,315   | <人血白蛋白,在售,莱士人蛋白人血白蛋白>   |
| need_check     | 疾病所需检查 | 39,418   | <单侧肺气肿,所需检查,支气管造影>         |
| no_eat         | 疾病忌吃食物 | 22,239   | <成人呼吸窘迫综合征,忌吃, 啤酒>          |
| recommand_drug | 疾病推荐药品 | 59,465   | <混合痔,推荐用药,京万红痔疮膏>           |
| recommand_eat  | 疾病推荐食谱 | 40,221   | <成人呼吸窘迫综合征,推荐食谱,百合糖粥>   |
| has_symptom    | 疾病症状     | 54,710   | <成人呼吸窘迫综合征,疾病症状,呼吸困难>   |
| acompany_with  | 疾病并发疾病 | 12,024   | <成人呼吸窘迫综合征,并发疾病,细菌性肺炎> |
| cure_way       | 疾病治疗方法 | 21，047  | <急性肺脓肿,治疗方法,抗生素药物治疗>     |
| Total          | 总计         | 312,159  | 约31万关系量级                           |

创建的知识图谱如下图所示（某一检索结果）：

<img src="img/neo4j.png" style="zoom:100%;" />

## 实体识别(NER)

什么是NER？

<img src="img/shitishibie.png" style="zoom:90%;" />





**<u>数据集创建：</u>**

你可以运行```ner_data.py```，这段代码会根据```data/medical_new_2.json```中的文字，结合规则匹配技术，创建一个NER数据集，保存在```data/ner_data_aug.txt```中。

```
python ner_data.py #可以不运行
```

注1：我们已经上传了```ner_data_aug.txt```文件，您可以选择不运行```ner_data.py```。

注2：我们采用BIO的策略对数据集进行标注，标注的结果如下图所示：

<img src="img/nerdata.png" style="zoom:40%;" />



**<u>模型训练：</u>**

```ner_model.py``` 代码定义了NER模型的网络架构和训练方式。若您需要重新训练一个模型，请您在Huggingface上下载一个[chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)，并保存在```model```文件夹下，然后运行```ner_model.py``` 。

```
python ner_model.py #可以不运行
```

注1：若您不想训练，可以[下载](https://pan.baidu.com/s/1kwiNDyNjO2E2uO0oYmK8SA?pwd=08or)我们训练好的模型，并保存在```model```文件夹下，无需运行训练代码。

注2：我们的NER模型采用了简单的BERT架构。

```python
class Bert_Model(nn.Module):
    def __init__(self, model_name, hidden_size, tag_num, bi):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=bi)
        if bi:
            self.classifier = nn.Linear(hidden_size*2, tag_num)
        else:
            self.classifier = nn.Linear(hidden_size, tag_num)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, label=None):
        bert_0, _ = self.bert(x, attention_mask=(x > 0), return_dict=False)
        gru_0, _ = self.lstm(bert_0)
        pre = self.classifier(gru_0)
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1).squeeze(0)
```

注3:我们在训练过程运用了实体替换、实体掩码、实体拼接三种数据增强策略，改进了模型的性能。下面是在测试集上的F1 Score：

| 模型名称                | 未数据增强 | 数据增强 |
| ----------------------- | ---------- | -------- |
| bert-base-chinese       | 97.13%     | 97.42%   |
| chinese-roberta-wwm-ext | 96.77%     | 97.40%   |

注4：为了使模型的识别结果与知识图谱上的实体名相匹配，我们使用了TF-IDF实体对齐。

## 意图识别

什么是意图识别？

<img src="img/yitushibie.jpg" style="zoom:50%;" />



我们对比了3种意图识别的策略(规则匹配、训练模型、提示工程)：

| 策略     | 准确性 | 多意图识别 | 人工成本     | 推理速度 | 资源消耗 |
| -------- | ------ | ---------- | ------------ | -------- | -------- |
| 规则匹配 | 低     | x          | 低           | 快       | 低       |
| 训练模型 | 高     | x          | 高(数据标注) | 中等     | 中等     |
| 提示工程 | 高     | ✓          | 低           | 慢       | 高       |

综合考虑，我们采用了提示工程的手段：我们将意图分为16种，根据16类意图设计Prompt，让大模型对用户的查询进行意图分析。

注1：我们结合了上下文学习和思维链技术，最终取得了良好的结果。

注2：这部分代码整合到了```webui.py```中，您无需进行任何操作。

## 知识图谱查询

我们为每一个意图，设置了一个查询语句。

<img src="img/yuju.jpg" style="zoom:30%;" />

注：这部分代码整合到了```webui.py```中，您无需进行任何操作。

## 运行界面



我们将意图识别、知识库查询、对话界面都写在了```webui.py```中。2024.5.21，我们为界面增加了登陆、注册界面，设置了用户和管理员两种身份，您可以使用命令启动：

```
streamlit run login.py
```

登陆界面如下图所示：

<img src="img/login.png" style="zoom:100%;" />

注册界面如下图所示：

<img src="img/register.png" style="zoom:100%;" />

管理员登陆界面如下图所示：

<img src="img/admin.png" style="zoom:70%;" />

用户登陆界面如下图所示：

<img src="img/user.png" style="zoom:70%;" />

几个运行例子：

<img src="img/e1.png" style="zoom:40%;" />

<img src="img/e2.png" style="zoom:40%;" />

<img src="img/e3.png" style="zoom:40%;" />

<img src="img/e4.png" style="zoom:40%;" />

<img src="img/e5.png" style="zoom:40%;" />

<img src="img/e6.png" style="zoom:40%;" />

<img src="img/e7.png" style="zoom:40%;" />

## 未来工作

### NL2Cyhper

我们将意图归为16类，已经涵盖了大部分意图，但是无法穷尽所有的意图，无法充分利用知识图谱中的数据。因此，我们尝试进行NL2Cyhper：抛弃实体识别和意图识别两个操作，直接根据用户的问题生成查询语句。

<img src="img/nl2cyhper.jpg" style="zoom:30%;" />

问题：需要人工进行数据标注。

## 联系方式

如果您的复现遇到了困难，请随时联系！

邮箱：zeromakers@outlook.com

