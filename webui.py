import os
import streamlit as st
import ner_model as zwk
import pickle
import ollama
from transformers import BertTokenizer
import torch
import py2neo
import random
import re



@st.cache_resource
def load_model(cache_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #加载ChatGLM模型
    # glm_tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b-128k", trust_remote_code=True)
    # glm_model = AutoModel.from_pretrained("model/chatglm3-6b-128k",trust_remote_code=True,device=device)
    # glm_model.eval()
    glm_model = None
    glm_tokenizer= None
    #加载Bert模型
    with open('tmp_data/tag2idx.npy', 'rb') as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)
    rule = zwk.rule_find()
    tfidf_r = zwk.tfidf_alignment()
    model_name = 'model/chinese-roberta-wwm-ext'
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = zwk.Bert_Model(model_name, hidden_size=128, tag_num=len(tag2idx), bi=True)
    bert_model.load_state_dict(torch.load(f'model/{cache_model}.pt'))
    
    bert_model = bert_model.to(device)
    bert_model.eval()
    return glm_tokenizer,glm_model,bert_tokenizer,bert_model,idx2tag,rule,tfidf_r,device



def Intent_Recognition(query,choice):
    prompt = f"""
阅读下列提示，回答问题（问题在输入的最后）:
当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在16个预定义的查询类别中一一进行判断。对于每一个类别，思考用户的问题是否含有与该类别对应的意图。如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。这样的方法要求你对每一个可能的查询意图进行系统性的考虑和评估，确保没有遗漏任何一个可能的分类。

**查询类别**
- "查询疾病简介"
- "查询疾病病因"
- "查询疾病预防措施"
- "查询疾病治疗周期"
- "查询治愈概率"
- "查询疾病易感人群"
- "查询疾病所需药品"
- "查询疾病宜吃食物"
- "查询疾病忌吃食物"
- "查询疾病所需检查项目"
- "查询疾病所属科目"
- "查询疾病的症状"
- "查询疾病的治疗方法"
- "查询疾病的并发疾病"
- "查询药品的生产商"

在处理用户的问题时，请按照以下步骤操作：
- 仔细阅读用户的问题。
- 对照上述查询类别列表，依次考虑每个类别是否与用户问题相关。
- 如果用户问题明确或隐含地包含了某个类别的查询意图，请将该类别的描述添加到输出列表中。
- 确保最终的输出列表包含了所有与用户问题相关的类别描述。

以下是一些含有隐晦性意图的例子，每个例子都采用了输入和输出格式，并包含了对你进行思维链形成的提示：
**示例1：**
输入："睡眠不好，这是为什么？"
输出：["查询疾病简介","查询疾病病因"]  # 这个问题隐含地询问了睡眠不好的病因
**示例2：**
输入："感冒了，怎么办才好？"
输出：["查询疾病简介","查询疾病所需药品", "查询疾病的治疗方法"]  # 用户可能既想知道应该吃哪些药品，也想了解治疗方法
**示例3：**
输入："跑步后膝盖痛，需要吃点什么？"
输出：["查询疾病简介","查询疾病宜吃食物", "查询疾病所需药品"]  # 这个问题可能既询问宜吃的食物，也可能在询问所需药品
**示例4：**
输入："我怎样才能避免冬天的流感和感冒？"
输出：["查询疾病简介","查询疾病预防措施"]  # 询问的是预防措施，但因为提到了两种疾病，这里隐含的是对共同预防措施的询问
**示例5：**
输入："头疼是什么原因，应该怎么办？"
输出：["查询疾病简介","查询疾病病因", "查询疾病的治疗方法"]  # 用户询问的是头疼的病因和治疗方法
**示例6：**
输入："如何知道自己是不是有艾滋病？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有艾滋病，一定一定要进行相关检查，这是根本性的！其次是查看疾病的病因，看看自己的行为是不是和病因重合。
**示例7：**
输入："我该怎么知道我自己是否得了21三体综合症呢？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有21三体综合症，一定一定要进行相关检查(比如染色体)，这是根本性的！其次是查看疾病的病因。
**示例8：**
输入："感冒了，怎么办？"
输出：["查询疾病简介","查询疾病的治疗方法","查询疾病所需药品","查询疾病所需检查项目","查询疾病宜吃食物"]  # 问怎么办，首选治疗方法。然后是要给用户推荐一些药，最后让他检查一下身体。同时，也推荐一下食物。
**示例9：**
输入："癌症会引发其他疾病吗？"
输出：["查询疾病简介","查询疾病的并发疾病","查询疾病简介"]  # 显然，用户问的是疾病并发疾病，随后可以给用户科普一下癌症简介。
**示例10：**
输入："葡萄糖浆的生产者是谁？葡萄糖浆是谁生产的？"
输出：["查询药品的生产商"]  # 显然，用户想要问药品的生产商
通过上述例子，我们希望你能够形成一套系统的思考过程，以准确识别出用户问题中的所有可能查询意图。请仔细分析用户的问题，考虑到其可能的多重含义，确保输出反映了所有相关的查询意图。

**注意：**
- 你的所有输出，都必须在这个范围内上述**查询类别**范围内，不可创造新的名词与类别！
- 参考上述5个示例：在输出查询意图对应的列表之后，请紧跟着用"#"号开始的注释，简短地解释为什么选择这些意图选项。注释应当直接跟在列表后面，形成一条连续的输出。
- 你的输出的类别数量不应该超过5，如果确实有很多个，请你输出最有可能的5个！同时，你的解释不宜过长，但是得富有条理性。

现在，你已经知道如何解决问题了，请你解决下面这个问题并将结果输出！
问题输入："{query}"
输出的时候请确保输出内容都在**查询类别**中出现过。确保输出类别个数**不要超过5个**！确保你的解释和合乎逻辑的！注意，如果用户询问了有关疾病的问题，一般都要先介绍一下疾病，也就是有"查询疾病简介"这个需求。
再次检查你的输出都包含在**查询类别**:"查询疾病简介"、"查询疾病病因"、"查询疾病预防措施"、"查询疾病治疗周期"、"查询治愈概率"、"查询疾病易感人群"、"查询疾病所需药品"、"查询疾病宜吃食物"、"查询疾病忌吃食物"、"查询疾病所需检查项目"、"查询疾病所属科目"、"查询疾病的症状"、"查询疾病的治疗方法"、"查询疾病的并发疾病"、"查询药品的生产商"。
"""
    rec_result = ollama.generate(model=choice, prompt=prompt)['response']
    print(f'意图识别结果:{rec_result}')
    return rec_result
    # response, _ = glm_model.chat(glm_tokenizer, prompt, history=[])
    # return response


def add_shuxing_prompt(entity,shuxing,client):
    add_prompt = ""
    try:
        sql_q = "match (a:疾病{名称:'%s'}) return a.%s" % (entity,shuxing)
        res = client.run(sql_q).data()[0].values()
        add_prompt+=f"<提示>"
        add_prompt+=f"用户对{entity}可能有查询{shuxing}需求，知识库内容如下："
        if len(res)>0:
            join_res = "".join(res)
            add_prompt+=join_res
        else:
            add_prompt+="图谱中无信息，查找失败。"
        add_prompt+=f"</提示>"
    except:
        pass
    return add_prompt
def add_lianxi_prompt(entity,lianxi,target,client):
    add_prompt = ""
    
    try:
        sql_q = "match (a:疾病{名称:'%s'})-[r:%s]->(b:%s) return b.名称" % (entity,lianxi,target)
        res = client.run(sql_q).data()#[0].values()
        res = [list(data.values())[0] for data in res]
        add_prompt+=f"<提示>"
        add_prompt+=f"用户对{entity}可能有查询{lianxi}需求，知识库内容如下："
        if len(res)>0:
            join_res = "、".join(res)
            add_prompt+=join_res
        else:
            add_prompt+="图谱中无信息，查找失败。"
        add_prompt+=f"</提示>"
    except:
        pass
    return add_prompt
def generate_prompt(response,query,client,bert_model, bert_tokenizer,rule, tfidf_r, device, idx2tag):
    entities = zwk.get_ner_result(bert_model, bert_tokenizer, query, rule, tfidf_r, device, idx2tag)
    # print(response)
    # print(entities)
    yitu = []
    prompt = "<指令>你是一个医疗问答机器人，你需要根据给定的提示回答用户的问题。请注意，你的全部回答必须完全基于给定的提示，不可自由发挥。如果根据提示无法给出答案，立刻回答“根据已知信息无法回答该问题”。</指令>"
    prompt +="<指令>请你仅针对医疗类问题提供简洁和专业的回答。如果问题不是医疗相关的，你一定要回答“我只能回答医疗相关的问题。”，以明确告知你的回答限制。</指令>"
    if '疾病症状' in entities and  '疾病' not in entities:
        sql_q = "match (a:疾病)-[r:疾病的症状]->(b:疾病症状 {名称:'%s'}) return a.名称" % (entities['疾病症状'])
        res = list(client.run(sql_q).data()[0].values())
        # print('res=',res)
        if len(res)>0:
            entities['疾病'] = random.choice(res)
            all_en = "、".join(res)
            prompt+=f"<提示>用户有{entities['疾病症状']}的情况，知识库推测其可能是得了{all_en}。请注意这只是一个推测，你需要明确告知用户这一点。</提示>"
    pre_len = len(prompt)
    if "简介" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'疾病简介',client)
            yitu.append('查询疾病简介')
    if "病因" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'疾病病因',client)
            yitu.append('查询疾病病因')
    if "预防" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'预防措施',client)
            yitu.append('查询预防措施')
    if "治疗周期" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'治疗周期',client)
            yitu.append('查询治疗周期')
    if "治愈概率" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'治愈概率',client)
            yitu.append('查询治愈概率')
    if "易感人群" in response:
        if '疾病' in entities:
            prompt+=add_shuxing_prompt(entities['疾病'],'疾病易感人群',client)
            yitu.append('查询疾病易感人群')
    if "药品" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病使用药品','药品',client)
            yitu.append('查询疾病使用药品')
    if "宜吃食物" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病宜吃食物','食物',client)
            yitu.append('查询疾病宜吃食物')
    if "忌吃食物" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病忌吃食物','食物',client)
            yitu.append('查询疾病忌吃食物')
    if "检查项目" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病所需检查','检查项目',client)
            yitu.append('查询疾病所需检查')
    if "查询疾病所属科目" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病所属科目','科目',client)
            yitu.append('查询疾病所属科目')
    # if "所属科目" in response:
    #     if '疾病' in entities:
    #         prompt+=add_lianxi_prompt(entities['疾病'],'疾病所属科目','科目')
    #         yitu.append('查询疾病所属科目')
    if "症状" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病的症状','疾病症状',client)
            yitu.append('查询疾病的症状')
    if "治疗" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'治疗的方法','治疗方法',client)
            yitu.append('查询治疗的方法')
    if "并发" in response:
        if '疾病' in entities:
            prompt+=add_lianxi_prompt(entities['疾病'],'疾病并发疾病','疾病',client)
            yitu.append('查询疾病并发疾病')
    if "生产商" in response:
        try:
            sql_q = "match (a:药品商)-[r:生产]->(b:药品{名称:'%s'}) return a.名称" % (entities['药品'])
            res = client.run(sql_q).data()[0].values()
            prompt+=f"<提示>"
            prompt+=f"用户对{entities['药品']}可能有查询药品生产商的需求，知识图谱内容如下："
            if len(res)>0:
                prompt+="".join(res)
            else:
                prompt+="图谱中无信息，查找失败"
            prompt+=f"</提示>"
        except:
            pass
        yitu.append('查询药物生产商')
    if pre_len==len(prompt) :
        prompt += f"<提示>提示：知识库异常，没有相关信息！请你直接回答“根据已知信息无法回答该问题”！</提示>"
    prompt += f"<用户问题>{query}</用户问题>"
    prompt += f"<注意>现在你已经知道给定的“<提示></提示>”和“<用户问题></用户问题>”了,你要极其认真的判断提示里是否有用户问题所需的信息，如果没有相关信息，你必须直接回答“根据已知信息无法回答该问题”。</注意>"

    prompt += f"<注意>你一定要再次检查你的回答是否完全基于“<提示></提示>”的内容，不可产生提示之外的答案！换而言之，你的任务是根据用户的问题，将“<提示></提示>”整理成有条理、有逻辑的语句。你起到的作用仅仅是整合提示的功能，你一定不可以利用自身已经存在的知识进行回答，你必须从提示中找到问题的答案！</注意>"
    prompt += f"<注意>你必须充分的利用提示中的知识，不可将提示中的任何信息遗漏，你必须做到对提示信息的充分整合。你回答的任何一句话必须在提示中有所体现！如果根据提示无法给出答案，你必须回答“根据已知信息无法回答该问题”。<注意>"
    
    
    print(f'prompt:{prompt}')
    return prompt,"、".join(yitu),entities



def ans_stream(prompt):
    
    result = ""
    for res,his in glm_model.stream_chat(glm_tokenizer, prompt, history=[]):
        yield res



def main(is_admin, usname):
    cache_model = 'best_roberta_rnn_model_ent_aug'
    st.title(f"医疗智能问答机器人")

    with st.sidebar:
        col1, col2 = st.columns([0.6, 0.6])
        with col1:
            st.image(os.path.join("img", "logo.jpg"), use_column_width=True)

        st.caption(
            f"""<p align="left">欢迎您，{'管理员' if is_admin else '用户'}{usname}！当前版本：{1.0}</p>""",
            unsafe_allow_html=True,
        )

        if 'chat_windows' not in st.session_state:
            st.session_state.chat_windows = [[]]
            st.session_state.messages = [[]]

        if st.button('新建对话窗口'):
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])

        window_options = [f"对话窗口 {i + 1}" for i in range(len(st.session_state.chat_windows))]
        selected_window = st.selectbox('请选择对话窗口:', window_options)
        active_window_index = int(selected_window.split()[1]) - 1

        selected_option = st.selectbox(
            label='请选择大语言模型:',
            options=['Qwen 1.5', 'Llama2-Chinese']
        )
        choice = 'qwen:32b' if selected_option == 'Qwen 1.5' else 'llama2-chinese:13b-chat-q8_0'

        show_ent = show_int = show_prompt = False
        if is_admin:
            show_ent = st.sidebar.checkbox("显示实体识别结果")
            show_int = st.sidebar.checkbox("显示意图识别结果")
            show_prompt = st.sidebar.checkbox("显示查询的知识库信息")
            if st.button('修改知识图谱'):
            # 显示一个链接，用户可以点击这个链接在新标签页中打开百度
                st.markdown('[点击这里修改知识图谱](http://127.0.0.1:7474/)', unsafe_allow_html=True)



        if st.button("返回登录"):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.experimental_rerun()

    glm_tokenizer, glm_model, bert_tokenizer, bert_model, idx2tag, rule, tfidf_r, device = load_model(cache_model)
    client = py2neo.Graph('http://localhost:7474', user='neo4j', password='wei8kang7.long', name='neo4j')

    current_messages = st.session_state.messages[active_window_index]

    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if show_ent:
                    with st.expander("实体识别结果"):
                        st.write(message.get("ent", ""))
                if show_int:
                    with st.expander("意图识别结果"):
                        st.write(message.get("yitu", ""))
                if show_prompt:
                    with st.expander("点击显示知识库信息"):
                        st.write(message.get("prompt", ""))

    if query := st.chat_input("Ask me anything!", key=f"chat_input_{active_window_index}"):
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        response_placeholder = st.empty()
        response_placeholder.text("正在进行意图识别...")

        query = current_messages[-1]["content"]
        response = Intent_Recognition(query, choice)
        response_placeholder.empty()

        prompt, yitu, entities = generate_prompt(response, query, client, bert_model, bert_tokenizer, rule, tfidf_r, device, idx2tag)

        last = ""
        for chunk in ollama.chat(model=choice, messages=[{'role': 'user', 'content': prompt}], stream=True):
            last += chunk['message']['content']
            response_placeholder.markdown(last)
        response_placeholder.markdown("")

        knowledge = re.findall(r'<提示>(.*?)</提示>', prompt)
        zhishiku_content = "\n".join([f"提示{idx + 1}, {kn}" for idx, kn in enumerate(knowledge) if len(kn) >= 3])
        with st.chat_message("assistant"):
            st.markdown(last)
            if show_ent:
                with st.expander("实体识别结果"):
                    st.write(str(entities))
            if show_int:
                with st.expander("意图识别结果"):
                    st.write(yitu)
            if show_prompt:
                
                
                with st.expander("点击显示知识库信息"):
                    st.write(zhishiku_content)
        current_messages.append({"role": "assistant", "content": last, "yitu": yitu, "prompt": zhishiku_content, "ent": str(entities)})


    st.session_state.messages[active_window_index] = current_messages
