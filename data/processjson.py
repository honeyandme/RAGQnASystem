import os
from tqdm import tqdm
import json
import ollama

def get_cure_way_result(problem):
    prompt = f"""
请根据任务说明和两个实例,模仿着解决"问题输入"中的问题。
任务说明：
对于给定的文本，识别并提取所有提及的“治疗方法”实体。这包括具体的治疗措施、药物名称、手术方法等。

示例1：
输入: "消融"
输出: ['消融']
示例2：
输入: "药物治疗如黏膜保护剂、抑酸剂，有幽门螺杆菌感染者进行根除治疗"
输出: ['药物治疗','黏膜保护剂','抑酸剂','根除治疗']
示例3：
输入: "药物治疗包括抑酸剂、生长抑素等，胃镜下止血治疗，介入手术及外科手术治疗"
输出: ['药物治疗','抑酸剂','生长抑素','胃镜下止血治疗','介入手术','外科手术治疗']
示例4：
输入: "调整饮食、补充优质蛋白质为主；同时积极去除病因，积极防治并发症"
输出: ['调整饮食', '补充优质蛋白质', '防治并发症']
示例5：
输入: "药物治疗包括口服碳酸氢钠、抑酸剂及支持性治疗，药物治疗无效者可在胃镜下碎石或取出"
输出: ['药物治疗', '口服碳酸氢钠', '口服抑酸剂', '支持性治疗', '胃镜下碎石', '胃镜下取出']
示例6：
输入: "主要是休息，给予低蛋白、低盐、适量脂肪饮食，利尿、降血压等对症治疗"
输出: ['休息', '低蛋白饮食', '低盐饮食', '适量脂肪饮食', '利尿', '降血压','对症治疗']
示例7：
输入: "药物治疗如保护胃肠黏膜、抑酸剂、抗生素、补液等"
输出: ['药物治疗','保护胃肠黏膜', '抑酸剂', '抗生素', '补液']


注意：
- 不要对问题输入进行改写和扩写，问题输入可能就是个病句,但是没关系,你只需要从里面提取治疗方法的实体！
- 请严格参考示例的输出格式,这类似于一个一维python的列表。直接输出符合该格式的答案，无需包含任何额外文本。

问题输入: "{problem}"

【仅输出如下格式的答案，不要包含任何其他文本或说明。】
    """
    response = ollama.generate(model='qwen:32b', prompt=prompt)['response']
    # print(response)
    return eval(response)

def get_drug_detail_result(problem):
    prompt = f"""
请根据任务说明和两个实例,模仿着解决"问题输入"中的问题。
任务说明：
对于给定的文本，识别并提取具体的药物名称和药品商名称,并输出。中间用逗号(英文)分隔，如果不知道厂商，请输出“未知”。

示例1：
输入: "惠普森穿心莲内酯片(穿心莲内酯片)"
输出: 惠普森穿心莲内酯片,惠普森
示例2：
输入: "博美欣(阿莫西林克拉维酸钾(4:1)分散片)"
输出: 阿莫西林克拉维酸钾分散片,博美欣
示例3：
输入: "标准桃金娘油肠溶胶囊(成人装(标准桃金娘油肠溶胶囊(成人装))"
输出: 标准桃金娘油肠溶胶囊,未知
示例4：
输入: "士强(肠内营养混悬液(TPSPA))"
输出: 肠内营养混悬液,士强
示例5：
输入: "北京同仁堂百咳静糖浆(百咳静糖浆)"
输出: 北京同仁堂百咳静糖浆,北京同仁堂
示例6：
输入: "迪克乐克(双氯芬酸钠缓释片(IV))"
输出: 双氯芬酸钠缓释片(IV),迪克乐克
示例7：
输入: "海斯制药注射用盐酸溴己新(注射用盐酸溴己新)"
输出: 海斯制药注射用盐酸溴己新,海斯制药
示例8：
输入: "注射用甲泼尼龙琥珀酸钠"
输出: 注射用甲泼尼龙琥珀酸钠,未知

注意：
- 需要精确地提取药物名称和药品商名称,去除其他一切无关信息。
- 输入的内容可能有格式问题，你需要调整好格式，忽略无关的信息。
- 请严格参考示例的输出格式,无需包含任何额外文本。你只需要输出药物名称、逗号(,)、药品商名称!

问题输入: "{problem}"
任务说明：
对于给定的文本，识别并提取具体的药物名称和药品商名称,并输出。中间用逗号(英文)分隔，如果不知道厂商，请输出“未知”。
【仅输出如下格式的答案，不要包含任何其他文本或说明。】
"""
    response = ollama.generate(model='qwen:32b', prompt=prompt)['response']
    # print(response)
    return response
    response = ollama.generate(model='qwen:32b', prompt=prompt)['response']
    # print(response)
    return response

with open('./data/medical.json','r',encoding='utf-8') as f:
    all_data = f.read().split('\n')
    for i,data in enumerate(tqdm(all_data[2855:])):
        if (len(data) < 3):
            continue
        data = eval(data)

        
        if 'cure_way' in data:
            cure = data['cure_way']
            cure_way_all = []
            for x in cure:
                try:
                    modified_way = get_cure_way_result(x)
                    cure_way_all.extend(modified_way)
                except:
                    cure_way_all.extend([x])
            data['cure_way'] = cure_way_all

        drug_detail = data.get("drug_detail",[])
        if drug_detail:
            for t,detail in enumerate(drug_detail):
                drug_detail[t] = get_drug_detail_result(drug_detail[t])
            data['drug_detail'] = drug_detail
                
        
        json_str = json.dumps(data,ensure_ascii=False)
        
        with open('./data/medical_new_2.json', 'a',encoding='utf8') as f:
            # 确保内容从新的一行开始
            f.write(json_str+',\n')