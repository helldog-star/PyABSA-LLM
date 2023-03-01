from utiles import process_chatgpt_response_into_sentiment_label, process_xml_file_data, call_text003_api, calculate_acc_for_chatgpt_classify
import os
from tqdm import tqdm
import json
# 可以多试试
cot_demonstration = (
    "Q: What\'s the sentiment polarity of \"keyboard\" in the following sentence?"
    " \"It has all the expected features and more +plus a wide screen and more than roomy keyboard.\""
    " A: \"keyboard\" in this sentence is described by \"roomy\". So the sentiment polarity of \"keyboard\" is positive."
    " Q: What\'s the sentiment polarity of \"{}\" in the following sentence? \"{}\" A:"
)

# 后面可以考虑用一个dict去映射所有路径 name: path
dataset_name = "laptop14"
dataset_path = "/home/dingyan/ABSA/PyABSA-LLM/integrated_datasets/apc_datasets/110.SemEval/113.laptop14/Laptops_Test_Gold.xml.seg"

data_group = process_xml_file_data(dataset_path)

# 判断文件名对应的实验文件夹是否存在,若不存在则创建
experiment_path = f"/home/dingyan/ABSA/PyABSA-LLM/experiments/chatgpt_aug/{os.path.basename(__file__).split('.')[0]}"
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)
# 判断数据集对应的文件夹是否存在,若不存在则创建
save_path = f"{experiment_path}/{dataset_name}"
if not os.path.exists(save_path):
    os.mkdir(save_path)

all_sentence=[]
# 意外中断程序可继续写入，防止浪费
# 以读写的方式打开文件，首先检查是否已经处理过一部分
with open(save_path + "/aug.json", "r+") as r_w_f:
    processed_data=json.load(r_w_f)
    # 生成可以唯一标识数据的三元组列表(sentence,term,label)
    check_processed_list = []
    processed_data.pop()
    for sen in processed_data:
        check_processed_list.append((sen["sentence"],sen["term"],sen["label"]))

    # 生成chatgpt回复
    pbar = tqdm(data_group)
    num=0
    pbar.set_description("Processing")
    for data in pbar:
        dic={}
        term = data[1]
        sentence = data[0].replace("$T$", term, 1)
        label = data[2]
        # 若生成过则跳过这条数据
        if (sentence, term, label) in check_processed_list:
            all_sentence.append(processed_data[num])
            num+=1
            continue
        all_labels=["positive","negative","neutral"]
        # 调用api
        #prompt1:更改目标方面词情感极性
        dic["sentence"]=sentence
        dic["term"]=term
        dic["label"]=label
        all_labels.remove(label.lower())
        prompt1_dic={}
        for another_label in all_labels:
            prompt1 = "\"{}\" Make the sentiment description of \"{}\" \"{}\" with minimal changes and keep \"{}\" in the sentence.".format(sentence,term,another_label,term)
            response =call_text003_api(prompt1)
            prompt1_dic[another_label]=response
        dic["prompt1"] = prompt1_dic


        #prompt2:显式隐式
        prompt2_dic={}
        for type in ["implicitly","explicitly"]:
            prompt2 = "\"{}\" \"{}\" express the \"{}\" sentiment of \"{}\" with minimal changes and keep \"{}\" in the sentence.".format(sentence,type,label,term,term)
            prompt2_dic[type] = call_text003_api(prompt2)
        dic["prompt2"]=prompt2_dic
        #重写背景
        prompt3_dic={}
        prompt3 = "\"{}\" Rewrite the sentence with the same semantics and maximal changes, keep \"{}\"  in the sentence.".format(sentence,term)
        prompt3_dic[label]=call_text003_api(prompt3)
        dic["prompt3"]=prompt3_dic
        num+=1
        all_sentence.append(dic)

    r_w_f.seek(0)
    r_w_f.truncate()
    json.dump(fp=r_w_f,obj=all_sentence,indent=-1)

# 计算结果并重命名
# calculate_acc_for_chatgpt_classify(save_path + "/tmp_result.xml.seg")
