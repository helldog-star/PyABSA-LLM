from utiles import process_chatgpt_response_into_sentiment_label, process_xml_file_data, call_text003_api, calculate_acc_for_chatgpt_classify
import os
from tqdm import tqdm

# 可以多试试
cot_demonstration = ("Q: What\'s the sentiment polarity of \"keyboard\" in the following sentence?"
" \"It has all the expected features and more +plus a wide screen and more than roomy keyboard.\""
" A: \"keyboard\" in this sentence is described by \"roomy\". So the sentiment polarity of \"keyboard\" is positive."
" Q: What\'s the sentiment polarity of \"{}\" in the following sentence? \"{}\" A:")

# 后面可以考虑用一个dict去映射所有路径 name: path
dataset_name = "laptop14"
dataset_path = "/home/liuxinyu/PyABSA-LLM/integrated_datasets/apc_datasets/110.SemEval/113.laptop14/Laptops_Test_Gold.xml.seg"

data_group = process_xml_file_data(dataset_path)

# 判断文件名对应的实验文件夹是否存在,若不存在则创建
experiment_path = f"/home/liuxinyu/PyABSA-LLM/experiments/{os.path.basename(__file__).split('.')[0]}"
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)
# 判断数据集对应的文件夹是否存在,若不存在则创建
save_path = f"{experiment_path}/{dataset_name}"
if not os.path.exists(save_path):
    os.mkdir(save_path)

# 意外中断程序可继续写入，防止浪费
# 以读写的方式打开文件，首先检查是否已经处理过一部分
with open(save_path + "/tmp_result.xml.seg", "a+") as r_w_f:
    r_w_f.seek(0)
    processed_data = r_w_f.readlines()
    # 清除空行和换行符
    clean_processed_data = []
    for tmp in processed_data: 
        if tmp.strip(): clean_processed_data.append(tmp.strip())
    # 判断数据量是否能被4整除
    assert len(clean_processed_data) % 4 == 0
    # 打印处理过的数据
    if len(clean_processed_data) > 0:
        print("Processed: ")
        print(clean_processed_data)
    # 生成可以唯一标识数据的三元组列表(sentence,term,label)
    check_processed_list = []
    i = 0
    while i + 3 < len(clean_processed_data):
        check_processed_list.append((clean_processed_data[i], clean_processed_data[i+1], clean_processed_data[i+2]))
        i += 4
    
    # 生成chatgpt回复
    pbar = tqdm(data_group)
    pbar.set_description("Processing")
    for data in pbar:
        term = data[1]
        sentence = data[0].replace("$T$", term, 1)
        label = data[2]
        # 若生成过则跳过这条数据
        if (sentence, term, label) in check_processed_list:
            continue
        
        # 调用api
        prompt = cot_demonstration.format(term, sentence)
        response = process_chatgpt_response_into_sentiment_label(call_text003_api(prompt))
        # 如果生成结果中没有label则反复调用
        while not response:
            response = process_chatgpt_response_into_sentiment_label(call_text003_api(prompt))

        # 得到一条结果就写入一条
        r_w_f.write(sentence + "\n" + term + "\n" + label + "\n" + response + "\n")

# 计算结果并重命名    
calculate_acc_for_chatgpt_classify(save_path + "/tmp_result.xml.seg")
