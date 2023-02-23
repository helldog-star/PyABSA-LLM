import random
import time
import openai
import os
import backoff
from typing import Union
import re

api_key_list = ["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"]
openai.api_key = api_key_list[0]

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def __completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def call_text003_api(prompt) -> str:
    for delay_secs in (2**x for x in range(0, 6)):
        try:
            # Call openai request such as text completion
            call_res = __completions_with_backoff(
                        model="text-davinci-003",
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=64,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0)
            break   
        except openai.OpenAIError as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            # 如果账户没钱了自动续上下一个，全没钱了就退出
            pattern = r"You exceeded your current quota, please check your plan and billing details"
            if re.search(pattern, e.args[0]):
                api_key_list.pop(0)
                if len(api_key_list) > 0:
                    openai.api_key = api_key_list[0]
                else:
                    print("it's time to pay :(")
                    break
            time.sleep(sleep_dur)
            continue
    return call_res["choices"][0]["text"].strip().lower()

def process_chatgpt_response_into_sentiment_label(text) -> Union[str, None]:
    text = text.lower()
    pattern = r"positive|negative|neutral"
    # 获得第一个匹配的pattern
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        print("chatgpt response: " + text)
        return None

def calculate_acc_for_chatgpt_classify(result_file):
    '''
    此方法用于计算chatgpt response的准确率,并为结果文件重新命名
    result_file(.xml.seg)中数据格式
    sentence termb label chatgpt_response_label
    '''
    with open(result_file, "r") as r_f:
        data = r_f.readlines()
        # 删除空行
        clean_data = []
        for tmp in data:
            if tmp.strip(): clean_data.append(tmp.strip())
        # 判断数据量是否能被4整除
        assert len(clean_data) % 4 == 0
        i = 0
        correct_num = 0
        total_num = len(clean_data) // 4

        while i + 3 < len(clean_data):
            if clean_data[i + 2].lower() == clean_data[i + 3].lower():
                correct_num += 1
            i += 4
        acc = correct_num / total_num * 100

    new_file_name = os.path.dirname(result_file) + f"/total{total_num}_correct{correct_num}_acc{round(acc, 2)}.xml.seg"
    # 判断文件名是否已存在
    if os.path.isfile(new_file_name):
        file_exists = True
        index = 1
        while file_exists:
            base, extension = os.path.splitext(new_file_name)
            new_file_name = f"{base}_{index}{extension}"
            if os.path.isfile(new_file_name):
                index += 1
            else:
                file_exists = False   
    os.rename(result_file, new_file_name)

def process_xml_file_data(data_file) -> list:
    '''
    讲xml数据文件处理成list[tuple]并返回
    '''
    with open(data_file, "r") as r_f:
        data = r_f.readlines()
        i = 0
        data_group = []
        while i + 2 < len(data):
            data_group.append((data[i].strip(), data[i + 1].strip(), data[i + 2].strip()))
            i += 3
        return data_group
