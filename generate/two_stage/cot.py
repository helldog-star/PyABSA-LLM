import random
import time
import openai
import os
import backoff
import json
from tqdm import tqdm
# openai.api_key = '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'
# openai.api_key = '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'
# openai.api_key = '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'
# openai.api_key = "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"
openai.api_key = "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"


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def eval_chatgpt_res_acc(file):
    f = open(file, "r")
    data_list = json.load(f)
    correct_num = 0
    for data in data_list:
        if data["ori_label"] == data["gpt_ori_cot"]:
            correct_num += 1
    acc = correct_num / len(data_list) * 100
    f.close()
    return f"total{len(data_list)}_correct{correct_num}_acc{round(acc, 2)}"


def call_chatgpt(file_path, save_path, prompt_template):
    if not os.path.isdir(save_path):
        return

    f = open(file_path, "r")

    data = json.load(f)
    pbar = tqdm(data)
    res = []
    write_file = save_path + f"/chatgpt_res.json"
    fw = open(write_file, "w")
    for sample in pbar:
        num = 0
        pbar.set_description("Processing")
        term1 = sample["term"]
        sentence1 = sample["ori"]
        label = sample["ori_label"]

        prompt = prompt_template.format(term1,sentence1 )
        for delay_secs in (2**x for x in range(0, 6)):
            try:
                # Call openai request such as text completion
                call_res = completions_with_backoff(model="text-davinci-003",
                                                    prompt=prompt,
                                                    temperature=0.7,
                                                    max_tokens=64,
                                                    top_p=1.0,
                                                    frequency_penalty=0.0,
                                                    presence_penalty=0.0)
                break

            except openai.OpenAIError as e:
                randomness_collision_avoidance = random.randint(0,
                                                                1000) / 1000.0
                sleep_dur = delay_secs + randomness_collision_avoidance
                print(
                    f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                time.sleep(sleep_dur)
                continue
        gpt_res = call_res["choices"][0]["text"].strip().lower()
        label = label.lower()

        # 重复调用直到result在[pos,neg,neu]中，或者模型输出中含有positive或negative或neutral
        while gpt_res not in ["positive", "negative", "neutral"]:
            print(gpt_res)
            num += 1
            if num > 10:
                break
            if "positive" in gpt_res:
                gpt_res = "positive"
            elif "negative" in gpt_res:
                gpt_res = "negative"
            elif "neutral" in gpt_res:
                gpt_res = "neutral"
            else:
                for delay_secs in (2**x for x in range(0, 6)):
                    try:
                        # Call openai request such as text completion
                        call_res = completions_with_backoff(
                            model="text-davinci-003",
                            prompt=prompt,
                            temperature=0.7,
                            max_tokens=64,
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0)
                        break

                    except openai.OpenAIError as e:
                        randomness_collision_avoidance = random.randint(
                            0, 1000) / 1000.0
                        sleep_dur = delay_secs + randomness_collision_avoidance
                        print(
                            f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds."
                        )
                        time.sleep(sleep_dur)
                        continue

                gpt_res = call_res["choices"][0]["text"].strip().lower()
        sample["gpt_ori_cot"] = gpt_res
        res.append(sample)

    json.dump(res, fw, indent=-1)
    fw.close()
    # 评估结果
    eval_res = eval_chatgpt_res_acc(write_file)
    os.rename(save_path + f"/chatgpt_res.json",
              save_path + f"/chatgpt_res_{eval_res}.json")
    f.close()


if __name__ == "__main__":
    # file_path = "/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/aug_absa_data/laptop/Laptops_Test_Gold.xml.seg"
    # file_path = "/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/aug_absa_data/res/Restaurants_Test_Gold.xml.seg"
    # file_path = '/home/liuxinyu/dingyan/ABSAcode/generate/two_stage/match_data/laptop/laptop_addDiff_ori.json'
    # file_path = '/home/liuxinyu/dingyan/ABSAcode/generate/two_stage/match_data/laptop/laptop_revNon_ori.json'
    file_path = '/home/liuxinyu/dingyan/ABSAcode/generate/two_stage/match_data/laptop/test/laptop_addDiff_ori.json'
    file_dir = os.path.abspath(os.path.dirname(file_path))
    if not os.path.exists(file_dir + "/cot"):
        os.mkdir(file_dir + "/cot")
    save_path = file_dir + "/cot"
    # laptop_example = [
    #     "for an appetizer, their calamari is a winner.", "positive",
    #     "calamari",
    #     "for an appetizer, their calamari is a winner, but meals is terrible, green chills is not edible and service is horrible.",
    #     "calamari", "positive"
    # ]
    # laptop_example = [
    #     "the retina display display make pictures i took years ago jaw dropping.",
    #     "retina display display", "positive",
    #     "the retina display display make pictures i took years ago jaw dropping, but ram is a gripe and mac OS is a blame.",
    #     "retina display display", "positive"
    # ]
    laptop_example = [
        "It has all the expected features and more +plus a wide screen and more than roomy keyboard.",
        "keyboard", "positive",
        "It has all the not expected features but more + plus a narrow screen but more than roomy keyboard.",
        "keyboard", "positive"
    ]#revNon
    # laptop_example = [
    #     "3D rendering slows it down considerably.", "3D rendering", "negative",
    #     "3D rendering not slows it down considerably.", "3D rendering",
    #     "positive"
    # ]  #revtgt
    # prompt_template = "In sentence \"{}\", the sentiment polarity of \"{}\" is {}. In sentence \"{}\", the sentiment polarity of \"{}\" is {}.".format(
    #     laptop_example[0], laptop_example[1], laptop_example[2],
    #     laptop_example[3], laptop_example[4], laptop_example[5])
    # prompt_template = prompt_template + " In sentence \"{}\", the sentiment polarity of \"{}\" is {}. In sentence \"{}\", the sentiment polarity of \"{}\" is"
    prompt_template = "Q: What\'s the sentiment polarity of \"keyboard\" in the following sentence? It has all the expected features and more +plus a wide screen and more than roomy keyboard. A: \"keyboard\" in this sentence is described as \"roomy\". So the sentiment polarity of \"keyboard\" is positive. Q: What\'s the sentiment polarity of \"{}\" in the following sentence? {} A:"
    # prompt_template = "sentence 1:{} term1: {} sentiment polarity1: {}. sentence 2:{} term2: {} The sentiment polarity of term1 and term2 is same. So sentiment polarity2: {}. ".format(
    #     laptop_example[0], laptop_example[1], laptop_example[2],
    #      laptop_example[3], laptop_example[4], laptop_example[5])
    # prompt_template = prompt_template + " sentence 1:{} term1: {} sentiment polarity1: {}. sentence 2:{} term2: {} sentiment polarity2: "

    for delay_secs in (2**x for x in range(0, 6)):
        try:
            # Call openai request such as text completion
            call_chatgpt(file_path, save_path, prompt_template)
            break

        except openai.OpenAIError as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue
