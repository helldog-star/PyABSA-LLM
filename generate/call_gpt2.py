import random
import time
import openai
import os
import backoff
import argparse
import json
from tqdm import tqdm
# openai.api_key = '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'
openai.api_key = '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'


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def eval_chatgpt_res_acc(file):
    f = open(file, "r")
    data_list = json.load(f)
    correct_num = 0
    for data in data_list:
        if data["gold"] == data["chat_gpt"]:
            correct_num += 1
    acc = correct_num / len(data_list) * 100
    f.close()
    return f"total{len(data_list)}_correct{correct_num}_acc{round(acc, 2)}"


# 自动跑三次chatgpt
def call_chatgpt(sens, save_path, promptid,datatype):

    pbar = tqdm(sens)
    res = []
    write_file = save_path + f"/chatgpt_{datatype}.json"
    fw = open(write_file, "w")
    for sample in pbar:
        pbar.set_description("Processing")
        term = sample[1]
        sentence = sample[0]
        label = sample[2]
        if promptid==1:
            prompt_template = "{} What is the sentiment polarity of the target aspect \"{}\" in the above sentence? positive, negative or neutral?"
            prompt = prompt_template.format(sentence, term)
        elif promptid == 2:
            prompt_template = "What is the sentiment polarity of the target aspect \"{}\" in the following sentence? positive, negative or neutral? {}"
            prompt = prompt_template.format(term,sentence)
        call_res = completions_with_backoff(model="text-davinci-003",
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=64,
                                            top_p=1.0,
                                            frequency_penalty=0.0,
                                            presence_penalty=0.0,
                                            stop=".")
        gpt_res = call_res["choices"][0]["text"].strip().lower()
        label = label.lower()
        num=0
        # 重复调用直到result在[pos,neg,neu]中，或者模型输出中含有positive或negative或neutral
        while gpt_res not in ["positive", "negative", "neutral"]:
            print(gpt_res)
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
                        call_res = completions_with_backoff(model="text-davinci-003",
                                                    prompt=prompt,
                                                    temperature=0.7,
                                                    max_tokens=64,
                                                    top_p=1.0,
                                                    frequency_penalty=0.0,
                                                    presence_penalty=0.0,
                                                    stop=".")
                        break

                    except openai.OpenAIError as e:
                        randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                        sleep_dur = delay_secs + randomness_collision_avoidance
                        print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                        time.sleep(sleep_dur)
                        continue

                gpt_res = call_res["choices"][0]["text"].strip().lower()
            num+=1
            if num>=10:
                break
        res.append({
            "sentence": sentence,
            "aspect": term,
            "gold": label,
            "chat_gpt": gpt_res
        })

    json.dump(res, fw)
    fw.close()
    # 评估结果
    eval_res = eval_chatgpt_res_acc(write_file)
    os.rename(save_path + f"/chatgpt_{datatype}.json",
              save_path + f"/chatgpt_{datatype}_{eval_res}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--key', default="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",type=str,help="openai_key")
    parser.add_argument(
        '--data_path',
        type=str,
        default=
        "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_lap_test_prompt2.json"
    )
    parser.add_argument('--prompt',type=int,default=1)
    parser.add_argument('--type',type=str,default="lap")
    args=parser.parse_args()
    openai.api_key = args.key
    file_path = args.data_path
    prompt=args.prompt
    data_type=args.type
    file_dir = os.path.abspath(os.path.dirname(file_path))
    if not os.path.exists(file_dir + "/noprompt"+str(prompt)):
        os.mkdir(file_dir + "/noprompt"+str(prompt))
    save_path = file_dir + "/noprompt"+str(prompt)


    load_dict=json.load(open(file_path,'r'))
    sens=[]
    for sen in load_dict:
        # sens.append([sen["sentence"]+" "+sen["gpt3_sentence"].strip()," ".join(sen["aspects"][0]["term"]),sen["aspects"][0]["polarity"]])
        sens.append([
            sen["sentence"],
            " ".join(sen["aspects"][0]["term"]), sen["aspects"][0]["polarity"]
        ])
    for delay_secs in (2**x for x in range(0, 6)):
        try:
            # Call openai request such as text completion
            call_chatgpt(sens,save_path, prompt,data_type)
            break

        except openai.OpenAIError as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue
