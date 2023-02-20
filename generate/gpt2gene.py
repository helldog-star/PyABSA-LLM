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
        if data["gold"] == data["chat_gpt"]:
            correct_num += 1
    acc = correct_num / len(data_list) * 100
    f.close()
    return f"total{len(data_list)}_correct{correct_num}_acc{round(acc, 2)}"


def call_chatgpt(file_path, save_path, prompt_template):

    f = open(file_path, "r")
    lines = f.readlines()

    def process_data_group(data: list):
        i = 0
        data_group = []
        while i + 2 < len(data):
            data_group.append(
                [data[i].strip(), data[i + 1].strip(), data[i + 2].strip()])
            i += 3
        return data_group

    data = process_data_group(lines)
    pbar = tqdm(data)
    res = []
    write_file = save_path
    fw = open(write_file, "w")
    for sample in pbar:
        num = 0
        pbar.set_description("Processing")
        term = sample[1]
        sentence = sample[0].replace("$T$", term, 1)
        label = sample[2]

        prompt = prompt_template.format(term, sentence)
        for delay_secs in (2**x for x in range(0, 6)):
            try:
                # Call openai request such as text completion
                call_res = completions_with_backoff(model="text-davinci-003",
                                                    prompt=prompt,
                                                    temperature=0.7,
                                                    max_tokens=64,
                                                    top_p=1.0,
                                                    frequency_penalty=0.0,
                                                    presence_penalty=0.0
                                                    )
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
        print(gpt_res)
        label = label.lower()

        res.append({
            "sentence": sentence,
            "aspect": term,
            "gold": label,
            "gpt3_sentence": gpt_res
        })

    json.dump(res, fw)
    fw.close()


if __name__ == "__main__":
    # file_path = "/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/aug_absa_data/laptop/Laptops_Test_Gold.xml.seg"
    # file_path = "/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/aug_absa_data/res/Restaurants_Test_Gold.xml.seg"
    # file_path = '/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/integrated_datasets/apc_datasets/110.SemEval/113.laptop14valid/Laptops_Valid_Gold.xml.seg'
    # save_path = "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_lap_dev_prompt3.json"

    # prompt_template = "What is the sentiment polarity of the target aspect \"{}\" in the following sentence and why? {}"

    # for delay_secs in (2**x for x in range(0, 6)):
    #     try:
    #         # Call openai request such as text completion
    #         call_chatgpt(file_path, save_path, prompt_template)
    #         break

    #     except openai.OpenAIError as e:
    #         randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
    #         sleep_dur = delay_secs + randomness_collision_avoidance
    #         print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
    #         time.sleep(sleep_dur)
    #         continue

    # file_path = '/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/integrated_datasets/apc_datasets/110.SemEval/113.laptop14valid/Laptops_Test_Gold.xml.seg'
    # save_path = "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_lap_test_prompt3.json"

    # prompt_template = "What is the sentiment polarity of the target aspect \"{}\" in the following sentence and why? {}"

    # for delay_secs in (2**x for x in range(0, 6)):
    #     try:
    #         # Call openai request such as text completion
    #         call_chatgpt(file_path, save_path, prompt_template)
    #         break

    #     except openai.OpenAIError as e:
    #         randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
    #         sleep_dur = delay_secs + randomness_collision_avoidance
    #         print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
    #         time.sleep(sleep_dur)
    #         continue

    # file_path = '/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/integrated_datasets/apc_datasets/110.SemEval/113.laptop14valid/Laptops_Train.xml.seg'
    # save_path = "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_lap_train_prompt3.json"

    # prompt_template = "What is the sentiment polarity of the target aspect \"{}\" in the following sentence and why? {}"

    # for delay_secs in (2**x for x in range(0, 6)):
    #     try:
    #         # Call openai request such as text completion
    #         call_chatgpt(file_path, save_path, prompt_template)
    #         break

    #     except openai.OpenAIError as e:
    #         randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
    #         sleep_dur = delay_secs + randomness_collision_avoidance
    #         print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
    #         time.sleep(sleep_dur)
    #         continue
    file_path = '/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/integrated_datasets/apc_datasets/110.SemEval/114.restaurant14valid/Restaurants_Valid_Gold.xml.seg'
    save_path = "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_res_dev_prompt3.json"

    prompt_template = "What is the sentiment polarity of the target aspect \"{}\" in the following sentence and why? {}"

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

    file_path = '/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/integrated_datasets/apc_datasets/110.SemEval/114.restaurant14valid/Restaurants_Test_Gold.xml.seg'
    save_path = "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_res_test_prompt3.json"

    prompt_template = "What is the sentiment polarity of the target aspect \"{}\" in the following sentence and why? {}"

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

    file_path = '/home/liuxinyu/dingyan/ABSAcode/deberta-LSA/integrated_datasets/apc_datasets/110.SemEval/114.restaurant14valid/Restaurants_Train.xml.seg'
    save_path = "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_res_train_prompt3.json"

    prompt_template = "What is the sentiment polarity of the target aspect \"{}\" in the following sentence and why? {}"

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