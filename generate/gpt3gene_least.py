import os
import openai
import json
import time
# openai.api_key = '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'
openai.api_key = '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'
# openai.api_key='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'
# openai.api_key="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"
# openai.api_key = "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"


def generate_simple(prompt):

    while True:
        try:
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=50,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            )
        except openai.error.ServiceUnavailableError or openai.error.RateLimitError as e:
            time.sleep(60)
        else:
            break
    sentence=response["choices"][0]["text"]
    return sentence


# print(generate_simple("This apple seems delicious. what's the opinion of \"apple\" in the above sentence? "))


def ctot(token, term):
    i = 0
    flag = 0
    while i < len(token):
        if (term[0] == token[i]):
            j = 1
            if (len(term) == 1):
                flag = 1
            while j < len(term):
                if (term[j] != token[i + j]):
                    break
                j = j + 1
                if j == len(term):
                    flag = 1
        if flag == 1:
            break
        i = i + 1
    return i


# json_path = "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Laptop/Laptop_Test.json"
# json_path="/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Restaurants/Restaurants_Test.json"
# json_path = "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/ARTS/Restaurants/ARTS_Restaurants_Test.json"
# json_path = "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/ARTS/Laptop/ARTS_Laptop_Test.json"
json_path = "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Restaurants/Restaurants_Train.json"
# json_path="/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Laptop/Laptop_Dev.json"
json_file = open(json_path, 'r')
load_dict = json.load(json_file)
all_generated_sentences = []
i=0
for sen in load_dict:
    if i%10==0 and i!=0:
        time.sleep(60)
    term_token = sen["aspects"][0]["term"]
    # prompt = "Des \"" + " ".join(term_token) + "\"\'s sentiment is positive, neutral, or negative."+sen[
    #     "sentence"]
    # prompt = "What's the opinion of \"" + " ".join(term_token) + "\" in the following sentence? "+sen["sentence"]
    # prompt = "What's the description of \"" + " ".join(term_token) + "\" in the following sentence? " + sen["sentence"]
    prompt = "What's the sentiment polarity of \"" + " ".join(term_token) + "\" in the following sentence and why? " + sen["sentence"]
    gene_sentence=generate_simple(prompt)
    part = gene_sentence.replace("\n", " ")
    part=part+"."
    sen["gpt3_sentence"] = part
    print(part)
    part=part.replace("."," .")
    sentence_space=part.replace(","," ,")
    tokens=sentence_space.split(" ")
    sen["token"]=tokens
    # fromid=ctot(sen["token"],term_token)
    # toid=fromid+len(term_token)
    # sen["aspects"][0]["from"]=fromid
    # sen["aspects"][0]["to"]=toid
    all_generated_sentences.append(sen)
    i=i+1

with open(
        "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_res_train_prompt3.json",
        'w') as f_obj:
    json.dump(all_generated_sentences, f_obj, indent=1)
# with open(
#         "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_res_arts_prompt2.json",
#         'w') as f_obj:
#     json.dump(all_generated_sentences, f_obj, indent=1)

json_path = "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Restaurants/Restaurants_Dev.json"
# json_path="/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Laptop/Laptop_Dev.json"
json_file = open(json_path, 'r')
load_dict = json.load(json_file)
all_generated_sentences = []
i=0
for sen in load_dict:
    if i%10==0 and i!=0:
        time.sleep(60)
    term_token = sen["aspects"][0]["term"]
    # prompt = "Des \"" + " ".join(term_token) + "\"\'s sentiment is positive, neutral, or negative."+sen[
    #     "sentence"]
    # prompt = "What's the opinion of \"" + " ".join(term_token) + "\" in the following sentence? "+sen["sentence"]
    # prompt = "What's the description of \"" + " ".join(term_token) + "\" in the following sentence? " + sen["sentence"]
    prompt = "What's the sentiment polarity of \"" + " ".join(term_token) + "\" in the following sentence and why? " + sen["sentence"]
    gene_sentence=generate_simple(prompt)
    part = gene_sentence.replace("\n", " ")
    part=part+"."
    sen["gpt3_sentence"] = part
    print(part)
    part=part.replace("."," .")
    sentence_space=part.replace(","," ,")
    tokens=sentence_space.split(" ")
    sen["token"]=tokens
    # fromid=ctot(sen["token"],term_token)
    # toid=fromid+len(term_token)
    # sen["aspects"][0]["from"]=fromid
    # sen["aspects"][0]["to"]=toid
    all_generated_sentences.append(sen)
    i=i+1

with open(
        "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_res_dev_prompt3.json",
        'w') as f_obj:
    json.dump(all_generated_sentences, f_obj, indent=1)

json_path = "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Restaurants/Restaurants_Test.json"
# json_path="/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Laptop/Laptop_Dev.json"
json_file = open(json_path, 'r')
load_dict = json.load(json_file)
all_generated_sentences = []
i=0
for sen in load_dict:
    if i%10==0 and i!=0:
        time.sleep(60)
    term_token = sen["aspects"][0]["term"]
    # prompt = "Des \"" + " ".join(term_token) + "\"\'s sentiment is positive, neutral, or negative."+sen[
    #     "sentence"]
    # prompt = "What's the opinion of \"" + " ".join(term_token) + "\" in the following sentence? "+sen["sentence"]
    # prompt = "What's the description of \"" + " ".join(term_token) + "\" in the following sentence? " + sen["sentence"]
    prompt = "What's the sentiment polarity of \"" + " ".join(term_token) + "\" in the following sentence and why? " + sen["sentence"]
    gene_sentence=generate_simple(prompt)
    part = gene_sentence.replace("\n", " ")
    part=part+"."
    sen["gpt3_sentence"] = part
    print(part)
    part=part.replace("."," .")
    sentence_space=part.replace(","," ,")
    tokens=sentence_space.split(" ")
    sen["token"]=tokens
    # fromid=ctot(sen["token"],term_token)
    # toid=fromid+len(term_token)
    # sen["aspects"][0]["from"]=fromid
    # sen["aspects"][0]["to"]=toid
    all_generated_sentences.append(sen)
    i=i+1

with open(
        "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/results/gpt3gene_res_test_prompt3.json",
        'w') as f_obj:
    json.dump(all_generated_sentences, f_obj, indent=1)