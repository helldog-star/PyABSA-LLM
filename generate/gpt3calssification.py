import os
import openai
import json
import time
openai.api_key = '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'



def generate_simple(prompt):

    while True:
        try:
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=30,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=".")
        except openai.error.ServiceUnavailableError as e:
            time.sleep(60)
        else:
            break
    sentence=response["choices"][0]["text"]
    return sentence


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


json_path = "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Laptop/Laptop_Test.json"
# json_path="/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Restaurants/Restaurants_Test.json"
json_file = open(json_path, 'r')
load_dict = json.load(json_file)
all_generated_sentences = []
i=0
for sen in load_dict:
    if i%10==0 and i!=0:
        time.sleep(60)
    term_token = sen["aspects"][0]["term"]
    prompt = sen["sentence"]+"What's the sentiment polarity of \"" + " ".join(term_token) + "\" in the above sentence, neutral, positive, or negative?"
    gene_sentence=generate_simple(prompt)
    # prompt = "Des \"" + " ".join(term_token) + "\"\'s sentiment is positive, neutral, or negative."+sen[
    #     "sentence"]
    part = gene_sentence.replace("\n", " ")
    part=part+"."
    sen["sentence"] = part
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
        "/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene_Laptop_Test_prompt2.json",
        'w') as f_obj:
    json.dump(all_generated_sentences, f_obj, indent=1)