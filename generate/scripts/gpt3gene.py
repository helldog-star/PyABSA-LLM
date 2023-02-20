import os
import openai
import json
import time
import argparse


def generate_simple(prompt):

    while True:
        try:
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=30,
            top_p=0,
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


parser = argparse.ArgumentParser()
parser.add_argument(
    '--key', default="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",type=str,help="openai_key")
parser.add_argument('--data_path',type=str,default="/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Restaurants/Restaurants_Test.json")
parser.add_argument('--output_path',type=str,default="/home/liuxinyu/dingyan/ABSAcode/generate/gpt3gene/gpt3classi_res_prompt1.json")
parser.add_argument('--prompt',type=int,default=1)
args=parser.parse_args()
openai.api_key = args.key
# json_path = "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/ARTS/Laptop/ARTS_Laptop_Test.json"
# json_path = "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Laptop/Laptop_Test.json"
# json_path="/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Restaurants/Restaurants_Test.json"
# json_path = "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/mams/test_processed.json"
json_path=args.data_path
json_file = open(json_path, 'r')
load_dict = json.load(json_file)
all_generated_sentences = []
i=0
correct=0
for sen in load_dict:
    if i%10==0 and i!=0:
        time.sleep(60)
    term_token = sen["aspects"][0]["term"]
    if args.prompt==1:
        prompt = "Decide whether the \"" + " ".join(term_token) + "\"'s sentiment is positive, neutral, or negative."+sen[
        "sentence"]    #prompt1
    elif args.prompt==2:
        prompt = sen["sentence"] + "What's the sentiment polarity of \"" + " ".join(term_token) + "\" in the above sentence, neutral, positive, or negative?"  # prompt2
    elif args.prompt == 3:
        prompt = "What's the sentiment of \"" + " ".join(
                term_token
            ) + "\" in the following sentence, neutral, positive, or negative? "+sen["sentence"]  # prompt3
    elif args.prompt ==4:
        prompt= sen["sentence"]+" "+sen["gpt3_sentence"]+" Decide whether the \"" + " ".join(term_token) + "\"'s sentiment is positive, neutral, or negative."
    print(prompt)
    gene_sentence=generate_simple(prompt)
    part = gene_sentence.strip("\n")
    part =part.strip()
    sen["gpt3_classification"] = part
    print(part)
    if sen["aspects"][0]["polarity"].lower()==part.lower():
        correct=correct+1
    all_generated_sentences.append(sen)
    i=i+1

print(correct)
print(i)
print(correct/i)
outputpath=args.output_path
with open(
        outputpath,
        'w') as f_obj:
    json.dump(all_generated_sentences, f_obj, indent=1)