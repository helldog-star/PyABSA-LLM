# m_state_dict = torch.load('/home/liuxinyu/dingyan/prompt/new/res_finetuned_prompt_gpt2_medium.pt')
# model.load_state_dict(m_state_dict)
# model.to(device)
# model.eval()
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json
import random
import numpy as np
import nltk

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
device = 0

pos_tags={}


def select_top_k(predictions, k=10):
    # predicted_index = random.choice(
    #     predictions[0, -1, :].sort(descending=True)[1][:k]).item()
    top10=predictions[0, -1, :].sort(descending=True)[1][:k]
    predicted_index=top10[0].item()
    return predicted_index


def generate(model,tokenizer,text):
    # text = "drink<|endoftext|>negative<|endoftext|>"  # 这里也可以输入不同的英文文本
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])
    total_predicted_text = ""
    nexttokentop15={}
    for _ in range(3):
        # tokens_tensor = tokens_tensor.to(device)

        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        # token=torch.argmax(predictions[0,-1,:])
        tokensoftmax=torch.softmax(predictions[0,-1,:],dim=-1)
        predicted_index = torch.argmax(tokensoftmax).tolist()
        top15 = tokensoftmax.sort(descending=True)[:15]
        # predicted_index = select_top_k(predictions, k=10)
        sentence=tokenizer.decode(indexed_tokens)
        nexttokentop15[sentence]=[]
        nexttokentop15[sentence].append([np.var(top15[0].tolist())])
        for j in range(15):
            word=tokenizer.decode(top15[1][j].item())
            nexttokentop15[sentence].append([word,top15[0][j].item()])
        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        tokens=nltk.word_tokenize(predicted_text)
        pos_tag=nltk.pos_tag(tokens)
        if pos_tag[-1][1] in pos_tags.keys():
            pos_tags[pos_tag[-1][1]].append(np.var(top15[0].tolist()))
        else:
            pos_tags[pos_tag[-1][1]]=[np.var(top15[0].tolist())]

        total_predicted_text += tokenizer.decode(predicted_index)
        # if '<|endoftext|>' in total_predicted_text:
        #     # 如果出现文本结束标志，就结束文本生成
        #     break

        indexed_tokens += [predicted_index]

        if len(indexed_tokens) > 1023:
            # 模型最长输入长度为1024，如果长度过长则截断
            indexed_tokens = indexed_tokens[-1023:]

        tokens_tensor = torch.tensor([indexed_tokens])
    return total_predicted_text,nexttokentop15

all_generated_sentences=[]
json_path = "/home/liuxinyu/dingyan/ABSAcode/generate/data/Restaurants_Test.json"
json_file = open(json_path, 'r')
load_dict = json.load(json_file)
for sen in load_dict:
    if len(sen["token"])<6:
        continue
    sen["generation"]={}
    for i in range(2,len(sen["token"])):
        inputpart=sen["token"][0:i]
        a=" ".join(inputpart)
        sen["generated"],gene_dict=generate(model,tokenizer,a)
        sen["generation"][a]=gene_dict
    all_generated_sentences.append(sen)
with open("/home/liuxinyu/dingyan/ABSAcode/generate/data/restaurants_test_gene.json", 'w') as f_obj:
    json.dump(all_generated_sentences, f_obj, indent=1)

pos_tags_ave={}
for i in pos_tags.keys():
    pos_tags_ave[i]=np.mean(pos_tags[i])
print(pos_tags_ave)
# a=[0.2,0.3,0.4,0.5]
# c = [0.25, 0.3, 0.4, 0.45]
# b=[2,3,4,5]
# print(np.var(a))
# print(np.var(b))
# print(np.var(c))


# nltk.download()

a="I love this beautiful world"
tokens = nltk.word_tokenize(a)
pos_tags = nltk.pos_tag(tokens)
print(pos_tags[-1][0],pos_tags[-1][1])
