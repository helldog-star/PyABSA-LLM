import json
dic1=json.load(open("/home/liuxinyu/dingyan/ABSAcode/generate/two_stage/match_data/laptop/stage2/chatgpt_res_total638_correct496_acc77.74.json",'r'))
dic2 = json.load(
    open(
        "/home/liuxinyu/dingyan/ABSAcode/generate/two_stage/match_data/laptop/test/laptop_addDiff_ori.json",'r'
    ))
correct=0
num=0
new=[]
for i in range(0,len(dic1)):
    dic1[i]["aug_label"]=dic2[i]["aug_label"]
    if dic1[i]["aug_label"]==dic1[i]["gpt_aug"]:
        correct+=1
    num+=1
    new.append(dic1[i])
json.dump(
    fp=open(
        "/home/liuxinyu/dingyan/ABSAcode/generate/two_stage/match_data/laptop/stage2/chatgpt_res_total638_correct_acc.json",
        'w'),obj=new,indent=-1)
print(correct)
print(num)
print(round(correct/num*100,2))