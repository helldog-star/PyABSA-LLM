import json

dict_train=json.load(open("/home/liuxinyu/dingyan/ABSAcode/generate/two_stage/match_data/laptop/train/laptop_addDiff_ori.json",'r'))
dict_test = json.load(
    open(
        "/home/liuxinyu/dingyan/ABSAcode/generate/two_stage/match_data/laptop/test/laptop_addDiff_ori.json",'r'
    ))
all_train={}
for sen in dict_train:
    term= sen["term"]
    all_train[term]=sen
i=0
res=[]
for sen in dict_test:
    term = sen["term"]
    if term in all_train:
        i+=1
        sen["train_example"]=all_train[term]
    res.append(sen)
print(i)
json.dump(
    fp=open(
        "/home/liuxinyu/dingyan/ABSAcode/generate/two_stage/match_data/laptop/test/adddiff_with_example.json",'w'
    ),obj=res,indent=-1)
