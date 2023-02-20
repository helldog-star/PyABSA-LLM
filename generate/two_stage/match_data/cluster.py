import json

dict_ori = json.load(
    open("/home/liuxinyu/dingyan/RoBertaABSA/Dataset/Laptop/Laptop_Test.json",
         "r"))
dict_aug = json.load(
    open(
        "/home/liuxinyu/dingyan/RoBertaABSA/Dataset/ARTS/ARTS_single_augment/laptop_from_arts/laptop_addDiff.json",
        "r"))
all=[]
aug={}
for sen in dict_aug:
    id=sen["id"]
    aug[id]=[sen["sentence"],sen["aspects"][0]["polarity"]," ".join(sen["aspects"][0]["term"])]
for i in range(0, len(dict_ori)):
    sen = {}
    ori = dict_ori[i]
    sen["ori"] = ori["sentence"]
    sen["ori_label"] = ori["aspects"][0]["polarity"]
    term=" ".join(ori["aspects"][0]["term"])
    id = ori["id"]
    aug_sen= aug[id][0]
    aug_label= aug[id][1]
    aug_term=aug[id][2]
    if aug_term!= term:
        for i in aug.keys():
            if i.split(":")[0]==id.split(":")[0] and term==aug[i][2]:
                aug_sen = aug[i][0]
                aug_label = aug[i][1]
                aug_term = aug[i][2]
    sen["aug"] = aug_sen
    sen["aug_label"] = aug_label
    sen["term"] = term
    all.append(sen)
# for i in range(0,len(dict_ori)):
#     sen={}
#     ori=dict_ori[i]
#     aug=dict_aug[i]
#     sen["ori"]=ori["sentence"]
#     sen["ori_label"]=ori["aspects"][0]["polarity"]
#     sen["aug"]=aug["sentence"]
#     sen["aug_label"]=aug["aspects"][0]["polarity"]
#     sen["term"]=" ".join(ori["aspects"][0]["term"])
#     all.append(sen)
json.dump(fp=open(
    "/home/liuxinyu/dingyan/ABSAcode/generate/two_stage/match_data/laptop/test/laptop_addDiff_ori.json",
    "w"),
          obj=all,
          indent=-1)
