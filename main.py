import glob
import numpy as np
import math
import os
from scipy.spatial import distance
from music import Speech_to_Text

def load_data_in_a_directory(data_path):
    file_paths = os.listdir(data_path)
    lst_contents = []
    set_words=set()
    for file_path in file_paths:
        with open('./data/'+file_path, 'r',encoding='utf8') as f:
            str = f.read()
            words = str.replace('"', '').replace('.', '').replace("'","").lower().split()
            set_words.update(words)
            lst_contents.append(words)
    return (lst_contents, file_paths,set_words)

# Doc noi dung cua tung file txt
# va xay dung tap "dictionary" chua danh sach cac tu
def build_dictionary(contents,paths,set_words):
    dictionary = set_words
    term_dict={}
    for term in dictionary:
        term_dict[term]={}
        for i in range(len(contents)):
            if term in contents[i]:
                term_dict[term][paths[i]]=contents[i].count(term)/len(contents[i]) 

    return term_dict

# MAIN

# BUOC 1: Load cac file trong 'data' va xay dụng tap cac tu vung
print("Đang xây dựng TF - IDF")
contents, paths,set_words = load_data_in_a_directory('./data/')
vocab = build_dictionary(contents,paths,set_words)
print("Done!!!")

# BUOC 2: Xay dung vector TF weighting cho 
# tap van ban va truy van
#tính độ tương đồng bằng L2


def calc_dist_L2(vocab,qcontent,paths,v_q):
    distances=[]
    for i in range(len(paths)):
        L2=0
        IDF=[]
        v_doc=[]
        for key in vocab.keys():
            IDF.append( 1+math.log2(len(paths)/len(vocab[key])) )
            v_doc.append( vocab[key].get(paths[i],0) )
        L2 = np.linalg.norm( np.array(v_doc)*np.array(IDF) - np.array(v_q)*np.array(IDF) )
        distances.append(math.sqrt(L2))
    return distances

def calc_dist_L1(vocab,qcontent,paths,v_q):
    distances=[]
    for i in range(len(paths)):
        L2=0
        IDF=[]
        v_doc=[]
        for key in vocab.keys():
            IDF.append( 1+math.log2(len(paths)/len(vocab[key])) )
            v_doc.append( vocab[key].get(paths[i],0) )
        L2 = np.linalg.norm( np.array(v_doc)*np.array(IDF) - np.array(v_q)*np.array(IDF) , ord=1)
        distances.append(math.sqrt(L2))
    return distances

def calc_dist_Cosine(vocab,qcontent,paths,v_q):
    distances=[]
    for i in range(len(paths)):
        L2=0
        IDF=[]
        v_doc=[]
        for key in vocab.keys():
            IDF.append( 1+math.log2(len(paths)/len(vocab[key])) )
            v_doc.append( vocab[key].get(paths[i],0) )
        L2 = distance.cosine( np.array(v_doc)*np.array(IDF) , np.array(v_q)*np.array(IDF) )
        distances.append(math.sqrt(L2))
    return distances


# BUOC 5: Tinh do tuong dong cua query va cac van ban
# su dung TF_IDF weighting
print("Nhap ten file audio: ")
filename_audio=input()
query=Speech_to_Text(filename_audio)
qcontent = query.split()
#tinh TF cho query
v_q=[]
for key in vocab.keys():
        v_q.append( qcontent.count(key)/len(qcontent) )
#Bat dau truy van
print('Câu truy vấn: '+query)
print('Searching......................')
dist=calc_dist_L2(vocab,qcontent,paths,v_q)

rank = np.argsort(dist)
#print(rank[:6])
topK = 1
for i in range(topK):
    print('Bài hát bạn muốn tìm là: ' + paths[rank[i]].replace('.txt','') )




