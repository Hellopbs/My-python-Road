# -*- coding: utf-8 -*-
from os import listdir
import jieba
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
all_file=listdir('D:/Python_work/Data Mining/db250')
labels=[]
corpus=[]
typetxt=open('D:/Python_work/Data Mining/文本相似度计算/停用词.txt')
texts=['\u3000','\n',' ']
for word in typetxt:
    word=word.strip()
    texts.append(word)
for i in range(0,len(all_file)):
    filename=all_file[i]
    filelabel=filename.split('.')[0]
    labels.append(filelabel)
    file_add='D:/Python_work/Data Mining/db250/'+ filename
    doc=open(file_add,encoding='utf-8').read()
    data=jieba.cut(doc)
    data_adj=''
    delete_word=[]
    for item in data:
        if item not in texts:
            data_adj+=item+' '
        else:
            delete_word.append(item)        
    corpus.append(data_adj)
vectorizer=CountVectorizer()
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
weight=tfidf.toarray()
word=vectorizer.get_feature_names()
from sklearn.cluster import KMeans
mykms=KMeans(n_clusters=10)
y=mykms.fit_predict(weight)
for i in range(0,10):
    label_i=[]
    for j in range(0,len(y)):
        if y[j]==i:
            label_i.append(labels[j])
    print('label_'+str(i)+':'+str(label_i))

