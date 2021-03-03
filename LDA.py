# -*- coding: utf-8 -*-
from platform import python_version
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
#from nltk.corpus import stopwords
#from gensim import corpora

if __name__ =='__main__':
    print(python_version())
    dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
    documents = dataset.data

    # 20 카테고리
    #print(dataset.target_names)

    # 텍스트 전처리: 특수문자 제거, 길이 3이하 단어 제거, 소문자 변환
    news_df = pd.DataFrame({'document': documents})
    news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")
    news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
    # print(news_df['clean_doc'][1])

    # 불용어 처리, 토큰화
    # stop_words = stopwords.words('english')
    tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
    # tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    print(tokenized_doc[1])



    print("done")
