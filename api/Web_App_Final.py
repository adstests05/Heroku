#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, redirect, url_for, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import pickle
import numpy as np 
import nltk
from nltk.corpus import stopwords 
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from newsplease import NewsPlease
import newspaper
from newspaper import Article
from sklearn import svm
import os
import boto3
nltk.download('stopwords')

# Set the limit for number of articles to download
def dailyNews(paper):
    LIMIT = 30
    title_csv = []
    content_csv=[]
    url_csv = []
    # It uses the python newspaper library to extract articles
#    print("Building site for ", paper)
    paper = newspaper.build(paper, memoize_articles=False)
    noneTypeCount = 0
    count=1

    for content in paper.articles:
        if count > LIMIT:
            break
        try:
            content.download()
            content.parse()
        except Exception as e:
            print(e)
            print("continuing...")
            continue
        # Again, for consistency, if there is no found publish date the article will be skipped.
        if content.publish_date is None:
            noneTypeCount = noneTypeCount + 1
            count = count + 1
            continue
        if content.title == 'Terms of Service' or content.title=='Privacy Policy' or content.url.startswith('https://cn.nytimes.com/') or content.url.startswith('http://cn.nytimes.com/'):
            err = 'error'
        else:
            title = content.title
            title_csv.append(title)

            text = content.text
            content_csv.append(text)
            
            url = content.url
            url_csv.append(url)

            count = count + 1
            noneTypeCount = 0     
    count = 1
    newsResult = dict(zip(title_csv, content_csv))
    urlResult = dict(zip(title_csv, url_csv))
    return newsResult,urlResult;

def url_Contents(url_article):
    article = NewsPlease.from_url(url_article)
    if (article.text) == None:
        print('None')
    else:
        content = article.text
    return content

porter_stemmer = nltk.stem.porter.PorterStemmer()

#spilts the sentences into words
def porter_tokenizer(text, stemmer=porter_stemmer):
    lower_txt = text.lower()
    tokens = nltk.wordpunct_tokenize(lower_txt)
    stems = [porter_stemmer.stem(t) for t in tokens]
    no_punct = [s for s in stems if re.match('^[a-zA-Z]+$', s) is not None]
    return no_punct

stop_words = set(stopwords.words('english'))

tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                   encoding='utf-8',
                                   decode_error='replace',
                                   strip_accents='unicode',
                                   analyzer='word',
                                   tokenizer=porter_tokenizer,
                                   ngram_range=(1,2),
                                   binary=False)


def vectorizeData():
    aws_id = 'AKIAJ33LPWAZXRN3DYUA'
    aws_secret = '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'
    client = boto3.client('s3',aws_access_key_id=aws_id, aws_secret_access_key=aws_secret)
    obj_job = client.get_object(Bucket='ads-final', Key='CleanData.csv')
    df = pd.read_csv(obj_job['Body'],encoding='utf-8')
    le = LabelEncoder()
    le.fit(df['Label'])

    df_labels = pd.DataFrame(np.array(le.transform(df['Label'])))
    
    skf = StratifiedKFold(n_splits = 5)
    
    for trn_indx, tst_indx in skf.split(df['Content'],df_labels):
        skf_X_train, skf_X_test = df['Content'].iloc[trn_indx], df['Content'].iloc[tst_indx]
        skf_Y_train, skf_Y_test = df_labels.iloc[trn_indx], df_labels.iloc[tst_indx]
        
    # Fit and transform the training data for tfidf
    tfidf_train = tfidf_vectorizer.fit_transform(skf_X_train)

    # Transform the test set 
    tfidf_test = tfidf_vectorizer.transform(skf_X_test)
    
    return tfidf_train,tfidf_test,skf_Y_train,skf_Y_test

#Generate the training and testing dataset
X_train, X_test ,Y_train, Y_test  = vectorizeData()

#Using the SVM Model
clf = svm.LinearSVC()
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)

###########################################

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/dailyNews',methods=['GET','POST'])
def daily_news():
    if request.method =='GET':
        methods='GET'
        return render_template('daily_news.html',methods=methods)
    elif request.method =='POST':
        url_final=[]
        title_lis =[]
        url_lis=[]
        content_lis=[]
        methods='POST'
        hidden1=request.form['input1']
        paper=request.form['paper']
        title_list,url_list = dailyNews(paper)
        my_prediction=[]
        for title,comment in title_list.items():
            data = [comment]
            vect = tfidf_vectorizer.transform(data).toarray()
            my_predict = clf.predict(vect)
            my_prediction.append(my_predict)
        for a,b in title_list.items():
            title_lis.append(a)
            content_lis.append(b)
        for c,d in url_list.items():
            url_lis.append(d)
        if paper =='https://www.nytimes.com/':
            paperName = 'New York Times'
        elif paper =='https://www.huffpost.com/':
            paperName = 'Huff Post'
        elif paper =='https://worldnewsdailyreport.com/':
            paperName = 'World News Daily Report'
        paperNames=[]
        c=len(title_lis)
        for i in range(1,c+1):
            paperNames.append(paperName)
        ################
#         count=1
#         if count==1:
#             title_csv = [e for e in title_lis]
#             url_csv = [f for f in url_lis]
#             content_csv = [g for g in content_lis]
#             name_csv = [h for h in paperNames]
#             count=count+1
#         elif count > 1:
#             for el in title_lis:
#                 title_csv.append(el)
#             for ele in url_lis:
#                 title_csv.append(ele)
#             for elem in content_lis:
#                 title_csv.append(elem)
#             for elemt in paperNames:
#                 title_csv.append(elemt)
        ################
        df = pd.DataFrame(data={'Title': title_lis,'Publication':paperNames,'URL': url_lis, 'Content': content_lis})
        df.to_csv(os.getcwd()+'/Daily_read.csv')
        return render_template('daily_news.html',methods=methods,prediction = my_prediction,title_lis=title_lis,url_lis=url_lis)

@app.route('/urlNews',methods=['GET','POST'])
def url_news():
    if request.method =='GET':
        methods='GET'
        return render_template('url_news.html',methods=methods)
    elif request.method =='POST':
        hidden2=request.form['input2']
        methods='POST'
        url_article = request.form['url']
        comment = url_Contents(url_article)
        data = [comment]
        vect = tfidf_vectorizer.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('url_news.html',hidden2=hidden2,methods=methods,url_article=url_article,prediction = my_prediction)


@app.route('/content',methods=['GET','POST'])
def content():
    if request.method =='GET':
        methods='GET'
        return render_template('content.html',methods=methods)
    
    elif request.method =='POST':
        hidden3=request.form['input3']
        methods='POST'
        comment = request.form['comment']
        data = [comment]
        vect = tfidf_vectorizer.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('content.html',hidden3=hidden3,methods=methods,prediction = my_prediction,comment=comment)

if __name__ == '__main__':
    app.run(host='0.0.0.0')


# In[ ]:





# In[ ]:




