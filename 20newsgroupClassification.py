import os
currentDir = os.path.dirname(__file__)
# os.chdir(currentDir)
DataDir = os.path.join(currentDir,'Data')
import pandas as pd
from html.parser import HTMLParser
# import itertools
# from cleantext import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings; warnings.filterwarnings('ignore')

'''
Reading Data from folders
'''
print("Reading Data from folders..........\n")
FullData = pd.DataFrame()
for folder in os.listdir(DataDir):
    folderPath = os.path.join(DataDir,folder)
    os.chdir(folderPath)
    for file in os.listdir(folderPath):
        with open(file,'r') as f:
            text = f.read()
        label = folder
        textdf = pd.DataFrame({'text':[text],'label':[label]})
        FullData = pd.concat([FullData,textdf])
            
# FullData.to_csv('FullData.csv',index=False)

''' 
Preprocessing Data
'''
print('Preprocessing Data..........\n')
# def cleanText(text):
#     # text = clean(text,no_urls=True,no_emails=True,
#     #        no_phone_numbers=True,         # replace all phone numbers with a special token
#     #         no_numbers=True,               # replace all numbers with a special token
#     #         no_digits=True,                # replace all digits with a special token
#     #         no_currency_symbols=True,      # replace all currency symbols with a special token
#     #         no_punct=True,   )
#     # text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
#     text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) \
#                                     |(\w+:\/\/\S+)", " ", tweet).split()) 
#     return text

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stem = PorterStemmer()

def process(doc):
    import re
    stop=stopwords.words('english')
    doc=re.sub("[\d_]+", " ",doc)
    doc=re.sub("[\W_]+"," ",doc)
    x= word_tokenize(doc)
    x=[i for i in x if len(i)>5]
    stemmed=[stemmer.stem(word) for word in x]
    clean=[word for word in stemmed if not word in stop]
    joined= ' '.join(clean)
    return joined

# Data = FullData.copy()
# Data.reset_index(drop=True,inplace=True)
# def preprocessData(Data):
#     for i in range(Data.shape[0]):
#         Data.iloc[i,0] = cleanText(Data.iloc[i,0])   
#     return Data

# Data = preprocessData(Data)

'''
Model Bulding
1) tfidf vectorizer
2) Classifier
'''
print('Model Bulding..........\n')
xtrain,xtest,ytrain,ytest = train_test_split(Data['text'],Data['label'])

def Modelbuilding(xtrain,xtest,ytrain,ytest):
    tfidf = TfidfVectorizer(max_features=10000,preprocessor = process)
    xtrain = tfidf.fit_transform(xtrain)
    xtest = tfidf.transform(xtest)
    
    lr = LogisticRegression()
    lr.fit(xtrain,ytrain)
    preds = lr.predict(xtest)
    print("Accuracy : ",accuracy_score(ytest,preds))
    
Modelbuilding(xtrain,xtest,ytrain,ytest)

'''
 Final Results 94% accuracy on validation data 
'''



