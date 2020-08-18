# utilize glove to be the initial word representation
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
input_file = r'glove.6B.50d.txt'
output_file = r'gensim_glove.6B.50d.txt'
glove2word2vec(input_file, output_file)

# Glove model
model = KeyedVectors.load_word2vec_format(output_file, binary=False)
#==========================================================================================#
#read the datasets you have (have "sentence" column , "wikidata claim" column, and "label" column)
import pandas as pd
f=pd.read_csv(r"data.csv",lineterminator='\n' )
plain=f["sentence"].tolist()
wikid=f["wikidata"].tolist()
#=========================================================================================#
#drop the stopwords for sentences and wikidata
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
EngStopWords=set(stopwords.words("english"))

drop_stop=[]
for p in range(0,len(plain)):
    j=[]
    lower=plain[p].lower()
    for word in lower.split():
        if word in EngStopWords:
            pass
        else:
            j.append(word)
    
    d=j[0]
    for i in range(1,len(j)):
        d=d+" "+j[i]
    drop_stop.append(d)
    
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

wikidata=[]
for p in range(0,len(wikid)):
    j=[]
    lower=wikid[p].lower()
    for word in lower.split():
        if word in EngStopWords:
            pass
        else:
            j.append(word)
    
    d=j[0]
    for i in range(1,len(j)):
        d=d+" "+j[i]
    
    wikidata.append(d)

#==============================================================================#
# do the stemming
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
plain=[]
for i in range(0,len(drop_stop)):
    tokens = word_tokenize(drop_stop[i])  
    tagged_sent = nltk.pos_tag(tokens)    
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) 
    delimiter = ' '
    ff=delimiter.join(lemmas_sent)
    plain.append(ff)
#===========================================================================#    
#Let wikidata and sentence to have their GloVe word representation    
wl=length of wikidata
sl=length of sentence
wikidata_e=[]
for i in range(len(wikidata)):
    a=wikidata[i].split()
    w=[]
    for j in range(len(a)):
        try:
            w.append(model[a[j]].tolist())
        except:
            w.append([0]*50)
    if len(w)>wl:
        w=w[0:wl]
    else:
        for k in range(wl-len(w)):
            w.append([0]*50)
    wikidata_e.append(w)
    
plain_e=[]
for i in range(len(plain)):
    a=plain[i].split()
    w=[]
    for j in range(len(a)):
        try:
            w.append(model[a[j]].tolist())
        except:
            w.append([0]*50)
    if len(w)>sl:
        w=w[0:sl]
    else:
        for k in range(sl-len(w)):
            w.append([0]*50)
    plain_e.append(w)
    
  
    
