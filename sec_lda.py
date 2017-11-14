from bs4 import BeautifulSoup
import urllib2
import time
import stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import gensim
import pandas as pd

def get_url_from_csv(file, num_url):
    df = pd.read_csv(file, delimiter=',', header=0)
    return df['Doc URL'][:num_url]

def get_filename_from_url(url):
    return './lda_doc/' + url.split('/')[-3] + '_' + url.split('/')[-1] + '.txt'

def save_html_to_txt(url):
    html = urllib2.urlopen(url)
    time.sleep(1)
    soup = BeautifulSoup(html, 'lxml')
    text = soup.get_text().encode('utf-8')
    filename = get_filename_from_url(url)
    file(filename, 'wb').write(text)
    return filename, text

def get_doc_dict(urls):
    doc_dict = dict()
    for url in urls:
        filename, text = save_html_to_txt(url)
        doc_dict[filename] = text
    return doc_dict

def read_doc_dict_from_disk(urls):
    doc_dict = dict()
    for url in urls:
        filename = get_filename_from_url(url)
        text = file(filename, 'rb').read().decode('utf-8')
        doc_dict[filename] = text.lower().strip()
    return doc_dict

# doc_urls = ['https://www.sec.gov/Archives/edgar/data/1171008/000119380514001778/e612662_ex2-1.htm',
#             'https://www.sec.gov/Archives/edgar/data/1370030/000118518511000522/ex2-1.htm',
#             'https://www.sec.gov/Archives/edgar/data/1370030/000118518511000522/ex2-2.htm',
#             'https://www.sec.gov/Archives/edgar/data/1170103/000109690607001270/ci8k091407ex2-1.htm',
#             'https://www.sec.gov/Archives/edgar/data/1342916/000116552709000645/ex2-1.txt',
#             'https://www.sec.gov/Archives/edgar/data/790024/000102317506000227/epat8k21.txt',
#             'https://www.sec.gov/Archives/edgar/data/837179/000101968700001178/0001019687-00-001178-0002.txt']
doc_urls = get_url_from_csv('./doc/CIK-8-K-EX-2.1.csv', 10)

# download the url docs into disk
# doc_dict = get_doc_dict(doc_urls)

# read the doc once it is on disk
doc_dict = read_doc_dict_from_disk(doc_urls)

print 'Tokenizing\n'

# 1) Tokenizing the document
tokenizer = RegexpTokenizer(r'[\w\-.]+')
for filename, text in doc_dict.iteritems():
    doc_dict[filename] = tokenizer.tokenize(text)
    print doc_dict[filename]

print '\nStop-words\n'

#  2) Filter out the stop words from the document
en_stop = stop_words.get_stop_words('en')
for filename, text in doc_dict.iteritems():
    stopped_tokens = [i for i in text if i not in en_stop]
    doc_dict[filename] = stopped_tokens
    print stopped_tokens

print '\nStemming\n'

# 3) Stem the words in the document
p_stemmer = PorterStemmer()
for filename, text in doc_dict.iteritems():
    stemmed_tokens = [p_stemmer.stem(i) for i in text]
    doc_dict[filename] = stemmed_tokens
    print stemmed_tokens

print '\n'

# construct a document-term matrix
dictionary = gensim.corpora.Dictionary(doc_dict.values())
corpus = [dictionary.doc2bow(text) for text in doc_dict.values()]

# train the LDA model by feeding the corpus and configuration
num_topics = 2
num_words = 7
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
for topic in ldamodel.print_topics(num_topics=num_topics, num_words=num_words):
    print topic