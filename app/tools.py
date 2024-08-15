import lxml
import html5lib
from bs4 import BeautifulSoup
import re
import string
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


def clean_html(text):
    soup = BeautifulSoup(text, "html5lib")
    for tag in soup(['style', 'script']):
        tag.decompose()
    return ' '.join(soup.stripped_strings)


def clean_text(text):
    pattern = re.compile(r'[^\w]|[\d_]')

    try:
        res = re.sub(pattern," ", text).lower()
    except TypeError:
        return text

    res = res.split(" ")
    res = list(filter(lambda x: len(x)>3 , res))
    res = " ".join(res)
    return res




custom_stopwords = {
    'value', 'code', 'data', 'column', 'function', 'file',
    'import', 'name', 'return', 'class', 'error', 'line',
    'number', 'type', 'list', 'const', 'time', 'example',
    'output', 'print', 'variable', 'method', 'module',
    'object', 'package', 'parameter', 'script', 'syntax',
    'argument', 'keyword', 'library', 'instance', 'operator',
    'statement', 'expression', 'loop', 'condition', 'index',
    'array', 'boolean', 'character', 'constant', 'constructor',
    'destructor', 'exception', 'identifier', 'interface',
    'iterator', 'namespace', 'pointer', 'protocol', 'recursion',
    'reference', 'resource', 'scope', 'stack', 'struct',
    'subroutine', 'thread', 'typecast', 'unittest', 'variable',
    'widget', 'framework', 'compiler', 'interpreter', 'runtime',
    'debug', 'deploy', 'execute', 'initialize', 'install',
    'instantiate', 'integrate', 'launch', 'maintain', 'parse',
    'profile', 'refactor', 'resolve', 'serialize', 'simulate',
    'terminate', 'validate', 'workflow', 'foreach'
}


def tokenize(text, additional_stopwords=None):

    stop_words = set(stopwords.words('english'))

    if additional_stopwords:
        stop_words = stop_words.union(additional_stopwords)

    try:
        res = word_tokenize(text, language='english')
    except TypeError:
        return text

    res = [token for token in res if token not in stop_words]
    return res




def filtering_nouns(tokens):

    res = nltk.pos_tag(tokens)

    res = [token[0] for token in res if token[1] == 'NN']
    return res



def lemmatization(tokens):

    lemmatizer = WordNetLemmatizer()
    lemmatized = []

    for token in tokens:
        lemmatized.append(lemmatizer.lemmatize(token))

    return lemmatized


class SupervisedModel:

    def __init__(self):
        filename_supervised_model = "./models/SVM_model.pkl"
        filename_mlb_model = "./models/mlb_model.pkl"
        filename_tfidf_model = "./models/TFIDF_model.pkl"
        filename_pca_model = "./models/pca_model.pkl"
        filename_vocabulary = "./models/vocabulary.pkl"

        self.supervised_model = pickle.load(open(filename_supervised_model, 'rb'))
        self.mlb_model = pickle.load(open(filename_mlb_model, 'rb'))
        self.tfidf_model = pickle.load(open(filename_tfidf_model, 'rb'))
        self.pca_model = pickle.load(open(filename_pca_model, 'rb'))
        self.vocabulary = pickle.load(open(filename_vocabulary, 'rb'))

    def predict_tags(self, text):
        """
        Predict tags according to a lemmatized text using a supervied model.
        
        Args:
            supervised_model(): Used mode to get prediction
            mlb_model(): Used model to detransform
        Returns:
            res(list): List of predicted tags 
        """
        input_vector = self.tfidf_model.transform(text)
        input_vector = pd.DataFrame(input_vector.toarray(), columns=self.vocabulary)
        input_vector = self.pca_model.transform(input_vector)
        res = self.supervised_model.predict(input_vector)
        res = self.mlb_model.inverse_transform(res)
        res = list({tag for tag_list in res for tag in tag_list if (len(tag_list) != 0)})
        res = [tag for tag  in res if tag in text]
        
        return res


class LdaModel:

    def __init__(self):
        filename_model = "./models/lda_model.pkl"
        filename_dictionary = "./models/dictionary.pkl"
        self.model = pickle.load(open(filename_model, 'rb'))
        self.dictionary = pickle.load(open(filename_dictionary, 'rb'))

    def predict_tags(self, text):
        """
        Predict tags of a preprocessed text
        
        Args:
            text(list): preprocessed text
        Returns:
            res(list): list of tags
        """
        corpus_new = self.dictionary.doc2bow(text)
        topics = self.model.get_document_topics(corpus_new)
        
        #find most relevant topic according to probability
        relevant_topic = topics[0][0]
        relevant_topic_prob = topics[0][1]
        
        for i in range(len(topics)):
            if topics[i][1] > relevant_topic_prob:
                relevant_topic = topics[i][0]
                relevant_topic_prob = topics[i][1]
                
        #retrieve associated to topic tags present in submited text
        res = self.model.get_topic_terms(topicid=relevant_topic, topn=20)
        
        res = [self.dictionary[tag[0]] for tag in res if self.dictionary[tag[0]] in text]
        
        return res
