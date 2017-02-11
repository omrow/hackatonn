import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from scipy import sparse
import numpy as np

ngram_min = 1
ngram_max = 1
countVec_min_df = 3

class WordTokenizer(object):
    '''
        WordTokenizer divides each string into isolated
        words, this operation is performed using python
        Regular Expression
    '''
    emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [D\)\]\(\]/\\OpP] # Mouth
        )"""

    regex_str = [
        emoticons_str,
        r'<[^>]+>',  # HTML tags
        r'(?:@[\w_]+)',  # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

        r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
        # r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
        r'(?:[\w_]+)',  # other words
        r'(?:\S)'  # anything else
    ]

    # punctation list
    punctation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
                  '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
                  '{', '|', '}', '~']

    def __init__(self):
        self.tokens_re = re.compile(r'(' + '|'.join(self.regex_str) + ')',
                                    re.UNICODE | re.VERBOSE | re.IGNORECASE)
        self.emoticon_re = re.compile(r'^' + self.emoticons_str + '$',
                                      re.UNICODE | re.VERBOSE | re.IGNORECASE)
        self.undef_re = re.compile(r'^' + self.regex_str[-1] + '$',
                                   re.UNICODE | re.VERBOSE | re.IGNORECASE)
        self.men_re = re.compile(r'^' + self.regex_str[2] + '$',
                                 re.UNICODE | re.VERBOSE | re.IGNORECASE)
        self.url_re = re.compile(r'(' + '|'.join([self.regex_str[1], self.regex_str[4]]) + ')',
                                 re.UNICODE | re.VERBOSE | re.IGNORECASE)

    def tokenize(self, word):
        return self.tokens_re.findall(word)

    def preprocess(self, s, lowercase=False, words_only=False):
        tokens = self.tokenize(s)
        if words_only:
            tokens = [token
                      for token in tokens
                      if not self.emoticon_re.search(token)
                      and not self.url_re.search(token)
                      and not self.undef_re.search(token)
                      and not self.men_re.search(token)
                      ]
        # Lowercase option for words, not emoticon
        if lowercase:
            tokens = [token if self.emoticon_re.search(token) else token.lower() for token in tokens]
        return tokens

    def divide_tweet(self, text):
        stop = self.punctation
        str_words = self.preprocess(text, lowercase=False,
                                         words_only=True)
        str_words = [term for term in str_words if term not in stop]
        return str_words

def get_lemmatized_words(words):
    lmtzr = WordNetLemmatizer()
    morphy_tag = {'NN': wordnet.NOUN, 'JJ': wordnet.ADJ, 'VB': wordnet.VERB, 'RB': wordnet.ADV}
    return [w_tuple[0] if w_tuple[1][:2] not in morphy_tag else lmtzr.lemmatize(w_tuple[0], morphy_tag.get(w_tuple[1][:2])) for w_tuple in pos_tag(words)]

def get_ngrams_and_countVect(train_data):
    count_vect = CountVectorizer(ngram_range=(ngram_min, ngram_max), lowercase=True, min_df=countVec_min_df)
    X = count_vect.fit_transform(train_data).toarray()
    return X, count_vect

def get_ngrams(test_data, cntvect):
    new_vec = CountVectorizer(ngram_range=(ngram_min, ngram_max), lowercase=True, vocabulary=cntvect.vocabulary_)
    X = new_vec.fit_transform(test_data).toarray()
    return X

def pca_opt(array, pcaModel=None, svdopt=500):
    chunkSize = 500
    if pcaModel is None:
        times_less = len(array[0]) / svdopt
        pca_desc = 'Dlugosc wektora opisujacego dzieło zredukowano ' + str(int(times_less)) + '-krotnie. Finalna liczba atrybutów dla dzieła: ' + str(svdopt) + '.'
        print(pca_desc)

        if len(array) < 1000:
            pca = PCA(n_components=svdopt)
            Xtransformed = pca.fit_transform(array)
            Xtransformed = sparse.csr_matrix(Xtransformed)
        else:
            iter = int(len(array) / chunkSize)
            chunks = [array[i:i + chunkSize] for i in range(0, iter * chunkSize, chunkSize)]
            pca = IncrementalPCA(n_components=svdopt)

            for chunk in chunks:
                pca.partial_fit(chunk)

            Xtransformed = None
            for chunk in chunks:
                Xchunk = pca.transform(chunk)
                if Xtransformed is None:
                    Xtransformed = Xchunk
                else:
                    Xtransformed = np.vstack((Xtransformed, Xchunk))
            Xtransformed = sparse.csr_matrix(Xtransformed)

        return pca, Xtransformed.toarray()
    else:
        if len(array) < 1000:
            Xtransformed = pcaModel.transform(array)
            Xtransformed = sparse.csr_matrix(Xtransformed)
        else:
            chunks = [array[i:i + chunkSize] for i in range(0, len(array), chunkSize)]

            Xtransformed = None
            for chunk in chunks:
                Xchunk = pcaModel.transform(chunk)
                if Xtransformed is None:
                    Xtransformed = Xchunk
                else:
                    Xtransformed = np.vstack((Xtransformed, Xchunk))
            Xtransformed = sparse.csr_matrix(Xtransformed)

        return Xtransformed.toarray()