data_path = 'data/all_twitter.csv'
import data_reader
import data_cleaner

if __name__ == '__main__':
    y, docs = data_reader.import_csv(data_path)
    y = y.tolist()
    docs = docs.tolist()
    tokenizer = data_cleaner.WordTokenizer
    for i, doc in enumerate(docs):
        words = tokenizer.divide_tweet(doc)
        lemmatized_words = data_cleaner.get_lemmatized_words(words)
        docs[i] = ' '.join(lemmatized_words)


    # doc, cntv = data_cleaner.get_ngrams_and_countVect(docs)

