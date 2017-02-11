data_path = 'data/all_twitter.csv'
clean_data_path = 'data/clean_data.txt'
test_data_path = 'data/hashed_comments.txt'
import data_reader
import data_cleaner
import core

if __name__ == '__main__':
    clean_data = False
    if clean_data:
        y, docs = data_reader.import_csv(data_path)
        y = y.tolist()
        docs = docs.tolist()
        tokenizer = data_cleaner.WordTokenizer()
        for i, doc in enumerate(docs):
            words = tokenizer.divide_tweet(doc)
            lemmatized_words = data_cleaner.get_lemmatized_words(words)
            neg_words = data_cleaner.negation_unigrams(lemmatized_words)
            docs[i] = ' '.join(neg_words)
        data_reader.save_data_to_txt_file(clean_data_path, y, docs)
    else:
        ys, docs = data_reader.read_data_from_txt_file(clean_data_path)
        test_data = data_reader.read_test_data(test_data_path)

        doc, cntv = data_cleaner.get_ngrams_and_countVect(docs)
        test_data = data_cleaner.get_ngrams(test_data, cntv)

        pcaModel, train_data_pca = data_cleaner.pca_opt(array=docs)
        test_data_pca = data_cleaner.pca_opt(array=test_data, pcaModel=pcaModel)

        clf = core.train_classifier(ys, docs)
        predicted = core.test_classifier(test_data, clf)


