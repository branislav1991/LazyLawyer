import argparse
from itertools import chain
from docai.content_generator import ContentGenerator
from docai.database import table_cases, table_docs, table_doc_contents
from docai import helpers
import numpy as np
import os
import pickle
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def train_nb(model_name, test_examples=5):
    assert model_name in ['bernoulli_nb', 'multinomial_nb']

    print("Initializing database and loading features and labels...")
    docs = table_docs.get_docs_with_names(['Judgment'])
    cases = [table_cases.get_case_for_doc(doc) for doc in docs]

    helpers.create_folder_if_not_exists('trained_models')
    model_path = os.path.join('trained_models', helpers.setup_json['subject_classifier_path'])

    content_gen = ContentGenerator(docs)
    contents = list(content_gen) # generate all contents at once
    contents = [chain.from_iterable(doc) for doc in contents]

    count_vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, lowercase=False, stop_words='english')
    X_train = count_vectorizer.fit_transform(contents[:-test_examples])
    X_test = count_vectorizer.transform(contents[-test_examples:])

    Y = [case['subject'] for case in cases]
    setY = set(Y)
    labels = dict(zip(setY, range(len(setY))))
    Y = [labels[subject] for subject in Y]
    Y = np.asarray(Y)
    Y_train = Y[:-test_examples]
    Y_test = Y[-test_examples:]

    print("Training model...")
    if model_name == 'bernoulli_nb':
        model = BernoulliNB()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        accuracy = np.sum(Y_pred == Y_test) / len(Y_pred)
        print('Bernoulli NB model accuracy: {0}'.format(accuracy))

    else: # if model_name == 'multinomial_nb'
        model = MultinomialNB()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        accuracy = np.sum(Y_pred == Y_test) / len(Y_pred)
        print('Multinomial NB model accuracy: {0}'.format(accuracy))

    print("Saving model...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train word2vec model on CURIA documents')
    parser.add_argument('model', choices=['bernoulli_nb', 'multinomial_nb'], help='which model should be trained')

    args = parser.parse_args()
    train_nb(args.model)