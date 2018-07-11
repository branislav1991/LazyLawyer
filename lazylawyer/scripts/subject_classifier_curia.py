import argparse
from itertools import chain
from lazylawyer.content_generator import ContentGenerator
from lazylawyer.database import table_cases, table_docs, table_doc_contents
from lazylawyer import helpers
import numpy as np
import os
import pickle
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer

# set general model path
helpers.create_folder_if_not_exists('trained_models')
model_path = os.path.join('trained_models', helpers.setup_json['subject_classifier_path'])

def _identity_func(x):
    return x

def _multiply_func(x):
    return x * 1000

def generate_class_labels():
    labels_path = os.path.join('trained_models', helpers.setup_json['subject_classifier_labels_path'])
    if os.path.exists(labels_path):
        print("Loading class labels...")
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
    else:
        print("Generating class labels...")
        docs = table_docs.get_docs_with_names(['Judgment'])
        cases = [table_cases.get_case_for_doc(doc) for doc in docs]
        Y = [case['subject'] for case in cases]
        setY = set(Y)
        labels = dict(zip(setY, range(len(setY))))
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f)
    return labels

def build_pipeline_bernoulli_nb(vectorizer):
    pipeline = make_pipeline(vectorizer, BernoulliNB())
    return pipeline

def build_pipeline_multinomial_nb(vectorizer):
    pipeline = make_pipeline(vectorizer, FunctionTransformer(_multiply_func, accept_sparse=True), MultinomialNB())
    return pipeline

def train_nb(model_name, vectorizer_name, labels_dict, test_examples=5):
    """Train Naive Bayes classifier on case content and subject labels.
    Input params:
    model_name: either bernoulli_nb or multinomial_nb.
    vectorizer_name: count or tfidf weights.
    labels_dict: label dictionary.
    test_examples: how many examples to leave out.
    """
    assert model_name in ['bernoulli_nb', 'multinomial_nb']
    assert vectorizer_name in ['count', 'tfidf']

    print("Building pipeline...")
    if vectorizer_name == 'count':
        vectorizer = CountVectorizer(tokenizer=_identity_func, preprocessor=_identity_func, lowercase=False, stop_words='english')
    else: # if vectorizer_name == 'tfidf'
        vectorizer = TfidfVectorizer(tokenizer=_identity_func, preprocessor=_identity_func, lowercase=False, stop_words='english')

    if model_name == 'bernoulli_nb':
        pipeline = build_pipeline_bernoulli_nb(vectorizer)
    else: # if model_name == 'multinomial_nb'
        pipeline = build_pipeline_multinomial_nb(vectorizer)

    print("Initializing database and loading features and labels...")
    docs = table_docs.get_docs_with_names(['Judgment'])
    cases = [table_cases.get_case_for_doc(doc) for doc in docs]

    content_gen = ContentGenerator(docs)
    contents = list(content_gen) # generate all contents at once
    contents = [chain.from_iterable(doc) for doc in contents]

    X_train = contents[:-test_examples]

    Y = [case['subject'] for case in cases]
    Y = [labels_dict.get(subject) for subject in Y]
    Y = [y for y in Y if y is not None] # filter out cases with nonexistent labels
    Y = np.asarray(Y)
    Y_train = Y[:-test_examples]

    print("Training model...")
    pipeline.fit(X_train, Y_train)

    print("Saving model...")
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    return pipeline

def validate_model(test_examples=5):
    """Validate model accuracy based on the predictions
    and true labels extracted from the database.
    """
    print("Initializing database and loading features and labels...")
    docs = table_docs.get_docs_with_names(['Judgment'])
    cases = [table_cases.get_case_for_doc(doc) for doc in docs]

    content_gen = ContentGenerator(docs)
    contents = list(content_gen) # generate all contents at once
    contents = [chain.from_iterable(doc) for doc in contents]

    X_test = contents[-test_examples:]

    Y = [case['subject'] for case in cases]
    Y = [labels_dict.get(subject) for subject in Y]
    Y = [y for y in Y if y is not None] # filter out cases with nonexistent labels
    Y = np.asarray(Y)
    Y_test = Y[-test_examples:]

    print("Loading model...")
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    print("Validating accuracy...")
    Y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    print('Classifier accuracy: {0}'.format(accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train subject classifier on CURIA documents')
    subparsers = parser.add_subparsers(dest='command')

    # train new classifier pipeline
    train_parser = subparsers.add_parser('train', help='Train new classifier pipeline')
    train_parser.add_argument('classifier', choices=['bernoulli_nb', 'multinomial_nb'], help='classifier to use')
    train_parser.add_argument('vectorizer', choices=['count', 'tfidf'], help='vectorizer to use')

    # validate existing pipeline
    validate_parser = subparsers.add_parser('validate', help='Validate existing classifier pipeline')

    args = parser.parse_args()

    # build class labels
    labels_dict = generate_class_labels()

    if args.command == 'train':
        train_nb(args.classifier, args.vectorizer, labels_dict)

    elif args.command == 'validate':
        validate_model()