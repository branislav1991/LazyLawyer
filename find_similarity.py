from database import table_docs, table_doc_contents
import helpers
from nlp.word2vec_model import Word2Vec
from nlp.curia_preprocessor import remove_header
from nlp.curia_preprocessor import normalize
from nlp.curia_preprocessor import split_to_sentences
from nlp.curia_preprocessor import tokenize
from nlp.vocabulary import Vocabulary
from nlp.phrases import build_phrases_regex
import os
import Stemmer

stemmer = Stemmer.Stemmer('english')

save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

print('Loading vocabulary...')
vocabulary = Vocabulary()
vocabulary.load(save_path)

print('Loading model...')
model = Word2Vec(vocabulary)
model.load(save_path)

rules = [r"article \d+\w*"]

def process_document(doc):
    content = table_doc_contents.get_doc_content(doc)
    content = remove_header(content)
    content = normalize(content)
    content = split_to_sentences(content)

    return [tokenize(sent) for sent in content]

docs = table_docs.get_docs_with_name('Judgment')

doc1 = process_document(docs[5000])
#doc1 = stemmer.stemWords(doc1)
doc1 = build_phrases_regex([doc1], rules=rules)[0]
doc1 = [sent for sent in doc1 if len(sent) > 1] # only longer phrases are relevant

doc2 = process_document(docs[100])
#doc2 = stemmer.stemWords(doc2)
doc2 = build_phrases_regex([doc2], rules=rules)[0]
doc2 = [sent for sent in doc2 if len(sent) > 1] # only longer phrases are relevant

sim = model.doc_similarity(doc1, doc2, strategy='tf-idf')

print('Similarity: {0}'.format(sim))
