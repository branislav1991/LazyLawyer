import regex
import spacy

nlp = spacy.load('en_core_web_lg')

def similarity(content1, content2):
    doc1 = nlp(content1)
    doc2 = nlp(content2)

    return doc1.similarity(doc2)