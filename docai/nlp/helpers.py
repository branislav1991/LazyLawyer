from itertools import chain
import numpy as np

def combine_split_result(s):
    """Combines the result of a re.split splitting
    with the splitters retained. E.g. ['I', '/', 'am']
    -> ['I', '/am'].
    """
    if len(s) > 1:
        ss = [s[0]]
        for i in range(1, len(s)//2+1):
            ii = i-1
            ss.append(s[i] + s[i+1])
        return ss
    else:
        return s

def split_to_ngrams(word, n_min=3, n_max=6, max_ngram=10):
    """Splits word into various ngrams it contains.
    You can specify the minimum and maximum ngram
    size by n_min and n_max params. This function
    also returns the original word as part of the list.
    max_ngram specifies the maximal number of ngrams that
    are generated.
    """
    word = add_special_tags(word)

    word_ngrams = [word]
    if len(word) > n_min:
        for n in range(n_min, n_max+1):
            ngrams = [word[i:i+n] for i in range(0, len(word)-n+1)]
            word_ngrams.extend(ngrams)
    if len(word_ngrams) > max_ngram:
        word_ngrams = word_ngrams[:max_ngram]

    return word_ngrams

def add_special_tags(word):
    """Returns the word including special tags to
    distinguish between fasttext words and fasttext
    ngrams.
    """
    return '<' + word + '>'

def get_embedding_doc_lsi(content, model, dictionary, tfidf):
    """Obtains embedding for the document.
    Works with LSI models.
    """
    content_dict = dictionary.doc2bow(content)
    content_tfidf = tfidf[content_dict]
    embed = np.asarray(model[content_tfidf])
    return embed[:,1]

def get_embedding_doc_word2vec(content, model):
    """Obtains embedding for the whole document.
    Uses the dictionary built-in the model. Works 
    with word2vec and fasttext models.
    """
    embed = np.zeros((model.vector_size))
    words = chain.from_iterable(content)
    words = filter(lambda x: x in model.wv.vocab, words)
    for word in words:
        embed = embed + model.wv[word]
    return embed

def cosine_similarity(a, b):
        """Takes 2 vectors a, b and returns the cosine similarity according 
        to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if np.isclose(norm_a, 0) or np.isclose(norm_b, 0):
            return 0
        else:
            return dot_product / (norm_a * norm_b)