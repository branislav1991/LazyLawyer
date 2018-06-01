import re

def build_phrases_regex(documents, rules):
    """Builds phrases from tokens supplied by documents.
    Input params:
    documents: iterable of documents containing sentences,
    rules: regex rules to apply to find phrases.
    """
    # try regex rules on bigrams and connect them if rules apply
    phrases_documents = []
    for sentences in documents:
        phrases_sentences = []
        for words in sentences:
            last_word = None
            phrases = []
            for word in words:
                matched = False
                if last_word is not None:
                    phrase = last_word + ' ' + word # this can sometimes reach across sentences
                    matched = False
                    for rule in rules:
                        if re.match(rule, phrase):
                            phrases.append(phrase)
                            matched = True
                            break
                    if not matched:
                        phrases.append(last_word)
                if matched:
                    last_word = None
                else:
                    last_word = word
            phrases_sentences.append(phrases)
        phrases_documents.append(phrases_sentences)
    return phrases_documents