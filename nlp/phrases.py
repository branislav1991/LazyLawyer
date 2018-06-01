import re

def build_phrases_regex(document, rules):
    """Builds phrases from tokens supplied by document.
    Input params:
    document: document containing sentences,
    rules: regex rules to apply to find phrases.
    """
    # try regex rules on bigrams and connect them if rules apply
    phrases_document = []
    for sentence in document:
        last_word = None
        phrases = []
        for word in sentence:
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
        phrases_document.append(phrases)
    return phrases_document