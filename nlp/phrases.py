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
        phrases = []
        if len(sentence) < 2:
            phrases = sentence
        else:
            last_word = sentence[0]
            i = 1
            while i < len(sentence):
                matched = False
                word = sentence[i]
                last_word = sentence[i-1]
                phrase = last_word + ' ' + word
                for rule in rules: # try to match all rules
                    if re.match(rule, phrase):
                        phrases.append(phrase)
                        i += 2
                        matched = True
                        break
                if not matched:
                    phrases.append(last_word)
                    i += 1

        phrases_document.append(phrases)
    return phrases_document