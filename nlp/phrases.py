import re

# a set of default rules for law-related phrases
default_rules = [r"articl \d+\w*",
        r"paragraph \d+\w*",
        r"law no",
        r"law no \d+\w*",
        r"direct \d+\w*",
        r"^((31(?!\ (feb(ruary)?|apr(il)?|june?|(sep(?=\b|t)t?|nov)(emb)?)))|((30|29)(?!\ feb(ruary)?))|(29(?=\ feb(ruary)?\ (((1[6-9]|[2-9]\d)(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00)))))|(0?[1-9])|1\d|2[0-8])\ (jan(uary)?|feb(ruary)?|ma(r(ch)?|y)|apr(il)?|ju((li?)|(ne?))|aug(ust)?|oct(ob)?|(sep(?=\b|t)t?|nov|dec)(emb)?)$",
        r"^((31(?!\ (feb(ruary)?|apr(il)?|june?|(sep(?=\b|t)t?|nov)(emb)?)))|((30|29)(?!\ feb(ruary)?))|(29(?=\ feb(ruary)?\ (((1[6-9]|[2-9]\d)(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00)))))|(0?[1-9])|1\d|2[0-8])\ (jan(uary)?|feb(ruary)?|ma(r(ch)?|y)|apr(il)?|ju((li?)|(ne?))|aug(ust)?|oct(ob)?|(sep(?=\b|t)t?|nov|dec)(emb)?)\ ((1[6-9]|[2-9]\d)\d{2})$"]


def build_phrases_regex(document, rules=default_rules):
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
