from docai import helpers
from docai.nlp.vocabulary import Vocabulary
import os

word = 'arbitr'
save_path = os.path.join('trained_models', helpers.setup_json['model_path'])

if __name__ == '__main__':
    print('Loading vocabulary...')
    vocabulary = Vocabulary()
    vocabulary.load(save_path)

    idx = vocabulary.get_index(word)
    if idx == -1:
        print('Word not in vocabulary')
    else:
        idf = vocabulary.get_idf_weight(word)
        print('Word in vocabulary; idf = {0}'.format(idf))