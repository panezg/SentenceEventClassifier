import logging
import os
from datetime import date, timedelta
from itertools import islice

import gensim
import nltk
import numpy
from gensim.models import Word2Vec
from nltk.corpus import wordnet as wn

directory_root = '/Users/gpanez/Documents/news/the_guardian'
directory_preprocessed_output = '/Users/gpanez/Documents/news/the_guardian_preprocessed'
directory_preprocessed_sentences_output = '/Users/gpanez/Documents/news/the_guardian_preprocessed_sentences'


class RequestError(Exception):
    pass


class Classifier:
    def __init__(self):
        self.wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        self.wv.init_sims(replace=True)
        print(list(islice(self.wv.vocab, 13030, 13050)))

    def w2v_tokenize_text(self, text):
        tokens = []
        for sent in nltk.sent_tokenize(text, language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                if len(word) < 2:
                    continue
                tokens.append(word)
        return tokens

    def word_averaging(self, words):
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in self.wv.vocab:
                mean.append(self.wv.syn0norm[self.wv.vocab[word].index])
                all_words.add(self.wv.vocab[word].index)

        if not mean:
            logging.warning("cannot compute similarity with no input %s", words)
            # FIXME: remove these examples in pre-processing
            return np.zeros(wv.vector_size, )

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def word_averaging_list(self, text_list):
        return np.vstack([word_averaging(self.wv, post) for post in text_list])

    def classify(self, body_text):
        train, test = train_test_split(df, test_size=0.3, random_state=42)

        test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['post']), axis=1).values
        train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['post']), axis=1).values

        X_train_word_average = word_averaging_list(train_tokenized)
        X_test_word_average = word_averaging_list(test_tokenized)


def process_items():
    start = date(2000, 1, 1)
    end = date(2018, 12, 31)

    current = start
    delta = timedelta(days=1)
    while current <= end:
        directory = directory_preprocessed_output + '/' + current.isoformat()

        if not os.path.exists(directory):
            print('Directory does not exist: ', directory)
        else:
            for file_name in os.listdir(directory):
                with open(directory + '/' + file_name) as file:
                    lines = file.readlines()
                    count = 0
                    header = ''
                    body_text = ''
                    for line in lines:
                        if count == 4:
                            body_text += line + '\n'
                        else:
                            header += line + '\n'
                            if line == "#####":
                                count += 1
                    sentences_text = classify(body_text)

                    directory_output = directory_preprocessed_sentences_output + '/' + current.isoformat()
                    if not os.path.exists(directory_output):  # or not os.path.isdir(directory):
                        logging.debug('Creating directory: [%s]', directory_output)
                        os.makedirs(directory_output)

                    with open(directory_output + '/' + file_name, "w") as file_out:
                        # file_out.write(header + sentences_text)
                        pass

        current = current + delta
    print("done")


class LemmaExplorer:
    def __init__(self):
        self.triggers = ["resign", "run", "announce", "challenge", "promote", "demote", "exit",
                         "sign", "agreement", "vote", "quit", "condemn", "leave", "appear",
                         "hold", "refuse", "win", "lose", "statement", "appointed", "speech",
                         "approve", "negotiate", "reject", "refuse", "dismiss", "nomination",
                         "commission", "publish", "launch", "march", "endorse", "debate",
                         "stand", "express", "elect", "form", "open", "fundrise"]
        self.triggerLemmas = set()

    def get_lemmas(self):
        for trigger in self.triggers:
            synsets = wn.synsets(trigger)
            for synset in synsets:
                lemmas = [l.name() for l in synset.lemmas()]
                for lemma in lemmas:
                    self.triggerLemmas.add(lemma)
        return self.triggerLemmas


def main():
    logging.basicConfig(filename=directory_preprocessed_output + '/log.txt',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    logging.info("Sentence Event Classifier")
    logging.getLogger('SEC')
    le = LemmaExplorer()
    print(le.get_lemmas())
    # c = Classifier()
    # process_items()
    # TODO: Added timing thresholds but need to add limit per day
    # TODO: Added timing thresholds but need to add saving the queue
    # TODO: Need to add cron job, and overall begin and end


if __name__ == "__main__":
    main()
    print("done")
