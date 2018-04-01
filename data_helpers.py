import numpy as np
import pandas as pd
import re
import os
import pickle
import itertools
from collections import Counter


class DataHelper(object):
    def __init__(self, filepath_input, filepath_glove):
        self.filepath_input = filepath_input
        self.filepath_glove = filepath_glove


    def clean_str(self, text):
        text = re.sub(r"http\S+", " url ", text)
        text = re.sub(r"@[A-Za-z0-9]+", "", text)
        text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'[\?\.\!\-\,]+(?=[\?\.\!\-\,])', '', text)
        text = re.sub(r"[^A-Za-z0-9^.,!^+/:;'-]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"omgee", "omg", text)
        text = re.sub(r"ifthe", "if the", text)
        text = re.sub(r"wiyh", "with", text)
        text = re.sub(r"whch", "which", text)
        text = re.sub(r"hella", "hello", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ", text)
        text = re.sub(r"\+", " ", text)
        text = re.sub(r"\-", " ", text)
        text = re.sub(r"\=", " ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r":", " ", text)
        text = re.sub(r";", " ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"-", " ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r'\b\w\b', ' ', text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub("\d+", "numeric", text)

        return text.strip().lower()


    def load_data_and_labels(self):
        """
        Loads data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        data_df = pd.read_csv(self.filepath_input, encoding='latin1')
        data_df['Tag'].fillna("None", inplace=True)

        y_tag = pd.get_dummies(data_df['Tag'], prefix="tag").values
        y_sentiment = pd.get_dummies(data_df['airline_sentiment'], prefix="airline_sentiment").values

        data_df['airline'] = data_df['airline'].str.replace(' ', '')
        f = lambda text: self.clean_str(text).split(' ')
        text_df = (data_df["airline"] + " " + data_df["Text"]).map(f)
        #np.savetxt('tweets.txt', text_df.values, fmt='%s', newline='\n')

        return text_df.values, y_tag, y_sentiment



    def pad_sentences(self, sentences, padding_word="<PAD/>"):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        sequence_length = max(len(x) for x in sentences)
        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        return padded_sentences, sequence_length


    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]



    def build_input_data(self, sentences, y_tag, y_sentiment, vocabulary):
        """
        Maps sentencs and labels to vectors based on a vocabulary.
        """
        x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
        y_tag = np.array(y_tag)
        y_sentiment = np.array(y_sentiment)
        return [x, y_tag, y_sentiment]

    def read_data(self, filename):
        """Extract the first file enclosed in a zip file as a list of words."""
        with open(filename, "r") as f:
            data = f.read().split()
        return list(set(data))

    def load_glove_embeddings(self, filepath_glove):
        glove_dict = os.path.abspath(os.path.join(os.path.curdir, "embedding_dict.p"))
        if os.path.exists(glove_dict):
            embedding_dict = pickle.load(open("embedding_dict.p", "rb"))
            return embedding_dict

        glove_vocab = []
        embedding_dict = {}

        file = open(filepath_glove, 'r', encoding='UTF-8')
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab_word = row[0]
            #glove_vocab.append(vocab_word)
            embed_vector = [float(i) for i in row[1:]]  # convert to list of float
            embedding_dict[vocab_word] = embed_vector
        file.close()

        print('Loaded GLOVE')

        return embedding_dict

    def build_embedding_matrix(self, embedding_dict, vocabulary):
        embedding_file = os.path.abspath(os.path.join(os.path.curdir, "embedding.p"))
        if os.path.exists(embedding_file):
            embedding = pickle.load(open("embedding.p", "rb"))
            return embedding

        glove_vocab = list(embedding_dict.keys())
        glove_vocab_size = len(glove_vocab)
        embedding_dim = len(embedding_dict[glove_vocab[0]])

        embeddings_tmp = []
        doc_vocab_size = len(vocabulary)
        dict_as_list = sorted(vocabulary.items(), key=lambda x: x[1])
        for i in range(doc_vocab_size):
            item = dict_as_list[i][0]
            if item in glove_vocab:
                embeddings_tmp.append(embedding_dict[item])
            else:
                rand_num = np.random.uniform(low=-0.2, high=0.2, size=embedding_dim)
                embeddings_tmp.append(rand_num)

        # final embedding array corresponds to dictionary of words in the document
        embedding = np.asarray(embeddings_tmp)
        pickle.dump(embedding, open("embedding.p", "wb"))
        return embedding

    def load_data(self):
        """
        Loads and preprocessed data for the dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        sentences, y_tag, y_sentiment = self.load_data_and_labels()
        sentences_padded, sequence_length = self.pad_sentences(sentences)
        vocabulary, vocabulary_inv = self.build_vocab(sentences_padded)
        embedding_dict= self.load_glove_embeddings(self.filepath_glove)
        embedding_mat = self.build_embedding_matrix(embedding_dict, vocabulary)
        x, y_tag, y_sentiment = self.build_input_data(sentences_padded, y_tag, y_sentiment, vocabulary)
        return [x, y_tag, y_sentiment, vocabulary, vocabulary_inv, sequence_length, embedding_mat]

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """Iterate the data batch by batch"""
        data = np.array(data)
        data_size = data.shape[0]

        num_batches_per_epoch = int(data_size / batch_size) + 1

        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]



