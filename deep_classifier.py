import tensorflow as tf


class DeepLSTMClassifier(object):
    def stacked_LSTM(self, x, dropout_keep_prob, seq_length, embedding_dim, n_hidden, n_layers):
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))

        with tf.name_scope("stacked_lstm"):
            stacked_lstm = []
            for _ in range(n_layers):
                cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.1, state_is_tuple=True)
                lstm_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
                stacked_lstm.append(lstm_cell)

        lstm_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_lstm, state_is_tuple=True)
        output, _ = tf.nn.static_rnn(lstm_cell_m, x, dtype=tf.float32)

        return output[-1]

    def __init__(self, seq_length, n_tags, n_sentiments, vocab_size, embedding_dim, n_hidden, n_layers,
                 dropout_keep_prob, batch_size, l2_reg_lambda,
                 trainable_embeddings):
        self.input_x = tf.placeholder(tf.int32, shape=[None, seq_length], name="input_x")
        self.input_y_tag = tf.placeholder(tf.int32, shape=[None, n_tags], name="input_y")
        self.input_y_sentiment = tf.placeholder(tf.int32, shape=[None, n_sentiments], name="input_y")
        
        l2_loss = tf.constant(0.0)

        with tf.name_scope("embeddings"):
            self.W = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[vocab_size, embedding_dim]),
                                 trainable=trainable_embeddings, name="W")
            self.embedded_words = tf.nn.embedding_lookup(self.W, self.input_x)

        with tf.name_scope("stacked_lstm"):
            self.stacked_lstm_out = self.stacked_LSTM(self.embedded_words, dropout_keep_prob, seq_length,
                                                      embedding_dim,
                                                      n_hidden, n_layers)

        with tf.name_scope("output_tag"):
            self.W1 = tf.Variable(tf.truncated_normal(shape=[n_hidden, n_tags], stddev=0.1))
            b1 = tf.Variable(tf.truncated_normal(shape=[n_tags], stddev=0.1))
            self.y_tag_out = tf.nn.xw_plus_b(self.stacked_lstm_out, self.W1, b1, name="scores")
            l2_loss += tf.nn.l2_loss(self.W1)
            l2_loss += tf.nn.l2_loss(b1)

        with tf.name_scope("output_sentiment"):
            self.W2 = tf.Variable(tf.truncated_normal(shape=[n_hidden, n_sentiments], stddev=0.1))
            b2 = tf.Variable(tf.truncated_normal(shape=[n_sentiments], stddev=0.1))
            self.y_sentiment_out = tf.nn.xw_plus_b(self.stacked_lstm_out, self.W2, b2, name="scores")
            l2_loss += tf.nn.l2_loss(self.W2)
            l2_loss += tf.nn.l2_loss(b2)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y_tag, logits=self.y_tag_out)) + \
                        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y_sentiment,
                                                        logits=self.y_sentiment_out)) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.correct_prediction_tag = tf.equal(tf.argmax(self.y_tag_out, 1), tf.argmax(self.input_y_tag, 1))
            self.accuracy_tag = tf.reduce_mean(tf.cast(self.correct_prediction_tag, tf.float32), name="accuracy")

            self.correct_prediction_sentiment = tf.equal(tf.argmax(self.y_sentiment_out, 1),
                                                    tf.argmax(self.input_y_sentiment, 1))
            self.accuracy_sentiment = tf.reduce_mean(tf.cast(self.correct_prediction_sentiment, tf.float32), name="accuracy")

        with tf.name_scope('num_correct'):
            correct_tag = tf.equal(tf.argmax(self.y_tag_out, 1), tf.argmax(self.input_y_tag, 1))
            self.num_correct_tag = tf.reduce_sum(tf.cast(correct_tag, 'float'))

            correct_sentiment = tf.equal(tf.argmax(self.y_sentiment_out, 1), tf.argmax(self.input_y_sentiment, 1))
            self.num_correct_sentiment = tf.reduce_sum(tf.cast(correct_sentiment, 'float'))

