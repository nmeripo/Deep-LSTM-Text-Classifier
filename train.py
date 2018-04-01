import tensorflow as tf
from chatdesk.deep_classifier import DeepLSTMClassifier
from chatdesk.data_helpers import DataHelper
import time
import datetime
import os
from sklearn.model_selection import train_test_split
import shutil

"""load data"""
data_helper = DataHelper(filepath_input="./Airline-Tags (1).csv", filepath_glove='./glove.twitter.27B.100d.txt')
x, y_tag, y_sentiment, vocabulary, vocabulary_inv, sequence_length, embeddings = data_helper.load_data()

"""split the original dataset into train and test sets"""
x_, x_test, y_, y_test = train_test_split(x, list(zip(y_tag, y_sentiment)), test_size=0.1, random_state=42)

"""split the train set into train and dev sets"""
x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.3)
y_train_tag, y_train_sentiment = zip(*y_train)
y_dev_tag, y_dev_sentiment = zip(*y_dev)
y_test_tag, y_test_sentiment = zip(*y_test)

"model params"
trainable_embeddings = True
dropout_keep_prob = 0.4
batch_size = 1024
num_epochs = 20
evaluate_every = len(y_train_tag) // batch_size
checkpoint_every = 100

timestamp = str(int(time.time()))
trained_dir = './trained_results_' + timestamp + '/'
if os.path.exists(trained_dir):
    shutil.rmtree(trained_dir)
os.makedirs(trained_dir)

with tf.Graph().as_default():
    sess = tf.Session()
    print("started session")

    with sess.as_default():
        deepLSTMModel = DeepLSTMClassifier(
            seq_length=sequence_length,
            vocab_size=len(vocabulary),
            embedding_dim=100,
            n_hidden=50,
            n_layers=1,
            dropout_keep_prob=dropout_keep_prob,
            batch_size=batch_size,
            trainable_embeddings=trainable_embeddings,
            n_tags=y_tag.shape[1],
            n_sentiments=y_sentiment.shape[1],
            l2_reg_lambda=0.2
        )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("Initialized DeepLSTMModel")

    grads_and_vars = optimizer.compute_gradients(deepLSTMModel.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("Defined training_ops")

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    print("Defined gradient summaries")

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", deepLSTMModel.loss)
    acc_tag_summary = tf.summary.scalar("accuracy_tag", deepLSTMModel.accuracy_tag)
    acc_sentiment_summary = tf.summary.scalar("accuracy_sentiment", deepLSTMModel.accuracy_sentiment)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_tag_summary, acc_sentiment_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_tag_summary, acc_sentiment_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = './checkpoints_' + timestamp + '/'
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')


    def train_step(x_batch, y_tag_batch, y_sentiment_batch):
        feed_dict = {
            deepLSTMModel.input_x: x_batch,
            deepLSTMModel.input_y_tag: y_tag_batch,
            deepLSTMModel.input_y_sentiment: y_sentiment_batch,
        }

        _, step, loss, accuracy_tag, accuracy_sentiment, summaries = sess.run(
            [tr_op_set, global_step, deepLSTMModel.loss, deepLSTMModel.accuracy_tag, deepLSTMModel.accuracy_sentiment,
             train_summary_op], feed_dict)

        time_str = datetime.datetime.now().isoformat()
        train_summary_writer.add_summary(summaries, step)
        return loss, accuracy_tag, accuracy_sentiment, time_str


    def dev_step(x_batch, y_tag_batch, y_sentiment_batch):
        feed_dict = {
            deepLSTMModel.input_x: x_batch,
            deepLSTMModel.input_y_tag: y_tag_batch,
            deepLSTMModel.input_y_sentiment: y_sentiment_batch,
        }

        step, loss, num_correct_tag, num_correct_sentiment, summaries = sess.run(
            [global_step, deepLSTMModel.loss, deepLSTMModel.num_correct_tag, deepLSTMModel.num_correct_sentiment,
             dev_summary_op], feed_dict)

        time_str = datetime.datetime.now().isoformat()
        dev_summary_writer.add_summary(summaries, step)

        return loss, num_correct_tag, num_correct_sentiment, time_str


    print("Initialize all variables")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(deepLSTMModel.W.assign(embeddings))


    # Generate batches

    train_batches = data_helper.batch_iter(list(zip(x_train, y_train_tag, y_train_sentiment)), batch_size, num_epochs)

    best_accuracy_tag, best_accuracy_sentiment, best_accuracy = 0.0, 0.0, 0

    for train_batch in train_batches:
        x_train_batch, y_train_tag_batch, y_train_sentiment_batch = zip(*train_batch)
        loss, accuracy_tag, accuracy_sentiment, time_str = train_step(x_train_batch, y_train_tag_batch, y_train_sentiment_batch)
        current_step = tf.train.global_step(sess, global_step)

        if current_step % evaluate_every == 0:
            print(
                "TRAIN {}: step {}, loss {:g}, acc_tag {:g}, acc_sentiment {:g}".format(time_str, current_step, loss,
                                                                                        accuracy_tag,
                                                                                        accuracy_sentiment))
            dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev_tag, y_dev_sentiment)), batch_size, 1)
            total_dev_correct_tag, total_dev_correct_sentiment = 0, 0
            for dev_batch in dev_batches:
                x_dev_batch, y_dev_tag_batch, y_dev_sentiment_batch = zip(*dev_batch)
                loss, num_dev_correct_tag, num_dev_correct_sentiment, time_str = dev_step(x_dev_batch,
                                                                                          y_dev_tag_batch,
                                                                                          y_dev_sentiment_batch)
                total_dev_correct_tag += num_dev_correct_tag
                total_dev_correct_sentiment += num_dev_correct_sentiment

            dev_accuracy_tag = float(total_dev_correct_tag) / len(y_dev_tag)
            dev_accuracy_sentiment = float(total_dev_correct_sentiment) / len(y_dev_sentiment)

            print(
                "EVAL {}: loss {:g}, acc_tag {:g}, acc_sentiment {:g}".format(time_str, loss,
                                                                              dev_accuracy_tag,
                                                                              dev_accuracy_sentiment))
            print("")

            """Save the model if it is the best based on accuracy on dev set"""
            if dev_accuracy_tag >= best_accuracy_tag:
                best_accuracy_tag, best_at_step_tag = dev_accuracy_tag, current_step
                path = saver.save(sess, checkpoint_prefix + "_tag", global_step=current_step)
                print('Saved model at {} at step {}'.format(path, best_at_step_tag))
                print('Best tag accuracy is {} at step {}'.format(best_accuracy_tag, best_at_step_tag))
                print()

            if dev_accuracy_sentiment >= best_accuracy_sentiment:
                best_accuracy_sentiment, best_at_step_sentiment = dev_accuracy_sentiment, current_step
                path = saver.save(sess, checkpoint_prefix + "_sentiment", global_step=current_step)
                print('Saved model at {} at step {}'.format(path, best_at_step_sentiment))
                print('Best sentiment accuracy is {} at step {}'.format(best_accuracy_sentiment, best_at_step_sentiment))
                print()

    # Save the model files to trained_dir.
    saver.save(sess, trained_dir + "best_model.ckpt")

    # Evaluate x_test and y_test
    print(checkpoint_prefix + "_tag" + '-' + str(best_at_step_sentiment))
    saver.restore(sess, checkpoint_prefix + "_tag" + '-' + str(best_at_step_sentiment))

    """Predict x_test (batch by batch)"""
    test_batches = data_helper.batch_iter(list(zip(x_test, y_test_tag, y_test_sentiment)), batch_size, 1)
    total_test_correct_tag, total_test_correct_sentiment = 0, 0
    for test_batch in test_batches:
        x_test_batch, y_test_tag_batch, y_test_sentiment_batch = zip(*test_batch)
        loss, num_test_correct_tag, num_test_correct_sentiment, time_str = dev_step(x_test_batch, y_test_tag_batch, y_test_sentiment_batch)
        total_test_correct_tag += num_test_correct_tag
        total_test_correct_sentiment += num_test_correct_sentiment

    test_accuracy_tag = float(total_test_correct_tag) / len(y_test_tag)
    test_accuracy_sentiment = float(total_test_correct_sentiment) / len(y_test_sentiment)

    print(
        "TEST {}: loss {:g}, acc_tag {:g}, acc_sentiment {:g}".format(time_str, loss,
                                                                      test_accuracy_tag,
                                                                      test_accuracy_sentiment))
