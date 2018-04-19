import preprocess_helper

import tensorflow as tf
import numpy as np

import gensim

#  Parameters

# Data loading parameters
tf.flags.DEFINE_string("data_file_path", "data/sentences.continuation", "Path to the training data")
tf.flags.DEFINE_string("vocab_with_emb_path", "data/vocab_with_emb.txt", "Path to the vocabulary list")
tf.flags.DEFINE_string("checkpoint_dir", "model/pretrained_w2v_hidden_layer_512/checkpoints/", "Checkpoint directory from training run")

# Model parameters
tf.flags.DEFINE_integer("embedding_dimension", 100, "Dimensionality of word embeddings")
tf.flags.DEFINE_integer("vocabulary_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_integer("sentence_length", 30, "Length of the sentence to create")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")

# Embedding parameters
tf.flags.DEFINE_boolean("use_word2vec_emb", True, "Use word2vec embedding")
tf.flags.DEFINE_string("path_to_word2vec", "data/wordembeddings-dim100.word2vec", "Path to the embedding file")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("verbose_for_debugging", False, "Allow info to be printed to understand the behaviour of the network")
tf.flags.DEFINE_boolean("verbose_for_experiments", True, "Print only the predicted sentence")
FLAGS = tf.flags.FLAGS

# Prepare data

# Load data
print("Load vocabulary list \n")
vocab, generated_embeddings = preprocess_helper.load_frequent_words_and_embeddings(FLAGS.vocab_with_emb_path)

print("Loading and preprocessing test dataset \n")
x_test, y_test = preprocess_helper.load_and_process_data(FLAGS.data_file_path,
                                                         vocab,
                                                         FLAGS.sentence_length,
                                                         eos_token=False,
                                                         pad_sentence=False)
## EVALUATION ##

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        inputs = graph.get_operation_by_name("inputs").outputs[0]
        vocab_embedding = graph.get_operation_by_name("vocab_embedding").outputs[0]
        discard_last_prediction = graph.get_operation_by_name("discard_last_prediction").outputs[0]

        # Tensors we want to evaluate
        predicted_sentence = graph.get_operation_by_name("continuation/predicted_sentence").outputs[0]
        probabilities = graph.get_operation_by_name("softmax_layer/Reshape_2").outputs[0]

        # Generate batches for one epoch
        batches = preprocess_helper.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=False)

        # Construct the embedding matrix
        vocab_emb = np.zeros(shape=(FLAGS.vocabulary_size, FLAGS.embedding_dimension))
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.path_to_word2vec, binary=False)
        for tok, idx in vocab.items():
            if FLAGS.use_word2vec_emb and tok in w2v_model.vocab:
                vocab_emb[idx] = w2v_model[tok]
            else:
                vocab_emb[idx] = generated_embeddings[tok]

        # Collect the predictions here
        all_perplexity = []

        for batch_id, batch in enumerate(batches):
            if batch_id > 10:
                break
            x_batch, y_batch = zip(*batch)

            continuation_length = 30
            for cont in range(continuation_length):

                all_probabilities = sess.run([probabilities], {inputs: x_batch,
                                                               vocab_embedding: vocab_emb,
                                                               discard_last_prediction: False})
                all_probabilities = np.squeeze(all_probabilities)
                all_probabilities = all_probabilities[-1, :]

                # artificially set to zero the proba of the token <unk>
                all_probabilities[19996] = 0

                # sort and take the value of Nth largest one...
                n = FLAGS.vocabulary_size - 10
                sorted_proba = np.sort(all_probabilities)
                thresh = sorted_proba[n]
                all_probabilities[np.abs(all_probabilities) < thresh] = 0

                sum_all_probabilities = np.sum(all_probabilities)
                all_probabilities = all_probabilities / sum_all_probabilities

                # extract the 100 largest proba
                argmax_prediction = np.argmax(all_probabilities)
                predicted_word = np.random.choice(FLAGS.vocabulary_size, 1, p=all_probabilities)

                new = x_batch[0]
                new.append(predicted_word)

            sentence = ''
            for i in range(len(x_batch[0])):
                word = (list(vocab.keys())[list(vocab.values()).index(x_batch[0][i])])
                sentence += word
                sentence += ' '

            # remove all the brackets sign
            sentence = sentence.replace("<bos>", "")
            sentence = sentence.replace("<eos>", "")
            sentence = sentence.replace("<pad>", "")
            sentence = sentence.replace("<", "")
            sentence = sentence.replace(">", "")

            print(sentence)


