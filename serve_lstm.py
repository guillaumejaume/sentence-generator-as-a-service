import preprocess_helper

import tensorflow as tf
import numpy as np

import gensim

def get_model_api():
    """Returns lambda function for api"""

    # 1. initialize model once and for all
    print("- Load vocabulary list")
    vocabulary_size = 20000
    embedding_dimension = 100
    use_word2vec_emb = True
    vocab = preprocess_helper.load_frequent_words("data/k_frequent_words.txt")
    print("- Load checkpoint")
    checkpoint_file = tf.train.latest_checkpoint("model/1522325038/checkpoints/")
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            print("- Restore the model")
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            inputs = graph.get_operation_by_name("inputs").outputs[0]
            vocab_embedding = graph.get_operation_by_name("vocab_embedding").outputs[0]

            # Tensors we want to evaluate
            predicted_sentence = graph.get_operation_by_name("continuation/predicted_sentence").outputs[0]
            print("- Model successfully restored")
            # Construct the embedding matrix
            vocab_emb = np.zeros(shape=(vocabulary_size, embedding_dimension))
            model = gensim.models.KeyedVectors.load_word2vec_format("data/wordembeddings-dim100.word2vec", binary=False)
            for tok, idx in vocab.items():
                if use_word2vec_emb and tok in model.vocab:
                    vocab_emb[idx] = model[tok]
                else:
                    vocab_emb[idx] = np.random.uniform(low=-1, high=1, size=embedding_dimension)
            print("- Embedding done")

    def model_api(input_data):
        """
        Args:
            input_data: submitted to the API, raw string

        Returns:
            output_data: after some transformation, to be
                returned to the API

        """
        # 2. process input
        input_data_token = preprocess_helper.replace_unknown_words([input_data], vocab)
        input_data_token, _ = preprocess_helper.add_tokens_to_sentences(input_data_token,
                                                                        vocab,
                                                                        30,
                                                                        eos_token=False,
                                                                        pad_sentence=False)

        # 3. call model predict function
        predicted_sentence_batch = sess.run([predicted_sentence], {inputs: input_data_token,
                                                                   vocab_embedding: vocab_emb})

        # 4. process the output
        prediction_sentence = ''
        for i in range(len(predicted_sentence_batch[0][0])):
            word = (list(vocab.keys())[list(vocab.values()).index(predicted_sentence_batch[0][0, i])])
            prediction_sentence += word
            prediction_sentence += ' '

        # remove all the brackets sign
        prediction_sentence = prediction_sentence.replace("<", "")
        prediction_sentence = prediction_sentence.replace(">", "")
        
        # 5. return the output for the api
        return prediction_sentence

    return model_api
