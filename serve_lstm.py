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
    vocab, generated_embeddings = preprocess_helper.load_frequent_words_and_embeddings("data/vocab_with_emb.txt")
    print("- Load checkpoint")
    checkpoint_file = tf.train.latest_checkpoint("model/pretrained_w2v_hidden_layer_512/checkpoints/")
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
            discard_last_prediction = graph.get_operation_by_name("discard_last_prediction").outputs[0]

            # Tensors we want to evaluate
            probabilities = graph.get_operation_by_name("softmax_layer/Reshape_2").outputs[0]

            print("- Model successfully restored")
            # Construct the embedding matrix
            vocab_emb = np.zeros(shape=(20000, 100))
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format("data/wordembeddings-dim100.word2vec", binary=False)
            for tok, idx in vocab.items():
              if tok in w2v_model.vocab:
                vocab_emb[idx] = w2v_model[tok]
              else:
                  vocab_emb[idx] = generated_embeddings[tok]
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
        input_data = input_data.lower()
        input_data_token = preprocess_helper.replace_unknown_words([input_data], vocab)
        input_data_token, _ = preprocess_helper.add_tokens_to_sentences(input_data_token,
                                                                        vocab,
                                                                        30,
                                                                        eos_token=False,
                                                                        pad_sentence=False)

        continuation_length = 30
        for cont in range(continuation_length):
            all_probabilities = sess.run([probabilities], {inputs: input_data_token,
                                                           vocab_embedding: vocab_emb,
                                                           discard_last_prediction: False})
            all_probabilities = np.squeeze(all_probabilities)
            all_probabilities = all_probabilities[-1, :]

            # artificially set to zero the proba of the token <unk>
            all_probabilities[19996] = 0

            # sort and take the value of Nth largest one...
            n = 20000 - 10
            sorted_proba = np.sort(all_probabilities)
            thresh = sorted_proba[n]
            all_probabilities[np.abs(all_probabilities) < thresh] = 0

            sum_all_probabilities = np.sum(all_probabilities)
            all_probabilities = all_probabilities / sum_all_probabilities
            predicted_word = np.random.choice(20000, 1, p=all_probabilities)

            input_data_token = np.concatenate((input_data_token, [predicted_word]), axis=1)

        sentence = ''
        for i in range(len(input_data_token[0])):
            word = (list(vocab.keys())[list(vocab.values()).index(input_data_token[0][i])])
            sentence += word
            sentence += ' '

        # remove all the brackets sign
        sentence = sentence.replace("<bos>", "")
        sentence = sentence.replace("<eos>", "")
        sentence = sentence.replace("<pad>", "")
        sentence = sentence.replace("<", "")
        sentence = sentence.replace(">", "")

        print(sentence)

        # 5. return the output for the api
        return sentence

    return model_api
