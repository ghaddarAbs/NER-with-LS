import tensorflow as tf
import sys, pyhocon, joblib, time, os
import numpy as np
from batcher import Batcher


class Model:
    def __init__(self, config, emb_matrix, ls_matrix):

        self.keep_prob = tf.placeholder(tf.float32)

        # Define input and target tensors
        self.emb_input_data = tf.placeholder(tf.int32, [None, None], name="emb_input_data")
        self.ls_input_data = tf.placeholder(tf.int32, [None, None], name="ls_input_data")
        self.cap_input_data = tf.placeholder(tf.int32, [None, None], name="cap_input_data")
        self.target = tf.placeholder(tf.int32, [None, None], name="target")
        self.word_length = tf.placeholder(tf.int32, [None], name="word_length")

        if "chars" in config["features"]:
            self.char_input_data = tf.placeholder(tf.int32,
                                                  [None, None, config["char_step_num"]],
                                                  name="char_input_data")

            self.char_length = tf.placeholder(tf.int32, [None, None], name="char_length")

        if "emb" in config["features"]:
            word_embedding = tf.get_variable(name="word_embedding",
                                             shape=emb_matrix.shape,
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(emb_matrix),
                                             trainable=not config["freeze"])
            emb_input = tf.nn.embedding_lookup(word_embedding, self.emb_input_data)
            self.word_input = emb_input

        if "ls" in config["features"]:
            ls_embedding = tf.get_variable(name="ls_embedding",
                                             shape=ls_matrix.shape,
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(ls_matrix),
                                             trainable=False)

            ls_input = tf.nn.embedding_lookup(ls_embedding, self.ls_input_data)
            if hasattr(self, "word_input"):
                self.word_input = tf.concat([self.word_input, ls_input], -1)
            else:
                self.word_input = ls_input

        if "caps" in config["features"]:
            cap_embedding = tf.get_variable(name="cap_embedding",
                                            initializer=tf.random_uniform(
                                                [config["vocab_cap"], config["cap_dim"]],
                                                minval=-(3. / config["cap_dim"]) ** .5,
                                                maxval=(3. / config["cap_dim"]) ** .5),
                                            trainable=True)

            cap_input = tf.nn.embedding_lookup(cap_embedding, self.cap_input_data)
            if hasattr(self, "word_input"):
                self.word_input = tf.concat([self.word_input, cap_input], -1)
            else:
                self.word_input = cap_input

        if "chars" in config["features"]:
            char_input = self.char_cnn_model(config)
            if hasattr(self, "word_input"):
                self.word_input = tf.concat([self.word_input, char_input], -1)
            else:
                self.word_input = char_input

        fw_cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(config, config["lstm_word"], i)
                                               for i in range(config["num_layers"])], state_is_tuple=True)

        bw_cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(config, config["lstm_word"], i)
                                               for i in range(config["num_layers"])], state_is_tuple=True)

        with tf.variable_scope("word_rnn") as scope:
            output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                        self.word_input,
                                                        sequence_length=self.word_length,
                                                        parallel_iterations=1024,
                                                        swap_memory=False,
                                                        dtype=tf.float32)

        output = tf.concat(output, axis=-1)
        ntime_steps = tf.shape(output)[1]
        output = tf.reshape(output, [-1, 2 * config["lstm_word"]])

        self.logit = tf.layers.dense(output, config["target_num"], activation=None)
        self.logit = tf.reshape(self.logit, [-1, ntime_steps, config["target_num"]])

        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logit,
                                                                                   self.target,
                                                                                   self.word_length)

        self.loss = tf.reduce_mean(-log_likelihood)
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config["clip"])

        optimizer = tf.train.MomentumOptimizer (self._lr, 0.9)

        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def char_cnn_model(self, config):
        ntime_steps = tf.shape(self.char_input_data)[1]

        char_input = tf.reshape(self.char_input_data, [-1, config["char_step_num"]])
        char_embedding = tf.get_variable(name="char_embedding",
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform([config["vocab_char"],  config["char_dim"]],
                                                                       -0.5, 0.5),
                                         trainable=True)

        char_input = tf.nn.embedding_lookup(char_embedding, char_input)

        conv = tf.layers.conv1d(
            inputs=char_input,
            filters=config['char_filters'],
            kernel_size=config['char_kernel_size'],
            padding="same",
            activation=tf.nn.relu)

        pool = tf.layers.max_pooling1d(inputs=conv,
                                       pool_size=config['char_pool_size'],
                                       strides=config['char_pool_size'])

        output = tf.reshape(pool, shape=[-1, ntime_steps,
                                         config['char_filters'] * config['char_step_num']//config['char_pool_size']])

        return output

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    def lstm_cell(self, config, lstm_dim, l_num):
        cell = tf.contrib.rnn.LSTMBlockCell(lstm_dim)

        if config["keep_prob"] < 1.:
            if 'emb' not in config["features"]:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            else:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     input_keep_prob=self.keep_prob,
                                                     output_keep_prob=self.keep_prob)

        if l_num > 0:
            cell = tf.contrib.rnn.HighwayWrapper(cell)

        return cell


def evaluate(config, sess, model, dataset, batcher, id_to_word, id_to_tag):
    sent_words, tags_gold, tags_pred, pred_idx = [], [], [], []
    avg_cost = 0.

    gen = batcher.iterator()

    for data in gen:
        dico = get_dico(model, data, 1)
        pred, tf_transition_params, cost = sess.run([model.logit, model.transition_params, model.loss], dico)

        avg_cost += cost * data["target"].shape[0]

        # viterbi decoder
        lst = []
        for tf_unary_scores_, sequence_length_ in zip(pred, data["word_length"]):
            # Remove padding from the scores and tag sequence.
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]

            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, tf_transition_params)

            lst.append(viterbi_sequence)

        # word pred gold data
        for i in range(data["target"].shape[0]):
            sent_len = data["word_length"][i]
            sent_words.append([id_to_word[item] for item in data["ls_input_data"][i].tolist()][:sent_len])
            tags_gold.append([id_to_tag[item] for item in data["target"][i].tolist()][:sent_len])
            tags_pred.append([id_to_tag[item] for item in lst[i]][:sent_len])

    avg_cost /= len(sent_words)

    output = [zip(x, y, z) for x, y, z in zip(sent_words, tags_gold, tags_pred)]

    st = '\n\n'.join(['\n'.join([' '.join(sub_lst) for sub_lst in lst]) for lst in output]) + "\n"

    with open(config["output_path"] + "." + dataset + ".output", 'w') as f:
        f.write(st + "\n")

    os.system("%s < %s > %s" % (config["eval_script"],
                                config["output_path"] + "." + dataset + ".output",
                                config["output_path"] + "." + dataset + ".scores"))

    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in open(config["output_path"] + "." + dataset + ".scores", 'r')]

    val = float(eval_lines[1].strip().split()[-1])

    return avg_cost, val


def get_dico(model, data, keep_prob):
    dico = {model.keep_prob: keep_prob}

    for k, v in data.items():
        if hasattr(model, k):
            dico[getattr(model, k)] = v

    return dico


def train(config):
    print(config["features"], config["lstm_word"])
    raw_data = joblib.load(config["data_path"])
    data = dict([(item, raw_data["data"][item]) for item in config["portion"]])
    id_to_word = {v: k for k, v in raw_data["word_to_id"].items()}
    id_to_tag = raw_data["id_to_tag"]

    emb_matrix = raw_data["emb_matrix"]
    ls_matrix = raw_data["ls_matrix"]

    id_to_emb_map = raw_data["id_to_emb_map"]
    word_to_char_mapping = raw_data["word_to_char_mapping"]
    word_to_cap_mapping = raw_data["word_to_cap_mapping"]

    config["vocab_word"] = emb_matrix.shape[0]
    config["emb_dim"] = emb_matrix.shape[1]
    config["target_num"] = len(id_to_tag)

    print("Creating batchers")
    batcher = {}
    for key, value in data.items():
        batcher[key] = Batcher(config, value, id_to_emb_map, word_to_char_mapping, word_to_cap_mapping, key == "train")

    g = tf.Graph()
    with g.as_default():
        with tf.variable_scope(config["dataset"]) as scope:
            model = Model(config, emb_matrix, ls_matrix)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    max_score = 0
    with tf.Session(config=gpu_config, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for e in range(config["max_max_epoch"]):
            config["lr"] *= 1 / (1.02 + config["lr_decay"] * e)
            model.assign_lr(sess, config["lr"])
            start_time = time.time()
            batch = 0
            avg_cost = 0.
            gen = batcher["train"].iterator()
            max_batch_num = len(batcher["train"].data) // config["batch_size"]

            for data in gen:
                dico = get_dico(model, data, config["keep_prob"])
                cost, lr, _ = sess.run([model.loss, model.lr, model.train_op], dico)

                # Compute average loss
                avg_cost += cost * data["target"].shape[0]
                if batch % config["log_interval"] == 0 and batch > 0:
                    cur_loss = avg_cost / batch
                    elapsed = time.time() - start_time
                    print(f'| epoch {e:3d} | {batch// config["batch_size"]:5d}/{max_batch_num:0d} batches' +
                          f'| lr {lr:1.5f} | ms/batch {elapsed * 1000 / config["log_interval"]:5.2f} |' +
                          f'loss {cur_loss:5.6f}')

                    start_time = time.time()
                    # break
                batch += data["target"].shape[0]

            avg_cost /= batch
            print("Epoch:", '%04d' % (e + 1), "cost=", "{:.9f}".format(avg_cost))

            for key, value in batcher.items():
                if key != "train":
                    cost, score = evaluate(config, sess, model, key, value, id_to_word, id_to_tag)
                    print("Data: ", key, "cost=", "{:.3f}".format(cost), "score=", "{:.2f}".format(score))

                if key == "dev" and score > max_score:
                    saver.save(sess, config["save_path"])
                    max_score = score

        saver.restore(sess, config["save_path"])
        print("model restored")

        for key, value in batcher.items():
            if key != "train":
                cost, score = evaluate(config, sess, model, key, value, id_to_word, id_to_tag)
                print("Data: ", key, "cost=", "{:.3f}".format(cost), "score=", "{:.2f}".format(score))

    del g
    sess.close()

    return score


def get_lstm_dim(config):
    if "ls" not in config["features"]:
        dim = 100 + 122 + config["cap_dim"]
        if "char" in config["features"]:dim += config["char_filters"]
        config["lstm_word"] = int((dim * config["lstm_word"])/(dim - 122))

    return config


def main(argv):
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[argv[0]]
    config["lr_decay"] = config["lr"] / config["max_max_epoch"]
    config = get_lstm_dim(config)
    train(config)


if __name__ == "__main__":
    main(sys.argv[1:])





