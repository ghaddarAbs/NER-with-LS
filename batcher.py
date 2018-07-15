import numpy as np
from random import shuffle


class Batcher:

    def __init__(self, config, data, id_to_emb_map,
                 word_to_char_mapping, word_to_cap_mapping,
                 is_train=False):

        self.batch_size = config["batch_size"]
        if not is_train and self.batch_size < 25:
            self.batch_size *= 50

        self.max_sent_len = (data.shape[1] - 1) // 2
        self.is_train = is_train
        self.char_step_num = config["char_step_num"]
        self.word_step_num = config["word_step_num"]
        self.data = data

        self.id_to_emb_map = id_to_emb_map
        self.word_to_char_mapping = word_to_char_mapping
        self.word_to_cap_mapping = word_to_cap_mapping
        self.att_lst = ["ls_input_data", "emb_input_data",
                        "cap_input_data", "char_input_data",
                        "target", "word_length"]

        self.data = self.load_data(self.data, id_to_emb_map, word_to_char_mapping, word_to_cap_mapping)

    def load_data(self, data, id_to_emb_map, word_to_char_mapping, word_to_cap_mapping):
        sample_num = data.shape[0]

        dico = {}
        for item in self.att_lst:
            if item == "ls_input_data":
                dico[item] = data[:, :self.max_sent_len]
            elif item == "target":
                dico[item] = data[:, self.max_sent_len:2*self.max_sent_len]
            elif item == "word_length":
                dico[item] = data[:, -1]
            elif item == "emb_input_data":
                dico[item] = id_to_emb_map[dico["ls_input_data"]]
            elif item == "char_input_data":
                dico[item] = word_to_char_mapping[dico["ls_input_data"]]
            elif item == "cap_input_data":
                dico[item] = word_to_cap_mapping[dico["ls_input_data"]]

        data = [dict([(item, dico[item][i]) for item in self.att_lst]) for i in range(sample_num)]

        return data

    def iterator(self):
        for j in range(0, len(self.data), self.batch_size):
            f, t = j, j + self.batch_size
            if t > len(self.data):
                t = len(self.data)

            max_len = max([item["word_length"] for item in self.data[f:t]])
            b = t - f

            # initialize batch input
            dico = {}
            for item in self.att_lst:
                if item == "word_length":
                    dico[item] = np.zeros((b), dtype=np.int32)
                elif item == "char_input_data":
                    dico[item] = np.zeros((b, max_len, self.char_step_num), dtype=np.int32)
                else:
                    dico[item] = np.zeros((b, max_len), dtype=np.int32)

            # fill batch input
            for i in range(b):
                for item in self.att_lst:
                    if item == "word_length":
                        dico[item][i] = self.data[f:t][i][item]

                    elif item == "char_input_data":
                        dico[item][i] = self.data[f:t][i][item][:max_len, :]

                    else:
                        dico[item][i] = self.data[f:t][i][item][:max_len]
            yield dico

        if self.is_train:
            shuffle(self.data)
