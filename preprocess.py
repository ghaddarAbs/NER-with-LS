import os, joblib, re, pyhocon, warnings, copy, sys
import numpy as np
from collections import defaultdict
from glob import glob

warnings.filterwarnings('ignore')


def dump_ner_data(config, dataset):

    raw_data = joblib.load(config["data_path"].replace("_with_data",""))
    word_to_id = raw_data["word_to_id"]
    tag_to_id = raw_data["tag_to_id"]

    # convert tsv to numpy matrix
    data = {}
    for p in config["portion"]:
        data[p] = tsv_to_numpy("data/"+dataset+"."+p+".iob", word_to_id, tag_to_id, config["max_sent_len"])
    raw_data["data"] = data

    # write the data and mapping
    with open(config["data_path"], 'wb') as fp:
        joblib.dump(raw_data, fp)


def tsv_to_numpy(data_path, word_to_id, tag_to_id_cls, max_sent_len):
    data = None
    vals = []

    with open(data_path) as f:
        for line in f:
            if line.strip():
                vals.append(line.strip().split(" "))
            else:
                output = prepare_sent(vals, word_to_id, tag_to_id_cls, max_sent_len)
                data = output if data is None else np.concatenate((data, output), axis=0)
                vals = []
    return data


def prepare_sent(vals, word_to_id, tag_to_id, max_sent_len):
    words = [word_to_id[word[0]] for word in vals]
    tags = [word[-1] for word in vals]
    tags_lst = []

    for i in range(len(tags)):

        if tags[i] == "O":
            tags_lst.append(0)
            continue

        tag = re.sub(r'^B-|^I-', '',  tags[i])
        if i != len(tags) - 1 and tags[i].startswith('B-') and not tags[i + 1].startswith('I-'):
            tags_lst.append(tag_to_id['U-' + tag])
        elif i != len(tags) - 1 and tags[i].startswith('B-') and tags[i + 1].startswith('I-'):
            tags_lst.append(tag_to_id['B-' + tag])
        elif i != len(tags) - 1 and tags[i].startswith('I-') and tags[i + 1].startswith('I-'):
            tags_lst.append(tag_to_id['I-' + tag])
        elif i != len(tags) - 1 and tags[i].startswith('I-') and not tags[i + 1].startswith('I-'):
            tags_lst.append(tag_to_id['L-' + tag])

        # last index
        elif i == len(tags) - 1 and tags[i].startswith('I-'):
            tags_lst.append(tag_to_id['L-' + tag])
        elif i == len(tags) - 1 and tags[i].startswith('B-'):
            tags_lst.append(tag_to_id['U-' + tag])

    tags = tags_lst

    return lst_to_array(words, tags, max_sent_len)


def lst_to_array(words, tags, max_sent_len):
    output = np.zeros((1, max_sent_len * 2 + 1), np.int32)
    output[0, :len(words)] = words
    output[0, max_sent_len:max_sent_len + len(tags)] = tags
    output[0, -1] = len(words)
    return output


###################################################
############### Ontonotes to IOB2 #################
###################################################
def create_onto_raw(raw_path, portion):
    datafile = raw_path + portion + "/data/english/annotations/"
    files = [y for x in os.walk(datafile) for y in glob(os.path.join(x[0], '*_gold_conll'))]

    words = []
    tags = []
    dico = defaultdict(int)

    for filename in files:
        if "/pt/nt" in filename:
            continue

        item = load_onto_file(filename)
        span = filename.replace(datafile, '').split('/')[0]
        dico[span] += len(item[0])

    for filename in files:
        if "/pt/nt" in filename:
            continue
        item = load_onto_file(filename)
        words += item[0]
        tags += item[1]

    output = [zip(x, y) for x, y in zip(words, tags)]
    st = '\n\n'.join(['\n'.join([' '.join(sub_lst) for sub_lst in lst]) for lst in output]) + "\n"

    with open("data/ontonotes." + portion + ".iob", 'w') as f:
        f.write(st + "\n")


def load_onto_file(filename):
    words = []
    tags = []

    sent_words = []
    tags_gold = []

    with open(filename) as data_file:
        for line in data_file:
            if line.strip():
                vals = line.strip().split()
                if vals[0] in ['#begin', '#end']:
                    continue

                words.append(replace_parantheses(vals[3]))
                tags.append(vals[10])
            elif len(words) > 0:
                tags = transform_onto_tags(tags)
                sent_words.append(copy.deepcopy(words))
                tags_gold.append(copy.deepcopy(tags))

                words = []
                tags = []

    return sent_words, tags_gold


def transform_onto_tags(lst):
    tags = ["O"] * len(lst)
    flag = False
    cur = "O"

    for i in range(len(lst)):
        if lst[i][0] == "(" and not flag:
            cur = lst[i].replace("(", "").replace(")", "").replace("*", "")
            tags[i] = "B-" + cur

            if lst[i][-1] != ")":
                flag = True

        elif flag and lst[i].startswith("*"):
            tags[i] = "I-" + cur
            if lst[i][-1] == ")":
                flag = False

    return tags


def replace_parantheses(word):
    word = word.replace('/.', '.')
    if not word.startswith('-'):
        return word

    if word == '-LRB-':
        return '('
    elif word == '-RRB-':
        return ')'
    elif word == '-LSB-':
        return '['
    elif word == '-RSB-':
        return ']'
    elif word == '-LCB-':
        return '{'
    elif word == '-RCB-':
        return '}'
    else:
        return word


###############################################
############### CoNLL to IOB2 #################
###############################################
def create_conll_raw(raw_path, portion):
    sent_words = []
    tags_gold = []

    words = []
    tags = []

    with open(raw_path+"conll."+portion+".txt") as data_file:
        for line in data_file:
            if line.strip():
                vals = line.strip().split(" ")
                if vals[0] != "-DOCSTART-":
                    words.append(vals[0])
                    tags.append(vals[-1])

            elif len(words) > 0:
                tags = iob_to_iob2(tags)
                sent_words.append(copy.deepcopy(words))
                tags_gold.append(copy.deepcopy(tags))

                words = []
                tags = []

    output = [zip(x, y) for x, y in zip(sent_words, tags_gold)]
    st = '\n\n'.join(['\n'.join([' '.join(sub_lst) for sub_lst in lst]) for lst in output]) + "\n"

    with open("data/conll." + portion + ".iob", 'w') as f:
        f.write(st + "\n")


def iob_to_iob2(tags):
    prev = "O"

    for i in range(len(tags)):
        tag = tags[i].replace("B-", "").replace("I-", "")
        if tags[i].startswith("I-") and not prev.endswith("-"+tag):
            tags[i] = "B-"+tag
        prev = tags[i]

    return tags


def main(argv):
    if not os.path.exists("models"):
        os.makedirs("models")
    
    dataset = argv[0]
    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[dataset]
    if dataset == "ontonotes":
        [create_onto_raw(config["raw_path"], p) for p in config["portion"]]

    elif dataset == "conll":
        [create_conll_raw(config["raw_path"], p) for p in config["portion"]]
    else:
        print("Unknown dataset")
        sys.exit(1)

    dump_ner_data(config, dataset)


if __name__ == "__main__":
    main(sys.argv[1:])






