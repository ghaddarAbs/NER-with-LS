Robust Lexical Features for Improved Neural Network Named-Entity Recognition
================================================================

This repository contains the source code for the NER system presented in the following research publication ([tmp link](https://arxiv.org/submit/2291406/view))

    Abbas Ghaddar and Philippe Langlais 
    "Robust Lexical Features for Improved Neural Network Named-Entity Recognition",
    To appear in COLING 2018

## Requirements

* python 3.6.3
* tensorflow 1.6
* pyhocon (for parsing the configurations)

## Prepare the Data
1. Download the data from [here](https://drive.google.com/open?id=1Trl1GQLWZn19LvelL-6clATvATKOPH77) and unzip the files in data directory.

2. Change the `raw_path` variables for [conll](http://www.cnts.ua.ac.be/conll2003/ner/) and [ontonotes](http://conll.cemantix.org/2012/data.html) datasets in `experiments.config` file to `path/to/conll-2003` and `path/to/conll-2012/v4/data` respectively. For conll dataset please rename eng.train eng.testa eng.testb files to conll.train.txt conll.dev.txt conll.test.txt respectively. 

3. Run: 
 
```
$ python preprocess.py dataset_name[conll|ontonotes]
```

## Training
Once the data preprocessing is completed, you can train and test a model with:
```
$ python model.py dataset_name[conll|ontonotes]
```
