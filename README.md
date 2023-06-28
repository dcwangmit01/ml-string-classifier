# Machine Learning Model for String Classification

This is a learning repo to help solidify a few AI/ML concepts learned from these videos:

* [An Intuitive and Visual explanation of AI Neural Networks by 3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks)
* [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

This code implements a CLI-driven tool for generic string classification.  When given a dataset and run in "training"
mode, the tool will train and save a multilayer perceptron neural net consisting of a character-level embeddings layer
and 3 hidden layers.  Afterwards, the tool may be run in "classify" mode to load the saved model and predict
classifications of user-provided arbitrary strings.

When applied to a corporate dataset consisting of a CSV of ProductID to ProductFamily associations, the resulting model
classifies with 99.999% accuracy.  In addition, it is also able to classify non-existent ProductIDs with great
accuracy.  For example, a real ProductID "C9100-NM-8X" may be determined to be of the ProductFamily "Catalyst 9K
Network Modules".  If in the future slightly different and never-seen-before ProductIDs such as "D9300-NM-9X" appear,
it successfully classifies the string to what we would logically assume to be the correct ProductFamily which in this
case is also "Catalyst 9K Network Modules".  The dataset I used to write this code is not released, and the prior
example is made up.


## Usage

### Dataset

A sampel dataset is not yet included, but it might look like this:

```CSV
# saved at ./data/dataset.csv.gz
# First column (string input)
# Second column (target classification)
hello,english
hola,spanish
bonjour,french
good,english
bueno,spanish
bien,french
...<many more records>
```

### CLI

Install deps first
```
make pyenv
```

Training the model
```
ml-string-classifier.py train --in-csv=./data/dataset.csv.gz
```

Running classification predictions
```
ml-string-classifier.py classify "string1" "string2"
```

### Usage
```
$ ./ml-string-classifier.py -h
ML String Classifier

Usage:
  ml-string-classifier.py train --in-csv=<file>
    [--out-model=<file>]
    [--hparam-input-size=<INPUT_SIZE>]
    [--hparam-output-size=<OUTPUT_SIZE>]
    [--hparam-hidden-size=<HIDDEN_SIZE>]
    [--hparam-epochs=<EPOCHS>]
    [--hparam-batch-size=<BATCH_SIZE>]
    [--hparam-learning-rate=<LEARNING_RATE>]
    [--hparam-vocab-size=<VOCAB_SIZE>]
    [--hparam-embedding-dim=<EMBEDDING_DIM>]
  ml-string-classifier.py classify <strings_to_classify>... [--in-model=<file>]
  ml-string-classifier.py (-h | --help)

Training Options:
  --in-csv=<file>                        Input file.
  --out-model=<file>                     Output model file.  [default: model.pt]
  --hparam-input-size=<INPUT_SIZE>       Hyperparam: Input size.  [default: 32]
  --hparam-output-size=<OUTPUT_SIZE>     Hyperparam: Output size.  [default: 256]
  --hparam-hidden-size=<HIDDEN_SIZE>     Hyperparam: Hidden size.  [default: 128]
  --hparam-epochs=<EPOCHS>               Hyperparam: Number of epochs.  [default: 2]
  --hparam-batch-size=<BATCH_SIZE>       Hyperparam: Batch size.  [default: 64]
  --hparam-learning-rate=<LEARNING_RATE> Hyperparam: Learning rate.  [default: 0.001]
  --hparam-vocab-size=<VOCAB_SIZE>       Hyperparam: Vocabulary size.  [default: 128]
  --hparam-embedding-dim=<EMBEDDING_DIM> Hyperparam: Embedding dimension.  [default: 32]

Classification Options:
  --in-model=<file>                         Input model file for classify. [default: model.pt]

General Options:
  -h --help                                 Show this screen.

Examples:
  ml-string-classifier.py train --in-csv=./data/dataset.csv.gz
  ml-string-classifier.py classify "string1" "string2"
```


## Developing and Debugging

Run the jupyter notebook

```
make jupyter
# The Makefile will:
# - Install dependencies
# - Run Jupyter Notebook
# - Launch a web browser
#
# Subsequently
# - open the jupyter notebook
# - select the "ml-string-classifier" kernel
# - select "restart kernel and run all"
```
