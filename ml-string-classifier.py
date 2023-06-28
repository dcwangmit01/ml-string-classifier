#!/usr/bin/env python3
"""
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
"""

from docopt import docopt
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class App:
    def __init__(self, args):
        self.args = args

        # Use generator for reproducible random results
        self.g = torch.Generator().manual_seed(42)  # Seed with the answer to life

        # Use CUDA if possible
        #  TODO: Only informative for now
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.device = torch.device("cuda")
            print("GPU is available")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, CPU used")

    def run(self):
        if args["train"]:
            self.handle_train_command(args)
        elif args["classify"]:
            self.handle_classify_command(args)

    def handle_train_command(self, args):
        IN_CSV = args.get("--in-csv")
        OUT_MODEL = args.get("--out-model")
        INPUT_SIZE = int(args.get("--hparam-input-size"))
        OUTPUT_SIZE = int(args.get("--hparam-output-size"))
        HIDDEN_SIZE = int(args.get("--hparam-hidden-size"))
        EPOCHS = int(args.get("--hparam-epochs"))
        BATCH_SIZE = int(args.get("--hparam-batch-size"))
        LEARNING_RATE = float(args.get("--hparam-learning-rate"))
        VOCAB_SIZE = int(args.get("--hparam-vocab-size"))
        EMBEDDING_DIM = int(args.get("--hparam-embedding-dim"))

        # Load dataset
        dataset = MyDataset(IN_CSV, INPUT_SIZE)

        # Separate the dataset into training, dev, and validation
        train_dataset, dev_dataset, val_dataset = random_split(dataset, [0.70, 0.15, 0.15])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=self.g)
        dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=self.g)  # noqa
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=self.g)

        # Initialize model and optimizer
        model = NeuralNet(VOCAB_SIZE, EMBEDDING_DIM, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # Print number of parameters in model
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of model parameters: {total_params}")

        # Track stats in the training loop
        losses = []

        # Training loop
        for epoch in range(EPOCHS):
            i = 0
            for inputs, targets in train_loader:
                outputs = model(inputs)

                # Compute the target matrix for this batch, one-hot encoded
                targets = F.one_hot(targets.to(torch.int64), OUTPUT_SIZE)

                # Calculate the loss
                loss = criterion(outputs, targets.to(torch.float32))

                # Back Propogation
                optimizer.zero_grad()
                loss.backward()

                # Gradient Descent
                optimizer.step()

                # Track Statistics
                losses.append(loss.item())

                # Print Status
                i += 1
                if i % 5000 == 0:
                    print(f"Training epoch {epoch} batch {i}")

        # Validation
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)  # returns tuple of values tensor and index tensor
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f"Accuracy on validation set: {100 * correct / total}%")

        # Save the model and metadata (label data)
        torch.save(
            {
                "training_args": args,
                "model_state_dict": model.state_dict(),
                "labels_state_dict": dataset.labels.state_dict(),
            },
            OUT_MODEL,
        )

    def handle_classify_command(self, args):
        IN_MODEL = args.get("--in-model")
        STRINGS_TO_CLASSIFY = args.get("<strings_to_classify>")

        checkpoint = torch.load(IN_MODEL)
        model = NeuralNet()
        model.load_state_dict(checkpoint["model_state_dict"])
        labels = Labels()
        labels.load_state_dict(checkpoint["labels_state_dict"])
        training_args = checkpoint["training_args"]

        # Print the highest probability Labels for each String to Classify
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            for str in STRINGS_TO_CLASSIFY:
                input = MyDataset.str_to_tensor_truncate(str, int(training_args["--hparam-input-size"]))
                input = input.unsqueeze(0)  # Add an extra dimension to convert it into a two-dimensional tensor
                output = model(input)
                probabilities = softmax(output)

                # Find the top 3 probabilities
                top_n = 3
                values, indices = probabilities.topk(top_n)
                values = values.squeeze()  # Remove outer dimension
                indices = indices.squeeze()  # Remove outer dimension

                print(f"For str [{str}] most likely classifications with >10% probability are:")

                for p, l in zip(values, [labels.label_from_int(i) for i in indices]):
                    if p > 0.1:
                        print(f"  probability[{p:.2f}] of label[{l}]")


class NeuralNet(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=32, input_size=32, hidden_size=128, output_size=256):
        super(NeuralNet, self).__init__()

        # Following this pattern to help model restore
        # https://stackoverflow.com/questions/69903636/how-can-i-load-a-model-in-pytorch-without-having-to-remember-the-parameters-used
        self.kwargs = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
        }

        # Embedding layer for ASCII characters
        #   VOCAB_SIZE x EMBEDDING_DIM
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(input_size * embedding_dim, hidden_size)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(hidden_size, output_size)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        # TODO: Understand if normalization layers might help.

    def forward(self, x):
        # x - dim is BATCH_SIZE(32), INPUT_SIZE(32)

        # embedding is VOCAB_SIZE(128), EMBEDDING_DIM(256)
        #   output dim is 32, 32, 256
        x = self.embedding(x)

        # Flatten to 32, 8192
        x = x.view(x.size(0), -1)

        # Propogate through the layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Do not use activation function on last layer

        # Return the result logits
        return x


class MyDataset(Dataset):
    def __init__(self, csv_gzip_file, truncate_length):
        self.truncate_length = truncate_length

        #   The first column of the CSV contains the input strings
        #   The second column of the CSV contains the label strings
        self.data = pd.read_csv(csv_gzip_file, compression="gzip")

        # Find all unique labels mapping each unique label to an integer
        labelstrs = self.data.iloc[:, 1].unique().tolist()
        labelstrs.sort(),
        self.labels = Labels(labelstrs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # A tensor representing a single data input string
        input_tensor = self.str_to_tensor(self.data.iloc[idx, 0])

        # A single element that represents the index of the label classification
        #   that represents the correct answer to be learned
        target = self.labels.label_to_int(self.data.iloc[idx, 1])

        # Return both as tensors
        #   input_tensor is dimension [1, INPUT_SIZE]
        #   target_tensor is dimension [1, 1]
        return input_tensor, torch.tensor(target, dtype=torch.int64)

    def str_to_tensor(self, input_str):
        # convert the input string to a tensor of numbers
        # pad and truncate the tensor to a fixed length
        return MyDataset.str_to_tensor_truncate(input_str, self.truncate_length)

    def str_from_tensor(self, t):
        return "".join(chr(int_val) for int_val in t).rstrip()

    @staticmethod
    def str_to_tensor_truncate(input_str, truncate_length):
        # convert the input string to a tensor of numbers
        # pad and truncate the tensor to a fixed length
        return torch.tensor([ord(c) for c in input_str.ljust(truncate_length)[:truncate_length]], dtype=torch.int64)


class Labels(object):
    def __init__(self, labels=[]):
        labelintmap = {v: i for i, v in enumerate(labels)}
        self.data = {"labels": labels, "labelintmap": labelintmap}

    def label_from_int(self, i):
        if i < len(self.data["labels"]):
            return self.data["labels"][i]
        else:
            return "UNKNOWN"

    def label_to_int(self, s):
        return self.data["labelintmap"][s]

    def state_dict(self):
        return {"data": self.data}

    def load_state_dict(self, state_dict):
        self.data = state_dict["data"]


if __name__ == "__main__":
    args = docopt(__doc__, version="ML String Classifier")
    app = App(args)
    app.run()
