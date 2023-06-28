# Machine Learning model for string classification

1. The code splits the dataset into train, dev, and validate.
2. The model is trained with the train dataset
3. The model is validated with teh val dataset
4. The dev dataset is unused for now

## Sample Dataset

Input is a gzipped CSV file with 2 columns of strings.
- Each row has 2 columns
- First column is any string up to 32 characters
- Second column is a string that represents a classification

Note: Sample data is not included for now.  I'm hoping to generate one.
```CSV
hello,english
hola,spanish
bonjour,french
good,english
bueno,spanish
bien,french
...
```

## Running it

```
make jupyter
# select the "ml-string-classifier" kernel
# select "restart kernel and run all"
```
