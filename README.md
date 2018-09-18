Generic Artificial Neural Network
===============

A simplified approach for quick and easy generation of neural networks

Features
--------

- Allows users to train an ANN on a csv of their choice
- Users can specify variable types by following csv patterns
- Provides users with simplified input framework to create advanced ANNs


Project Organization
--------

This project is organized in the following manner:

<details><summary>Root directory: Contains all code needed to run a simple neural network.</summary>
    <ul>
    <li><details><summary>Data: Contains two example data sets</summary>
    <ul>
      <li>[Iris]</li>
      <li>[Voting]</li>
    </ul>
    </details></li>
    <li>AnnCode: Define your choices and ANN Code</li>
    <li>BaseCode: Code to import your data and run your AnnCode</li>
    <li>requirements: Code to used to install all needed python requirements</li>
    </ul>
</details>


Prerequisites
-------------

This project uses a number of open source projects to work properly:

* [NumPy] - The fundamental package for scientific computing with Python.
* [Pandas] - An open-source software for data structures and analysis.
* [SkLearn] - Machine learning in python.
* [TensorFlow] - An open source machine learning framework for everyone
* [Keras] - The Python Deep Learning Library

How to Use
----------

Make sure you have Python 3.6.x installed on your system. You can download it [here](https://www.python.org/downloads/).

### Linux

1. Clone this repo in your preferred directory:
    ```sh
    $ git clone https://github.com/connor-makowski/SimpleANN.git
    ```
2. Go to the root directory of the project `SimpleANN` project using the `cd` command.
    ```sh
    $ cd SimpleANN
    ```
3. Create your working environment. If you are using `virtualenv`, you can create a new environment based on Python 3.6.x (or higher):
    ```sh
    $ virtualenv -p python3 SANN
    ```
    Where `SANN` is the directory name to place the new virtual environment. Then, you must activate it:
    ```sh
    $ source SANN/bin/activate
    ```
4. This project has some dependencies. You can install them all at once:
    ```sh
    (SANN)[...]$ pip install -Ur requirements.txt
    ```

### Getting Started

1. Put your data in the Data folder.
  - Your data should follow the patterns set by iris.csv and voting.csv where:
    - Row one of your CSV contains the name of the variables in that column
    - Row two of your CSV contains information about each variable
      - Categorical variables should be marked as `cat`
      - Continuous variables should be marked as `con`
      - Other variables should be marked as `oth`
    - Row three of your CSV contains how you want this variable to play into your model
      - Target variables should be marked as `tar`
      - Feature variables should be marked as `fet`
      - Rejected variables should be marked as `rej`
  - Your data should be in CSV format
  - Case (capitalization) matters

2. Open up the AnnCode.py file with your favorite text editor.
3. Edit filename (line 8) to exactly match your file in the Data folder. For example:
    ```sh
    filename = 'myfile.csv'
    ```
4. Edit the Choices section to match your requirements.
  - `test_pct` is the percentage of your data that you will test your trained model upon.
  - `num_epochs` is the number of forward and backward passes you will make on your data.
  - `batch_size` is the number of data to consider in each epoch.
  - `random_seed` is the seed at which you initialize your neural network.
  - `use_kfold` is a `True`/`False` parameter for using a k-fold cross validation process for testing.
  - `n_folds` is the number of folds to use if you are using k-fold cross validation.
5. Edit your Model.
  - This model structure is defined using [Keras] notation.
  - For this project all models should be `Sequential`
  - Layers should follow the structure below:
    - The first layer should always include `input_dim=input_dimensions`.
    - The final layer should always include `output_dimensions` as the output dimension.
    - You can add any number of any layers.
    - Each layer can have any number of nodes.
    - Each layer can have any [Keras] supported activation function.
  - Compiling the model is open to any [Keras] supported methods
    - Note that `accuracy` should always be the first listed metric.

### Run Your Model

Use Python to execute the `BaseCode.py` file. This will import your AnnCode.py file and your data to process the results.
```sh
$ python BaseCode.py
```

Your output should look something like this:
```sh
Expected Accuracy (kfold): 0.75
Tested Accuracy: 0.9800000190734863
Confusion Matrix:
    0   1   2
0  19   0   0
1   0  14   1
2   0   0  16
```

License
-------

Copyright (c) 2018 Connor Makowski

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

[Iris]: <https://en.wikipedia.org/wiki/Iris_flower_data_set>
[Voting]: <https://archive.ics.uci.edu/ml/datasets/congressional+voting+records>
[NumPy]: <http://www.numpy.org/>
[Pandas]: <https://pandas.pydata.org/>
[SkLearn]: <http://scikit-learn.org/stable/>
[TensorFlow]: <https://www.tensorflow.org/>
[Keras]: <https://keras.io/>
