# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, 
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight_dim = self.trainingSet.input.shape[1] + 1
        self.weight = np.random.rand(self.weight_dim)/100 # plus one for w_0


    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        
        # Write your code to train the perceptron here
        for i in range(0, self.epochs):
            print "Training perceptrion in iteration " + str(i) + "\n"

            error = np.zeros(self.weight_dim)

            # train perceptron with self.trainingSet
            for index, elem in enumerate(self.trainingSet):

                cur_class = self.classify(elem)
                if  cur_class != self.trainingSet.label[index]:
                    # not correctly classified
                    tmp_list = [1]
                    tmp_list.extend(elem)
                    error = error + np.absolute(tmp_list) # TODO add the one earlier

            self.updateWeights(error)


    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Write your code to do the classification on an input image
        return self.fire(testInstance)


    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, error):
        # Write your code to update the weights of the perceptron here
        self.weight = self.weight - self.learningRate * error # TODO why should we use a minus here???
         
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        tmp_list = [1]
        tmp_list.extend(input) # TODO add the one earlier
        return Activation.sign(np.dot(np.array(tmp_list), self.weight)) # add a "1" for x_0
