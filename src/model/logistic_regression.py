# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

import util.loss_functions

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])
        
        self.weight_dim = self.trainingSet.input.shape[1] + 1
        self.weight = np.random.rand(self.weight_dim)/100
        self.error_history=[]

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        #diff_err = DifferentError()
        error_history=[]
        for i in range(0, self.epochs):
            if verbose:
                print ("Training perceptron in iteration " + str(i) + "\n")
            
            error = np.zeros(self.weight_dim)
            
            #train perceptron
            for index, elem in enumerate(self.trainingSet):
                temp = [1]
                temp.extend(elem)
            
                output = self.classify(elem)
                de_dy = output - self.trainingSet.label[index]
                de_dx = de_dy * output * (1 - output)
                de_dw = de_dx * np.array(temp)
                error += de_dw
   
            error /= len(self.trainingSet.input)
            self.error_history.append(np.average(error,axis=0)) 
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
        classified = (map(self.classify, test))
        sign = lambda x: x > 0.5
        return list(map(sign, classified))

    def updateWeights(self, grad):
        self.weight -= grad * self.learningRate

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        temp = [1]
        temp.extend(input)
        return Activation.sigmoid(np.dot(np.array(temp), self.weight))
