#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator

from util.loss_functions import MeanSquaredError
from util.loss_functions import BinaryCrossEntropyError
from util.loss_functions import CrossEntropyError
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def main():


    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    '''myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)'''
                                        
    myLogisticClassifier = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=50)
                                        
    
	
    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nPerceptron has been training..")
    myLogisticClassifier.train()
    print("Done..")
    
    #Plot the error function
    fig=plt.figure()
    ax0=fig.add_subplot(111)
    ax0.plot(range(myLogisticClassifier.epochs),myLogisticClassifier.error_history)
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Average of Error')
    ax0.set_title('Error function of Logistic Regression, learning Rate: '+str(myLogisticClassifier.learningRate))
    plt.show()

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    logisticPred = myLogisticClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    #print(logisticPred)
    #print(len(logisticPred))
    #print(len(stupidPred))
    #print(stupidPred)
    evaluator.printAccuracy(data.testSet, logisticPred)
    
if __name__ == '__main__':
    main()
