

def main():

    # second part - implements the modeling
    # 1. Predict which party will win the majority of votes
    # 2. Predict the division of voters between the various parties
    # 3. On the Election Day, each party would like to suggest transportation
    #    services for its voters. Provide each party with a list of its most probable voters

    # step number 4
    # Load the prepared training set
    """
    Each training should be done via cross-validation on the training set,
    to maximize performance of the model while avoiding over fitting.
    """

    # step number 5
    # Train at least two models
    """
    Each training should be done via cross-validation on the training set,
    to maximize performance of the model while avoiding over fitting.
    """

    # step number 6
    # Load the prepared validation set

    # step number 7
    # Apply the trained models on the validation set and check performance
    """
        It is your call which performance measure to use,
        and it is possible to check multiple measures
    """

    # step number 8
    # Select the best model for the prediction tasks
    """
        The model selection is “manual” (not an automatic process),
        but it should be based on the performance measurements
    """

    # step number 9
    # Use the selected model to provide the following:
    """
        1. Predict to which party each person in the test set will vote
           (predict the label of each row in the test set)
           A CSV file that contain the voting predictions 
           (predicted labels) on the test set
           
        2. Construct the (test) confusion matrix and overall test error
    """


if __name__ == '__main__':
    main()
