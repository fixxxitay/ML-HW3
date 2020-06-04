def find_best_model(models):
    # Third part - Non-Mandatory Assignments
    # step number 10
    # Automate the model selection procedure
    """
        Automate the model selection procedure, i.e.
        the selection of the best model based on the performance
        measurements of all the trained models (Step 5 of the mandatory process)
    """
    max = 0
    for i in range(len(models)):
        if models[i][2] > models[max][2]:
            max = i

    print("Best model found:")
    print(models[max][0])
    return models[max][1]