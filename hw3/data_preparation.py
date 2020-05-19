right_feature_set = ["Yearly_IncomeK", "Number_of_differnt_parties_voted_for", "Political_interest_Total_Score",
                     "Avg_Satisfaction_with_previous_vote", "Avg_monthly_income_all_years",
                     "Most_Important_Issue", "Overall_happiness_score", "Avg_size_per_room",
                     "Weighted_education_rank"]


def main():
    # first part - data preparation
    # step number 1
    # Load the Election Challenge data from the ElectionsData.csv file
    """
    1. Can be found at the “The Election Challenge” section in the course site
    2. Make sure that you’ve identified the correct type of each attribute
    """

    # step number 2
    # Select the right set of features (as listed in the end of this document),
    # and apply the data preparation tasks that you’ve carried out in the former
    # exercise, on the train, validation, and test data sets
    """
    1. At the very least, handle fill up missing values
        All other actions, such as outlier detection and normalization,
        are not mandatory. Still, you are encouraged to do them,
        as they should provide you added values for the modeling part
    2. Whether to use the validation set for pre-processing steps is your call
    """

    # step number 3
    # Save the 3x2 data sets in CSV files
    # CSV files of the prepared train, validation and test data sets


if __name__ == '__main__':
    main()
