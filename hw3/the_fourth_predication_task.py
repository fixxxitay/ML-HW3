import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import implements_the_modeling as mod

def get_x_and_y(csv):
     df = pd.read_csv(csv)
     #shuffle
     df = df.sample(frac=1).reset_index(drop=True)

     x = df.drop("Vote", 1)
     y = df["Vote"]

     return x, y

right_feature_set = ["Vote", "Yearly_IncomeK", "Number_of_differnt_parties_voted_for", "Political_interest_Total_Score",
                     "Avg_Satisfaction_with_previous_vote", "Avg_monthly_income_all_years",
                     "Most_Important_Issue", "Overall_happiness_score", "Avg_size_per_room",
                     "Weighted_education_rank"]

def mod_vote(clf, x_test, feature, toAdd):
    x_test[feature] += toAdd
    y_test_pred_probability = np.mean(clf.predict_proba(x_test), axis=0)
    winner_pred = np.argmax(y_test_pred_probability)
    print(feature, toAdd)  
    print(mod.from_num_to_label[winner_pred])

def main():
    # Third part - Non-Mandatory Assignments
    # step number 14
    # Handle the fourth predication task

    x_train, y_train = get_x_and_y("prepared_train.csv")
    x_validation, y_validation = get_x_and_y("prepared_validation.csv")
    x_test, y_test = get_x_and_y("prepared_test.csv")

 
    clf = RandomForestClassifier(n_jobs=-1, random_state=0, criterion='entropy', min_samples_split=3,
                                    min_samples_leaf=1)
    
    x_train_and_validation = x_train.append(x_validation).reset_index(drop=True)
    y_train_and_validation = y_train.append(y_validation).reset_index(drop=True)

    clf.fit(x_train_and_validation, y_train_and_validation)

    mod_vote(clf, x_test,"Yearly_IncomeK", 1)
    mod_vote(clf, x_test,"Number_of_differnt_parties_voted_for", 2)
    mod_vote(clf, x_test,"Political_interest_Total_Score", 24)
    mod_vote(clf, x_test,"Avg_Satisfaction_with_previous_vote", 2)
    mod_vote(clf, x_test,"Avg_monthly_income_all_years", 0.5)
    mod_vote(clf, x_test,"Most_Important_Issue", 4)
    mod_vote(clf, x_test,"Overall_happiness_score", 3)
    mod_vote(clf, x_test,"Avg_size_per_room", 3)
    mod_vote(clf, x_test,"Weighted_education_rank", 3)


if __name__ == '__main__':
    main()