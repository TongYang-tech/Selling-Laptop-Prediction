import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder

def add_predictive_col(html_group, current_predictive_users, current_predictive__logs, new_predictive_col):
    # gets the logs and checks if the html is in the logs
    # group the df by user_id and sum all the time (seconds)
    # combine user and log dataframe (in this case using .merge() method)
    # rename the new column of df to from "new_predictiver_col"
    current_predictive__logs = current_predictive__logs[current_predictive__logs["url"].isin(html_group)]
    current_predictive__logs = current_predictive__logs.groupby(["user_id"]).sum("seconds")
    current_predictive_users = pd.merge(current_predictive_users, current_predictive__logs, how = 'outer', on = 'user_id').fillna(-1)
    current_predictive_users = current_predictive_users.rename(columns={"seconds": new_predictive_col})
    return current_predictive_users

class UserPredictor:
    def __init__(self):
        # make transformer with encoder, polynomial features with "age", "purchases, and "predictive samples"
        # then pipeline from transm, scalar, and logistic regression
        trans = make_column_transformer((OneHotEncoder(), ["badge"]),(PolynomialFeatures(degree = 5), ["age", "past_purchase_amt", "predictive_sample_1", "predictive_sample_2"]))
        self.model = Pipeline([("transformers", trans), ("std_scalar", StandardScaler()),("log_regr", LogisticRegression(max_iter = 1500))])
        # catergories - Important data names from the csv files that are important
        # PREDICTIVE SAMPLES (that are leading to a higher accuracy):
        # tech_pair - key board and laptop html file
        # technology - tablet and laptop html
        # other things - blender, cleats, keyboard
        self.categories = ["predictive_sample_1", "predictive_sample_2", "predictive_sample_3", "age", "past_purchase_amt", "badge"]
        self.sample_tech_pair = ["/keyboard.html", "/laptop.html"]
        self.sample_technology = ["/tablet.html", "/laptop.html"]
        self.sample_other = ["/blender.html", "/cleats.html", "/keyboard.html"]

    def fit(self, training_users_data, training_logs_data, training_y_data):
        # add predicted samples to the data frame "training_users_data"
        training_users_data = add_predictive_col(self.sample_technology, training_users_data, training_logs_data, "predictive_sample_1")
        training_users_data = add_predictive_col(self.sample_other, training_users_data, training_logs_data, "predictive_sample_2")
        training_users_data = add_predictive_col(self.sample_tech_pair, training_users_data, training_logs_data, "predictive_sample_3")
        training_logs_data = training_users_data[self.categories]
        self.model.fit(training_logs_data, training_y_data["y"])
      
    def predict(self, testing_users_data, testing_logs_data):
        # add predicted samples to the data frame "testing_users_data"
        testing_users_data = add_predictive_col(self.sample_technology, testing_users_data, testing_logs_data, "predictive_sample_1")
        testing_users_data = add_predictive_col(self.sample_other, testing_users_data, testing_logs_data, "predictive_sample_2")
        testing_users_data = add_predictive_col(self.sample_tech_pair, testing_users_data, testing_logs_data, "predictive_sample_3")
        # returns a numpy array
        return self.model.predict(testing_users_data[self.categories])