import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pydotplus
from IPython.display import display, Image
from sklearn import tree
import datetime

def get_data_from_db(query, conn_params):
    conn = mysql.connector.connect(**conn_params)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def calculate_age(birthdate):
    today = datetime.datetime.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

def preprocess_data(df_vtargetmailt, df_prospectivebuyer):
    df_vtargetmailt.fillna(method='ffill', inplace=True)
    df_prospectivebuyer.fillna(method='ffill', inplace=True)

    df_vtargetmailt['Gender'] = df_vtargetmailt['Gender'].map({'Male': 0, 'Female': 1})
    df_prospectivebuyer['Gender'] = df_prospectivebuyer['Gender'].map({'Male': 0, 'Female': 1})

    df_vtargetmailt['MaritalStatus'] = df_vtargetmailt['MaritalStatus'].map({'Single': 0, 'Married': 1})
    df_prospectivebuyer['MaritalStatus'] = df_prospectivebuyer['MaritalStatus'].map({'Single': 0, 'Married': 1})

    df_vtargetmailt['Age'] = df_vtargetmailt['BirthDate'].apply(lambda x: calculate_age(pd.to_datetime(x)))
    df_prospectivebuyer['Age'] = df_prospectivebuyer['BirthDate'].apply(lambda x: calculate_age(pd.to_datetime(x)))

    return df_vtargetmailt, df_prospectivebuyer

def train_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return clf, accuracy, report

def visualize_tree(clf, feature_names):
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=['No', 'Yes'],
                                    filled=True, rounded=True,
                                    special_characters=True,
                                    proportion=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("decision_tree.pdf")  # 输出为PDF
    graph.write_svg("decision_tree.svg")  # 输出为SVG
    return Image(graph.create_png())


    graph = pydotplus.graph_from_dot_data(dot_data)
    return Image(graph.create_png())

def predict_bike_buyers(clf, df_prospectivebuyer, feature_names):
    prospective_customers = df_prospectivebuyer[feature_names]
    prospective_predictions = clf.predict(prospective_customers)
    df_prospectivebuyer['BikeBuyerPrediction'] = prospective_predictions
    potential_customers = df_prospectivebuyer[df_prospectivebuyer['BikeBuyerPrediction'] == 1]
    return potential_customers[['FirstName', 'LastName', 'EmailAddress']]

def main():
    conn_params = {
        'host': 'localhost',
        'user': 'root',
        'password': '123456',
        'database': 'adventureworks'
    }

    query_vtargetmailt = "SELECT * FROM vtargetmailt"
    query_prospectivebuyer = "SELECT * FROM prospectivebuyer"

    df_vtargetmailt = get_data_from_db(query_vtargetmailt, conn_params)
    df_prospectivebuyer = get_data_from_db(query_prospectivebuyer, conn_params)

    df_vtargetmailt, df_prospectivebuyer = preprocess_data(df_vtargetmailt, df_prospectivebuyer)

    features = ['Age', 'Gender', 'YearlyIncome', 'TotalChildren', 'NumberChildrenAtHome', 'HouseOwnerFlag', 'NumberCarsOwned']
    target = 'BikeBuyer'

    X = df_vtargetmailt[features]
    y = df_vtargetmailt[target]

    clf, accuracy, report = train_decision_tree(X, y)
    print(f'Accuracy: {accuracy}')
    print(report)

    tree_image = visualize_tree(clf, features)
    display(tree_image)

    potential_customers = predict_bike_buyers(clf, df_prospectivebuyer, features)
    print(potential_customers)

if __name__ == "__main__":
    main()
