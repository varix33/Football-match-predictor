try :
    from preprocessing import preprocessing, get_diff_goals_last_enc, get_avg_goal_last_5matches, get_win_score_5last_enc
    from preprocessing import get_win_rate_tournament, get_rank_tournament
except ImportError:
    from pre.preprocessing import preprocessing, get_diff_goals_last_enc, get_avg_goal_last_5matches, get_win_score_5last_enc
    from pre.preprocessing import get_win_rate_tournament, get_rank_tournament

import  os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
import tkinter as tk

try :
    from tournament import demarrerTournois, write_file_csv
    from treeTournament import ArbreTournoi
except ImportError:
    from .tournament import demarrerTournois, write_file_csv
    from .treeTournament import ArbreTournoi

"""
-----------------------------------------------------------------------------------------------------------------
| This script is an optimised version of the original, it isn't representative of the way we dealt
| with the subject, and so, doesn't contain food for thought and analysis with graphics, statistical data.
| To have an overview of this part, you can refer to the report and the file annexe.py
-----------------------------------------------------------------------------------------------------------------
"""

# ---- Sets ------------------------------------------------------------------------------------------------------------

# a set of hyperparameters for the Support Vector Machine model
hyper_params_SVM = {
    'C': [100, 400, 700],
    'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    'gamma': ['scale', 'auto', 1e-3, 1e-4],
    'degree': [1, 2, 3, 4]
}

# a set of hyperparameters for the Support Vector Machine model just for show how it works
hyper_params_SVM_min = {
    'C': [100, 400],
    'kernel': ['linear', 'sigmoid'],
    'gamma': [1e-3, 1e-4],
    'degree': [1, 2, 3]
}

# a set of hyperparameters for the RandomForest model
hyper_params_RFC = {
    "max_depth": [15, 20, 25], "min_samples_split": [7, 10, 15],
    "max_leaf_nodes": [130,150, 170], "min_samples_leaf": [2, 3, 5],
    "n_estimators": [500, 750, 900], "max_features": ["sqrt"]
}

# a set of hyperparameters for the MLP model
hyper_params_MLP = {
    # "hidden_layer_sizes": [(250, 150, 10, 2), (50, 15, 5, 2), (40, 10, 2), (50, 20, 5 ,2)],
    "hidden_layer_sizes": [(50, 20, 5 ,2), (50, 20, 10, 5, 2)],
    "activation": ["relu", "sigmoid", "tanh"],
    "solver": ["adam"],
    "shuffle": [False],
    "random_state": [42],
    "early_stopping": [True],
    "n_iter_no_change": [15],
    "verbose": [True]
}

# list of teams participating in the FIFA championship
list_team = ['France','England','Brazil','Cameroon','Tunisia','Netherlands','Japan','Hungary','Estonia','Ghana','Italy','Mexico','Spain','Algeria','Andorra','Argentina','Taiwan','Congo','Croatia','Colombia','Jamaica','Jordan',"Kuwait",'Latvia','Laos','Lesotho','Liberia','Macau','Nepal','Peru','Russia','Rwanda']

# ---- sub functions ---------------------------------------------------------------------------------------------------

"""
/ return a dataset with matches since the year 'year'
/ param rera : the dataset
/ param year : the year from we keep our data
"""
def keep_data_since_year(rera, year: int):

    # we don't want to modify the original datasets
    df = rera.copy(deep=True)
    # keep data(matches) since the year "year"
    df = df[df['date'] > str(year-1)]
    # don't need anymore the date
    df.drop(columns={'date'}, inplace=True)

    return df


"""
/ evaluate the model and print some usefull metrics for Classification models
/ param model : the model
/ param model_name : the name of the model
/ param X_train : train data
/ param y_train : train label
/ param X_test : test data
/ param y_test : test label
"""
def evaluation_classification(model, model_name, X_train, X_test, y_train, y_test):

    y_pred = model.predict(X_test)

    # the metrics
    report = classification_report(y_test, y_pred)

    # display the learning curve
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=3, scoring='accuracy', train_sizes=np.linspace(0.1, 1, 10))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]})
    ax1.set_title(model_name)
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("number of data in the dataset")
    ax1.plot(N, train_score.mean(axis=1), label='training_set')
    ax1.plot(N, val_score.mean(axis=1), label='validation_set')
    ax1.legend(loc='upper right')

    # Afficher le rapport de classification en-dessous du graphique
    ax2.text(0.5, 0.5, report, transform=plt.gca().transAxes, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    ax2.axis('off') # desactive axes

    # Ajuster l'espace entre les sous-graphiques
    plt.subplots_adjust(hspace=0.7)

    plt.show()


"""
/ evaluate the model and print some usefull metrics for Regression models
/ param model : the model
/ param model_name : the name of the model
/ param X_train : train data
/ param y_train : train label
/ param X_test : test data
/ param y_test : test label
"""
def evaluation_regression(model, model_name, X_train, X_test, y_train, y_test):

    y_pred = model.predict(X_test)

    # the metrics
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred), 2)
    report = {"mean_abs_error": mae, "mean_sqr_error": mse, "r2_score": r2}

    # display the learning curve
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1, 10))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]})
    ax1.set_title(model_name)
    ax1.set_ylabel("neg_mean_squared_error")
    ax1.set_xlabel("number of data in the dataset")
    ax1.plot(N, train_score.mean(axis=1), label='training_set')
    ax1.plot(N, val_score.mean(axis=1), label='validation_set')
    ax1.legend(loc='upper right')

    # Afficher le rapport de classification en-dessous du graphique
    ax2.text(0.5, 0.5, report, transform=plt.gca().transAxes, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    ax2.axis('off') # desactive axes

    # Ajuster l'espace entre les sous-graphiques
    plt.subplots_adjust(hspace=0.7)

    plt.show()

"""
/ return the X_train, X_test, y_train_classif, y_test_classif, y_train_regres, y_test_regres from rera
/ for classification and regression task
/ param rera : the dataframe to split
/ param test_size : the length pourcentage of the test_set
"""
def Xy_train_test_split(rera, test_size):

    # slit our data without shuffle in order to test on recent years
    trainset, testset = train_test_split(rera, test_size=test_size, shuffle=False)

    X_train = trainset.drop(columns={'winner', 'home_score', 'away_score'}, axis=1)
    X_test = testset.drop(columns={'winner', 'home_score', 'away_score'}, axis=1)
    y_train_classif = trainset['winner']
    y_test_classif = testset['winner']
    y_train_regres = trainset[['home_score', 'away_score']]
    y_test_regres = testset[['home_score', 'away_score']]

    return X_train, X_test, y_train_classif, y_test_classif, y_train_regres, y_test_regres


"""
/ test a list of model and evaluate them by displaying some usefull metrics
/ param X_train : train data
/ param y_train : train label
/ param X_test : test data
/ param y_test : test label
"""
def test_model(X_train, X_test, y_train_classif, y_test_classif, y_train_regres, y_test_regres):
    warnings.filterwarnings("ignore")  # Désactiver tous les avertissements

    # all the models we want to try
    RandomForestC = RandomForestClassifier(random_state=0)
    AdaBoost = AdaBoostClassifier(random_state=0)
    SVM = SVC(random_state=0)
    MLP = MLPClassifier(random_state=0)
    RandomForestR = RandomForestRegressor(random_state=0)

    dict_of_models_classif = {'RandomForestClassifier': RandomForestC, 'AdaBoost': AdaBoost, 'SVM': SVM, 'MLP': MLP}
    dict_of_models_regres = {'RandomForestRegression': RandomForestR}

    nb_model = len(dict_of_models_classif) + len(dict_of_models_regres)
    current_model = 1

    # evaluation of classifier models
    for name, model in dict_of_models_classif.items():
        print(f'evaluation of {name} ({current_model}/{nb_model}):')
        model.fit(X_train, y_train_classif)
        evaluation_classification(model, name, X_train, X_test, y_train_classif, y_test_classif)
        current_model += 1

    # evaluation of regression models
    for name, model in dict_of_models_regres.items():
        print(f'evaluation of {name} ({current_model}/{nb_model}):')
        model.fit(X_train, y_train_regres)
        evaluation_regression(model, name, X_train, X_test, y_train_regres, y_test_regres)
        current_model += 1

    warnings.filterwarnings("default")  # Réactiver les avertissements si nécessaire


"""
/ return a match built thanks to "home_team","away_team","tournament", "neutral"
/ param reraBase : the dataset which contains all the matchd
/ param hom_team :  home team name
/ param away_team :  away team name
/ param tournament : type of tournament
/ param neutral : if hom_team don't play at home
"""
def build_match_to_predict(rankingData, reraBase, home_team, away_team, tournament, neutral, scaler):

    # we don't want to modify the original datasets
    base = reraBase.copy(deep=True)

    date = datetime.now()
    tournament = get_rank_tournament(tournament)
    teams = frozenset([home_team, away_team])
    base['teams'] = base.apply(lambda row: frozenset([row['home_team'], row['away_team']]), axis=1)
    base['date'] = pd.to_datetime(base['date'])
    partial_match = pd.Series([date, teams, home_team, away_team, tournament], index=["date", "teams", "home_team", "away_team", "tournament"])

    rankingHome = rankingData[rankingData['country_full'] == home_team].iloc[-1]
    rankingAway = rankingData[rankingData['country_full'] == away_team].iloc[-1]

    # get the features
    win_score_5last_enc_home, win_score_5last_enc_away = get_win_score_5last_enc(base, partial_match)
    avg_goals_last_five_home, avg_goals_last_five_away = get_avg_goal_last_5matches(base, partial_match)
    win_rate_tournament_home, win_rate_tournament_away = get_win_rate_tournament(base, partial_match)

    # build the match to return
    match = {
        'tournament': tournament,
        'neutral': int(neutral),
        'rank_home': rankingHome['rank'],
        'total_points_home': rankingHome['total_points'],
        'rank_change_home': rankingHome['rank_change'],
        'rank_away': rankingAway['rank'],
        'total_points_away': rankingAway['total_points'],
        'rank_change_away': rankingAway['rank_change'],
        'win_score_5last_enc_home': win_score_5last_enc_home,
        'win_score_5last_enc_away': win_score_5last_enc_away,
        'diff_goals_last_enc': get_diff_goals_last_enc(base, partial_match),
        'avg_goals_last_five_home': avg_goals_last_five_home,
        'avg_goals_last_five_away': avg_goals_last_five_away,
        'win_rate_tournament_home': win_rate_tournament_home,
        'win_rate_tournament_away': win_rate_tournament_away
    }

    # normalize the data of the match
    match = pd.DataFrame([match])
    match = pd.DataFrame(scaler.transform(match), columns=match.columns)

    return match

# ---- main function ---------------------------------------------------------------------------------------------------

def training():

    print("\n[PREPROCESSING] :\n")

    # get both datasets after preprocessed the data
    rera, reraBase, rankingData = preprocessing()

    # save both datasets
    rera.to_csv("rera.csv", index=False)
    reraBase.to_csv("reraBase.csv", index=False)
    rankingData.to_csv("rankingData.csv", index=False)

    # keep only the data from 2010 (for having tested, the cutting year don't change many things)
    rera = keep_data_since_year(rera, 2010)

    # normalise the data
    rera_without_target_col = rera.drop(columns={'winner', 'home_score', 'away_score'}).columns
    scaler = MinMaxScaler()
    scaler.fit(rera[rera_without_target_col])
    rera[rera_without_target_col] = scaler.transform(rera[rera_without_target_col])

    # slit our data
    X_train, X_test, y_train_classif, y_test_classif, y_train_regres, y_test_regres = Xy_train_test_split(rera, 0.2)

    # --------------- test we've done to find the best model, parameters -----------------------------------------------

    print("\n[TESTING MODELS] :\n")
    # evaluate all the models to have an overview on those that could work well and those that don't seems to work well for this project
    test_model(X_train, X_test, y_train_classif, y_test_classif, y_train_regres, y_test_regres)

    print("\n[HYPERPARAMS SELECTION] :\n")
    # here we choose the SVM model which is one of the more promising model we tested,
    # we processed a gridSearch on it (this hyper_parmas isn't the same we used because it would take too much time)
    # the right is called hyper_params_SVM (you can find him all above the document)
    grid = GridSearchCV(SVC(random_state=0), hyper_params_SVM_min, scoring='accuracy', cv=2, verbose=2)
    grid.fit(X_train, y_train_classif)
    print("\nbest_param for the SVM:", grid.best_params_)
    y_pred = grid.predict(X_test)
    print("\nclassification_report :\n")
    print(classification_report(y_test_classif, y_pred))

    # --------------- the best model with the best parameters we obtain ------------------------------------------------

    # the best parameters for SVM we find with hyper_params_SVM : {'C': 700, 'degree': 1, 'gamma': 0.001, 'kernel': 'sigmoid'}
    SVM = SVC(random_state=0, C=700, degree=1, gamma=0.01, kernel='sigmoid', probability=True)

    # train the model
    SVM.fit(X_train, y_train_classif)

    # evaluate it
    evaluation_classification(SVM, "SVM", X_train, X_test, y_train_classif, y_test_classif)

    print("\n[TOURNAMENT PREDICTION] :\n")
    # tournament creation
    write_file_csv(list_team, "tournament_teams.csv", 8)
    root = tk.Tk()
    app = ArbreTournoi(root, SVM, build_match_to_predict, rankingData, reraBase, scaler)
    print("test your championship in the window which just appears :)")
    root.mainloop()











