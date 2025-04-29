
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# Preprocess the dataset
def preprocess_dataset(dataset_path):
    player_df = pd.read_csv(dataset_path)
    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling',
               'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
               'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance', 'ShotPower',
               'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']

    player_df = player_df[numcols + catcols].dropna()
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)

    y = traindf['Overall'] >= 87
    X = traindf.drop(columns=['Overall'])
    num_feats = 30

    return X, y, num_feats


# Feature selection methods
def cor_selector(X, y, num_feats):
    cor_list = [np.corrcoef(X[col], y)[0, 1] if not np.isnan(np.corrcoef(X[col], y)[0, 1]) else 0 for col in X.columns]
    cor_support = [False] * len(cor_list)
    for idx in np.argsort(np.abs(cor_list))[-num_feats:]:
        cor_support[idx] = True
    return cor_support, X.columns[cor_support]


def chi_squared_selector(X, y, num_feats):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    chi_selector = SelectKBest(score_func=chi2, k=num_feats).fit(X_scaled, y)
    return chi_selector.get_support(), X.columns[chi_selector.get_support()]


def rfe_selector(X, y, num_feats):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    rfe = RFE(estimator=model, n_features_to_select=num_feats, step=1).fit(X_scaled, y)
    return rfe.get_support(), X.columns[rfe.get_support()]


def embedded_log_reg_selector(X, y, num_feats):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, solver='liblinear').fit(X_scaled, y)
    selector = SelectFromModel(estimator=model, max_features=num_feats, threshold=-np.inf, prefit=True)
    return selector.get_support(), X.columns[selector.get_support()]


def embedded_rf_selector(X, y, num_feats):
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    selector = SelectFromModel(estimator=model, max_features=num_feats, threshold=-np.inf, prefit=True)
    return selector.get_support(), X.columns[selector.get_support()]


def embedded_lgbm_selector(X, y, num_feats):
    model = LGBMClassifier(n_estimators=100, random_state=42).fit(X, y)
    selector = SelectFromModel(estimator=model, max_features=num_feats, threshold=-np.inf, prefit=True)
    return selector.get_support(), X.columns[selector.get_support()]


# Main function
def autoFeatureSelector(dataset_path, methods=[]):
    X, y, num_feats = preprocess_dataset(dataset_path)

    feature_selection_results = {}

    if 'pearson' in methods:
        cor_support, _ = cor_selector(X, y, num_feats)
        feature_selection_results['pearson'] = cor_support
    if 'chi-square' in methods:
        chi_support, _ = chi_squared_selector(X, y, num_feats)
        feature_selection_results['chi-square'] = chi_support
    if 'rfe' in methods:
        rfe_support, _ = rfe_selector(X, y, num_feats)
        feature_selection_results['rfe'] = rfe_support
    if 'log-reg' in methods:
        lr_support, _ = embedded_log_reg_selector(X, y, num_feats)
        feature_selection_results['log-reg'] = lr_support
    if 'rf' in methods:
        rf_support, _ = embedded_rf_selector(X, y, num_feats)
        feature_selection_results['rf'] = rf_support
    if 'lgbm' in methods:
        lgbm_support, _ = embedded_lgbm_selector(X, y, num_feats)
        feature_selection_results['lgbm'] = lgbm_support

    all_support = pd.DataFrame(feature_selection_results, index=X.columns)
    all_support['Total'] = all_support.sum(axis=1)  
    sorted_support = all_support.sort_values('Total', ascending=False)
    best_features = sorted_support.head(num_feats).index.tolist()
    sorted_support_styled = sorted_support.head(num_feats).style.set_caption("Best Features Selected").background_gradient(cmap="Blues").format({"Total": "{:.0f}"})
    display(sorted_support_styled) 

    return best_features


# Run script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Selection Tool")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--methods", nargs='+', required=True, help="List of feature selection methods to use")

    args = parser.parse_args()
    dataset_path = args.dataset
    methods = args.methods

    best_features = autoFeatureSelector(dataset_path, methods)
    print("Best features selected:", best_features)
