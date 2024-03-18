import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
from catboost import CatBoostClassifier
import optuna
from optuna.integration import CatBoostPruningCallback
optuna.logging.disable_default_handler()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger
import random
import argparse

# Инициализация логгера
logger.remove()
from pathlib import Path
log_file = Path("log/model_building.log")
log_file.unlink(missing_ok=True)
log_format = {
    "format": "{time:YYYY-MM-DD HH:mm:ss} [{level}] Message: {message}",
    "colorize": True,
}
logger.add(sys.stdout, **log_format)
logger.add(
    log_file,
    **log_format,
    backtrace=True,
    diagnose=False,
    rotation="64 MB",
    retention="90 days",
)
logger.info("Logger initialized successfully!")

# мой рандомный сид
my_seed = random.randint(10, 25000)
np.random.seed(my_seed)





# разделение фичи на несколько
def split_feature(df, feature, new_features, sep):
    try:
        df[new_features] = df[feature].str.split(sep, expand=True)
        logger.info(f'Splited feature {feature} to {new_features} in dataframe {df.name}')
        return df
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed split data in dataframe {df.name}")

# удаление неинформативных фичей
def drop_features(df, features):
    try:
        df.drop(features, axis=1, inplace=True)
        logger.info(f'Droped features {features} in dataframe {df.name}')
        return df
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed drop data in dataframe {df.name}")

# изменение типа данных фичей
def cast_feature(df, feature, cast):
    try:
        df[feature] = df[feature].astype(cast)
        logger.info(f'Changed type of {feature} to {cast} in dataframe {df.name}')
        return df
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed cast data in dataframe {df.name}")
    
def impute_cryo_sleep(df):
    """Вставляем данные в столбик CryoSleep по логике: если пассажир пользовался услугами, то он не спит и наоборот"""
    try:
        df.loc[
            ((df['RoomService'] == 0.0) | df['RoomService'].isnull()) & 
            ((df['FoodCourt'] == 0.0) | df['FoodCourt'].isnull()) & 
            ((df['ShoppingMall'] == 0.0) | df['ShoppingMall'].isnull()) & 
            ((df['Spa'] == 0.0) | df['Spa'].isnull()) &
            ((df['VRDeck'] == 0.0) | df['VRDeck'].isnull()) &
            (df['CryoSleep'].isnull()), 
            'CryoSleep'
        ] = True
        
        df.loc[
            ((df['RoomService'] > 0.0) | 
            (df['FoodCourt'] > 0.0) | 
            (df['ShoppingMall'] > 0.0) | 
            (df['Spa'] > 0.0) |
            (df['VRDeck'] > 0.0)) & (df['CryoSleep'].isnull()), 
            'CryoSleep'
        ] = False
        logger.info(f'We inputed cryo sleep data in dataframe {df.name}')
        return df
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed input cryo sleep data in dataframe {df.name}")

def impute_home_planet_by_deck(df):
    """Вставляем данные в столбик HomePlanet по соответствию с Deck"""
    try:
        df.loc[
            (df['Deck'] == 'G') & (df['HomePlanet'].isnull()), 
            'HomePlanet'
        ] = 'Earth'
        
        europa_decks = ['A', 'B', 'C', 'T']
        df.loc[
            (df['Deck'].isin(europa_decks)) & (df['HomePlanet'].isnull()), 
            'HomePlanet'
        ] = 'Europa'
        df.loc[
            (df['Deck'] == 'F') & (df['HomePlanet'].isnull()), 
            'HomePlanet'
        ] = 'Mars'
        logger.info(f'We inputed home planet by deck data in dataframe {df.name}')
        return df
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed input home planet by deck data in dataframe {df.name}")

def impute_deck_by_home_planet(df):
    """Вставляем данные в столбик Deck по соответствию с HomePlanet"""
    home_planet_deck = df.groupby(['HomePlanet', 'Deck']).size().unstack().fillna(0)
    earth = home_planet_deck.loc['Earth']
    earth_proba = list(earth / sum(earth))
    europa = home_planet_deck.loc['Europa']
    europa_proba = list(europa / sum(europa))
    mars = home_planet_deck.loc['Mars']
    mars_proba = list(mars / sum(mars))
    decks = df['Deck'].unique()
    deck_values = sorted(decks[~pd.isnull(decks)])
    planet_proba = dict(zip(['Earth', 'Mars', 'Europa'], [earth_proba, mars_proba, europa_proba]))
    try:
        for planet in planet_proba.keys():
            planet_null_decks_shape = df.loc[(df['HomePlanet'] == planet) & (df['Deck'].isnull()), 'Deck'].shape[0]
            df.loc[(df['HomePlanet'] == planet) & (df['Deck'].isnull()), 'Deck'] = np.random.choice(deck_values, planet_null_decks_shape, p=planet_proba[planet])
        logger.info(f'We inputed deck data by home planet in dataframe {df.name}')
        return df
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed input deck data by home planet in dataframe {df.name}")
    
def impute_age_by_planet(df):
    """Вставляем данные в столбик Age по среднему возрасту на HomePlanet"""
    try:
        for planet in ['Europa', 'Earth', 'Mars']:
            planet_median = df[df['HomePlanet'] == planet]['Age'].median()
            df.loc[(df["Age"].isnull()) & (df["HomePlanet"] == planet),"Age"] = planet_median
        logger.info(f'We inputed median age by home planet in dataframe {df.name}')
        return df
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed input median age by home planet in dataframe {df.name}")

def impute_usluga_by_age(df):
    """Вставляем данные в столбик сервисов по среднему возрасту их использования"""
    try:
        uniq_age = df['Age'].unique()
        uslugi = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for age in uniq_age:
            for usluga in uslugi:
                usluga_median = df[df['Age'] == age][usluga].median()
                df.loc[(df[usluga].isnull()) & (df['Age'] == age), usluga] = usluga_median
        logger.info(f'We inputed median usage of service by age in dataframe {df.name}')
        return df
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed input median age by home planet in dataframe {df.name}")
    
def input_median_data_in_numerical_columns(df):
    """Вставляем средние данные в столбики с числовыми значениями"""
    numerical_columns = df.describe().columns
    try:
        for col in numerical_columns:
            si = SimpleImputer(strategy='median')
            df[col] = si.fit_transform(df[col].values.reshape(-1, 1))
        logger.info(f'We inputed median data in numerical columns in dataframe {df.name}')
        return df
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed input median data in numerical columns in dataframe {df.name}")

def input_most_frequent_data_in_categorical_columns(df):
    """Вставляем средние данные в столбики с категориальными значениями"""
    numerical_columns = df.describe().columns
    categorical_columns = set(df.columns) - set(numerical_columns)
    try:
        for col in categorical_columns:
            si = SimpleImputer(strategy='most_frequent')
            df[[col]] = si.fit_transform(df[[col]])
        logger.info(f'We inputed most frequent data in categorical columns in dataframe {df.name}')
        return df
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed input most frequent data in categorical columns in dataframe {df.name}")

def log_transform_data(df):
    numerical_columns = df.describe().columns
    for col in numerical_columns[1:-1]:
        df[col] = np.log(1 + df[col])
    return df


class My_Classifier_Model:
    def __init__(self, results_dir="./data/results.csv", model_dir="./model/"):
        self.model = None
        self.results_dir = results_dir
        self.model_dir = model_dir

    def train(self, dataset_filename):
        # подгрузка файлов
        try:
            train = pd.read_csv(f'{dataset_filename}')
            X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1:]
            y_train = y_train.astype(bool).astype(int)
            logger.info(f'Files csv load: SUCCESS')
        except Exception as e:
            logger.error(e)
            raise Exception("Files csv load: FAIL")
        X_train.name = 'X_train'
        logger.info(f'Missed data count\n{X_train.isnull().sum()}')
        X_train = split_feature(X_train, 'PassengerId', ['GroupId', 'IdWithinGroup'], '_')
        X_train = split_feature(X_train, 'Cabin', ['Deck', 'Num', 'Side'], '/')
        X_train = drop_features(X_train, ['Name', 'PassengerId', 'Cabin', 'VIP', 'Num'])
        X_train = cast_feature(X_train, 'GroupId', 'float')
        X_train = impute_cryo_sleep(X_train)
        X_train = impute_home_planet_by_deck(X_train)      
        X_train = impute_deck_by_home_planet(X_train)
        X_train = impute_age_by_planet(X_train)
        X_train = impute_usluga_by_age(X_train)
        X_train = input_median_data_in_numerical_columns(X_train)
        X_train = input_most_frequent_data_in_categorical_columns(X_train)
        logger.info(f'Missed data count\n{X_train.isnull().sum()}')
        X_train = log_transform_data(X_train)
        try:
            X_train = pd.get_dummies(X_train)
            logger.info(f'Data type changed to dummies type')
        except Exception as e:
            logger.error(e)
            raise Exception(f"Failed change data type to dummies type")

        def objective(trial):
            """Функция подбора оптимальных параметров модели"""
            _X_train, X_valid, _y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)
            params = {
                'objective': trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 1, 12),
                'boosting_type': trial.suggest_categorical('boosting_type',['Ordered', 'Plain']),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'used_ram_limit': '8gb',
                'eval_metric': 'Accuracy',
                'logging_level': 'Silent',
                'random_seed': 21
            }
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
            clf = CatBoostClassifier(**params)
            pruning_callback = CatBoostPruningCallback(trial, 'Accuracy')
            clf.fit(_X_train, _y_train, eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds=100, callbacks=[pruning_callback],)
            pruning_callback.check_pruned()
            predictions = clf.predict(X_valid)
            prediction_labels = np.rint(predictions)
            accuracy = accuracy_score(y_valid, prediction_labels)
            return accuracy
        best_trials = pd.DataFrame(columns=[
                'objective',
                'colsample_bylevel',
                'depth',
                'boosting_type',
                'bootstrap_type',
                'best_value'
            ]
        )
        studies = 1
        trials = 100
        best_accuracy = 0.0
        logger.info(f'Start finding best parameters for model...')
        for n in range(studies):
            logger.info(f'Studies step #{n+1}')
            study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction='maximize')
            study.optimize(objective, n_trials=trials, timeout=600)
            if study.best_value > best_accuracy:
                logger.info(f'Best value = {study.best_value}')
                trial = study.best_trial
                logger.info(f'Best parameters')
                for k, v in trial.params.items():
                    logger.info(f'{k}: {v}')
                best_trials.loc[n] = study.best_trial.params
                best_trials['best_value'].loc[n] = study.best_value
                best_accuracy = study.best_value
            else:
                logger.info(f'Accuracy lower than prev study step')

        logger.info(f'Fitting model with best parameters...')
        best_trial = best_trials.sort_values('best_value', ascending=False).loc[0]
        clf = CatBoostClassifier(**best_trial[:-1], logging_level='Silent', random_seed=my_seed)
        logger.info(f'Fitting model with best parameters... Done!')
        clf.fit(X_train, y_train.astype(int))
        clf.save_model(f'models/catboost_model.bin')

    def predict(self, dataset_filename):
        clf = CatBoostClassifier()
        clf.load_model(f'models/catboost_model.bin')
        X_test = pd.read_csv(f'{dataset_filename}')
        X_test.name = 'X_test'
        X_test = split_feature(X_test, 'PassengerId', ['GroupId', 'IdWithinGroup'], '_')
        X_test = split_feature(X_test, 'Cabin', ['Deck', 'Num', 'Side'], '/')
        X_test = drop_features(X_test, ['Name', 'PassengerId', 'Cabin', 'VIP', 'Num'])
        X_test = cast_feature(X_test, 'GroupId', 'float')
        X_test = impute_cryo_sleep(X_test)
        X_test = impute_home_planet_by_deck(X_test)
        X_test = impute_deck_by_home_planet(X_test)
        X_test = impute_age_by_planet(X_test)
        X_test = impute_usluga_by_age(X_test)
        X_test = input_median_data_in_numerical_columns(X_test)
        X_test = log_transform_data(X_test)
        try:
            X_test = pd.get_dummies(X_test)
            logger.info(f'Data type changed to dummies type')
        except Exception as e:
            logger.error(e)
            raise Exception(f"Failed change data type to dummies type")
        logger.info(f'Make predictions...')
        try:
            predicted = clf.predict(X_test)
            logger.info(f'Make predictions... Done!')
        except:
            logger.warning(f'Error while making predictions...')
        sub = pd.DataFrame()
        sub['PassengerId'] = pd.read_csv('data/test.csv')['PassengerId']
        sub['Transported'] = pd.Series(predicted).astype(bool)
        sub.to_csv('submission.csv', index=False)
        logger.info(f'Make predictions... Done! Saved to submission.csv')


def main():
    parser = argparse.ArgumentParser(description="Train and predict using My_Classifier_Model")
    parser.add_argument("mode", choices=["train", "predict"], help="Mode: train or predict")
    parser.add_argument("dataset", help="Path to the dataset file")
    parser.add_argument("--results-dir", default="./data/results.csv", help="Directory to save results (default: ./data/results.csv)")
    parser.add_argument("--model-dir", default="./model/", help="Directory to save trained model (default: ./model/)")
    args = parser.parse_args()

    classifier = My_Classifier_Model(results_dir=args.results_dir, model_dir=args.model_dir)

    if args.mode == "train":
        classifier.train(args.dataset)
    elif args.mode == "predict":
        classifier.predict(args.dataset)

if __name__ == "__main__":
    main()

# For train: model.py train 'data/train.csv'
# For predict: model.py predict 'data/test.csv'