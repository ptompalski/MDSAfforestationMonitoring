from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from src.data_split import data_split
import click
import pandas as pd


def build_pipline():

    # Create a pipeline with a standard scaler and a random forest classifier
    pipeline = Pipeline([
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

def train_and_test_model(X_train, y_train, X_test, y_test, pipeline):

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    return score

def cross_validate():
    """
    Perform cross-validation on the model.
    """
    pass

@click.command()
@click.option('--input_path', type=click.Path(exists=True), required=True, help='Path to train data. Expecting parquet format.')
def main(input_path):
    df = pd.read_parquet(input_path)

    df_train, df_test = data_split(df)
    X_train, X_test = df_train.drop(columns=['target']), df_train['target']
    y_train, y_test = df_test.drop(columns=['target']), df_test['target']

    pipeline = build_pipline()
    score = train_and_test_model(X_train, y_train, X_test, y_test, pipeline)
    print(f'Model score: {score}')