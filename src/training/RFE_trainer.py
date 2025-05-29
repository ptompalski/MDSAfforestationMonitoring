import click
import pandas as pd
from pathlib import Path
import joblib

@click.command()
@click.option(
    '--model_path',
    type=click.Path(dir_okay=False),
    required=True,
    help='Path to model pipeline.'
)
@click.option(
    '--training_data',
    type=click.Path(dir_okay=False),
    required=True,
    help='Path to training parquet file'
)
@click.option(
    '--output_dir',
    type=click.Path(file_okay=False),
    help='Directory to save the trained results')
def main(model_path,training_data,output_dir):
    '''
    CLI for basic training of models with RFE or RFECV feature selection methods.
    Simplified training with no hyperparameter optimization.
    '''
    # load data and model
    train_df = pd.read_parquet(training_data)
    model = joblib.load(model_path)
    
    # prepare data
    train_df = train_df.dropna()
    X = train_df.drop(columns='target'); y = train_df['target']
    site_ids = train_df['ID']
    
    # run fitting/feature selection
    click.echo('Training model and selecing features...')
    model.fit(X,y,rfecv__groups=site_ids)
    click.echo('Done.')
    
    # save model
    model_name = f"fitted_{model_path.split('/')[-1]}"
    output_path = Path(output_dir)/model_name
    joblib.dump(model,output_path)
    click.echo(f'Fitted model saved to {output_path}')
    
if __name__ == '__main__':
    main()