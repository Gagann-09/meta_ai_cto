import traceback
import pandas as pd
from agents.automl_agent import train_model

try:
    df = pd.read_csv('data/adult_income.csv').head(100)
    train_model(df, 'income', time_limit=10)
except Exception as e:
    traceback.print_exc()
