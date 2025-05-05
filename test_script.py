import os
import pandas as pd
from code_executor_py import VenvExecutor
from langchain_openai import ChatOpenAI


func_code = """

from sklearn.preprocessing import StandardScaler
import pandas as pd

def hello():
    print("world")

def hello2():
    print("world")

def process_data(data_df):
    a = hello()
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns)
    """

executor = VenvExecutor() 
# Create executable function
process_data = executor.create_executable(func_code)


# Prepare test data
test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Execute the function and get the result
result = process_data(data_df = test_data)

print(result)

