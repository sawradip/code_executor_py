import pytest
import pandas as pd
from code_executor import VenvExecutor

def test_add_function():
    executor = VenvExecutor(
        # llm=some_llm
    ) 

    # Define function
    func_code = """
def add(x, y):
    return x + y
"""

    # Create executable function
    process_data = executor.create_executable(func_code)
    
    # Use function
    result = process_data({'x': 2, 'y': 3})
    
    assert result == 5


    def test_process_data_function():
        func_code = """
# from pandas import dsokfnls
# import fkgjg
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

        # Create executable function
        process_data = executor.create_executable(func_code)


        # Prepare test data
        test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Execute the function and get the result
        result = process_data({"data_df": test_data})

        # Expected result after scaling
        expected_result = pd.DataFrame([[-1.224745, 0, 1.224745], [-1.224745, 0, 1.224745]], columns=['A', 'B'])

        # Assert that the result matches the expected output
        pd.testing.assert_frame_equal(result, expected_result)



    