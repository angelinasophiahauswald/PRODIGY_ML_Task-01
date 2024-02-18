import sys
import pandas as pd

def read_data(data):
    df = pd.read_csv(data)
    return df

if __name__ == "__main__":
    data = sys.argv[1]
    print(read_data(data))
    