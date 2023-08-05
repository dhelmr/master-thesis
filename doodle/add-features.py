import argparse
import math

import numpy
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--output", "-o", required=True)
parser.add_argument("--columns", "-c", required=True, nargs="+")
parser.add_argument("--skip", "-s", required=True, nargs="+", default=["f1_cfa"])
args = parser.parse_args()

df = pd.read_csv(args.input, index_col=False)

def test_and_add(df, name, new_col):
    try:
        if numpy.isinf(new_col).any() or numpy.isnan(new_col).any():
            print("Skip", name)
        else:
            df[name] = new_col
    except Exception as e:
        print(e)

columns = [col for col in df.columns]
pbar = tqdm(columns)
for col in pbar:
    if col in args.skip or "." in col:
        continue
    test_and_add(df, name=f"log2-{col}", new_col=df[col].apply(lambda x: math.log2(x+1)))
    test_and_add(df, name=f"log10-{col}", new_col=df[col].apply(lambda x: math.log10(x+1)))
    test_and_add(df, name=f"inverse-{col}", new_col=df[col].apply(lambda x: 1/(x+1)))
    test_and_add(df, name=f"squared-{col}", new_col=df[col].apply(lambda x: x*x))
    test_and_add(df, name=f"sqrt-{col}", new_col=df[col].apply(lambda x: math.sqrt(x)))
    for col2 in args.columns:
        test_and_add(df, name = f"{col}/{col2}",  new_col = df[col] / df[col2])
#        test_and_add(df, name=f"{col}*{col2}", new_col=df[col] * df[col2])
#        test_and_add(df, name=f"{col}/{col2}^2", new_col=df[col] / (df[col2]*df[col2]))
#        test_and_add(df, name=f"{col}^2/{col2}", new_col=(df[col]*df[col]) / (df[col2]))

df.to_csv(args.output)




#for index, row in corr.iterrows():
#    print(row[args.target], index)

