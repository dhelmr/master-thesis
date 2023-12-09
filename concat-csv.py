import argparse

import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Input files", nargs="+")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--axis", default=0)

    args = parser.parse_args()
    dfs = []
    for f in args.input:
        df = pd.read_csv(f, index_col=False)
        dfs.append(df)
    df = pd.concat(dfs, axis=args.axis)
    df.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()