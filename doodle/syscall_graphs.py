import plotly.express as px
import pandas as pd
import sys

from tqdm import tqdm#

if len(sys.argv) != 4:
    print("Usage: [analysis_df] [syscall df] [output csv]")
    exit(1)

ngram_analyis_df = pd.read_csv(sys.argv[1])
syscall_df = pd.read_csv(sys.argv[2])

def map_n_syscalls(x):
    try:
        return int(x)
    except ValueError as e:
        print(e)
        return -1
syscall_df["metrics.dataloader.training_syscalls"] = syscall_df["metrics.dataloader.training_syscalls"].apply(map_n_syscalls)

def get_performance(syscall_num: int, scenario: str, num_attacks: int):
    selected = syscall_df.loc[(syscall_df["metrics.dataloader.training_syscalls"] == int(syscall_num)) &
                          (syscall_df["params.dataloader.scenario"] == scenario) &
                          (syscall_df["params.dataloader.num_attacks"] == int(num_attacks))]["metrics.ids.f1_cfa"]
    if len(selected) >= 1:
        aslist = selected.tolist()
        asset = set(aslist)
        if len(asset) > 1:
            raise ValueError(asset.__str__())
        return aslist[0]
    return None

data = []
for _, r in tqdm(ngram_analyis_df.iterrows(), total=len(ngram_analyis_df)):
    f1 = get_performance(r["syscalls"], r["dataloader.scenario"], r["dataloader.num_attacks"])
    if f1 is None:
        continue
    row_dict = r.to_dict()
    row_dict["f1_cfa"] = f1
    data.append(row_dict)

df = pd.DataFrame(data)
df.to_csv(sys.argv[3])

#fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
#fig.write_html('first_figure.html', auto_open=True)
