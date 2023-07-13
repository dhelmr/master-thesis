import plotly.express as px
import pandas as pd
import sys

from tqdm import tqdm

ngram_analyis_df = pd.read_csv(sys.argv[1])
syscall_df = pd.read_csv(sys.argv[2])


def get_performance(syscall_num: int, scenario: str, num_attacks: int):
    selected = syscall_df.loc[(syscall_df["params.dataloader.max_syscalls_training"] == int(syscall_num)) &
                          (syscall_df["params.dataloader.scenario"] == scenario) &
                          (syscall_df["params.dataloader.num_attacks"] == int(num_attacks))]["metrics.ids.f1_cfa"]
    if len(selected) >= 1:
        aslist = selected.tolist()
        asset = set(aslist)
        if len(asset) > 1:
            raise ValueError(asset.__str__())
        print(aslist)
        return aslist[0]
    return None

data = []
for _, r in tqdm(ngram_analyis_df.iterrows(), total=len(ngram_analyis_df)):
    f1 = get_performance(r["syscalls"], r["dataloader.scenario"], r["dataloader.num_attacks"])
    if f1 is None:
        continue
    data.append({
        "scenario": r["dataloader.scenario"],
        "syscalls":  r["syscalls"],
        "conditional_entropy": r["conditional_entropy"],
        "entropy": r["entropy"],
        "simpson_index": r["simpson_index"],
        "u/t": r["u/t"],
        "unique": r["unique"],
        "num_attacks": r["num_attacks"],
        "ngram_length": r["ngram_size"],
        "f1_cfa": f1
    })

df = pd.DataFrame(data)
df.to_csv("entropyXsyscalls.csv")

#fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
#fig.write_html('first_figure.html', auto_open=True)
