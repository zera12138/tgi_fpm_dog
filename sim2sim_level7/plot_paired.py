"""Outcome distribution and progress ratio for the paired A/B/C runs."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent; OUT=HERE/"perception_out/paired"
CONDS=("A","B","C"); COLORS={"pass":"#2a9d5c","stall":"#e0a800","fall":"#d62728"}

def main():
    sc=sorted({f.name.split("_")[0] for f in OUT.glob("*_*_*.json")})
    fig,axes=plt.subplots(1,len(sc),figsize=(4.6*len(sc),3.6),squeeze=False)
    for ax,name in zip(axes[0],sc):
        rows=[json.loads(f.read_text()) for f in OUT.glob(f"{name}_*.json")]
        bottom=np.zeros(3)
        for oc in ("pass","stall","fall"):
            vals=np.array([sum(r["runs"][c]["outcome"]==oc for r in rows) for c in CONDS])
            ax.bar(CONDS,vals,bottom=bottom,color=COLORS[oc],label=oc,width=.62)
            for i,v in enumerate(vals):
                if v: ax.text(i,bottom[i]+v/2,str(v),ha="center",va="center",fontsize=8,
                              color="white",fontweight="bold")
            bottom+=vals
        ax.set_title(f"{name}  (n={len(rows)} paired conditions)",fontsize=9)
        ax.set_ylabel("conditions",fontsize=8); ax.tick_params(labelsize=7)
        ax.legend(fontsize=7,loc="lower center",bbox_to_anchor=(0.5,-0.30),ncol=3,frameon=False)
        ax.set_ylim(0,len(rows)*1.10)
        for c in CONDS:
            r=[x["runs"][c]["progress_ratio_vs_A"] for x in rows if x["runs"][c]["progress_ratio_vs_A"]]
            ax.text(CONDS.index(c),len(rows)*1.03,f"×{np.mean(r):.2f}",ha="center",fontsize=7,color="0.35")
    fig.tight_layout(); p=OUT/"paired_outcomes.png"; fig.savefig(p,dpi=140); print(p)

main()
