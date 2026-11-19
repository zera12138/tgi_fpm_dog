"""Table + plot for the D-group single-factor sweep."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent; OUT=HERE/"perception_out/d_sweep"
ORDER=["latency","dx","dz","pitch","drop"]
LABEL={"latency":"publish latency (ms)","dx":"horizontal offset (cm)",
       "dz":"height offset (cm)","pitch":"extrinsic pitch error (deg)","drop":"observation gap (s)"}
UNIT={"latency":1000.0,"dx":100.0,"dz":100.0,"pitch":1.0,"drop":1.0}

def load():
    d={}
    for f in OUT.glob("*.json"):
        j=json.loads(f.read_text())
        d.setdefault(j["scenario"],{}).setdefault(j["factor"],[]).append(j)
    for sc in d:
        for fa in d[sc]:
            d[sc][fa].sort(key=lambda r:r["value"])
    return d

def main():
    data=load()
    for sc in sorted(data):
        print(f"\n=== {sc} ===")
        print(f"{'factor':<9}{'level':>9}{'pass':>6}{'stall':>7}{'fall':>6}{'ratio':>8}{'cov':>8}{'err_all':>9}")
        for fa in ORDER:
            for r in data[sc].get(fa,[]):
                print(f"{fa:<9}{r['value']*UNIT[fa]:>9.2f}{r['pass']:>6}{r['stall']:>7}{r['fall']:>6}"
                      f"{r['mean_progress_ratio']:>8.2f}{r['mean_coverage']*100:>7.1f}%{r['mean_err_all_mm']:>8.1f}mm")
    scs=sorted(data)
    fig,axes=plt.subplots(len(scs),len(ORDER),figsize=(3.0*len(ORDER),2.6*len(scs)),squeeze=False)
    for r_i,sc in enumerate(scs):
        for c_i,fa in enumerate(ORDER):
            ax=axes[r_i][c_i]; rows=data[sc].get(fa,[])
            if not rows: continue
            x=[r["value"]*UNIT[fa] for r in rows]
            ax.plot(x,[r["mean_progress_ratio"] for r in rows],"o-",color="#1f77b4",ms=3,lw=1.3)
            ax.axhline(1.0,color="0.6",ls="--",lw=.8)
            ax.axhline(0.85,color="#e0a800",ls=":",lw=.9)
            for xi,r in zip(x,rows):
                if r["fall"]: ax.plot(xi,0.02,"v",color="#d62728",ms=7)
                elif r["stall"]: ax.plot(xi,0.06,"s",color="#e0a800",ms=6)
            ax.set_title(f"{sc}: {LABEL[fa]}",fontsize=8)
            ax.tick_params(labelsize=6.5); ax.set_ylim(-0.02,1.15); ax.grid(alpha=.25,lw=.4)
            if c_i==0: ax.set_ylabel("progress / truth-map A",fontsize=7)
    fig.tight_layout(); p=OUT/"d_sweep.png"; fig.savefig(p,dpi=140); print("\n"+str(p))

main()
