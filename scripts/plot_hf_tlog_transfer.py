#!/usr/bin/env python3
import argparse, csv, glob, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
PHASES={"Hf_hcp":36,"Hf_bcc":54}
def read_prop(p,n):
 lines=[x for x in Path(p).read_text().splitlines() if x.strip() and not x.startswith("#")]
 return float(lines[-1].split()[1])/n
def vals(run,phase):
 direct=glob.glob(str(run/"eval"/f"{phase}.property.out.*"))
 files=direct or glob.glob(str(run/"eval"/"**"/f"{phase}.property.out.*"),recursive=True)
 return np.array([read_prop(p,PHASES[phase]) for p in sorted(files,key=lambda p:int(p.rsplit(".",1)[1]))])
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--reference",type=Path,required=True);ap.add_argument("--results",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 rows=list(csv.DictReader(a.reference.open())); T=np.array([float(x["T_K"]) for x in rows]); ref=np.array([float(x["delta_G_bcc_minus_hcp_meV_per_atom"])/1000 for x in rows])
 data=json.load(a.results.open()); seeds=[11,23,37,51,67]; curves={}
 for n in sorted(data,key=int):
  arr=[]
  for seed in seeds:
   run=a.root/f"hf_tlog_fes_fewshot{n}_seed{seed}"
   if (run/"eval"/"model.pth").exists(): arr.append(vals(run,"Hf_bcc")-vals(run,"Hf_hcp"))
  if arr: curves[n]=np.stack(arr)
 a.output.mkdir(parents=True,exist_ok=True)
 fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True); axs=axs.ravel()
 for ax,(n,arr) in zip(axs,curves.items()):
  ax.plot(T,ref,color="black",lw=1.8,label="reference")
  ax.plot(T,arr.mean(0),color="#1769aa",label="T-log mean")
  ax.fill_between(T,arr.min(0),arr.max(0),color="#1769aa",alpha=.18,label="seed range")
  ax.axhline(0,color="0.7",lw=.7); ax.set_title(f"{n} labels"); ax.grid(alpha=.2)
  ax.set_ylabel("Delta G bcc-hcp (eV/atom)")
 axs[-1].set_xlabel("Temperature (K)"); axs[-2].set_xlabel("Temperature (K)")
 axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(a.output/"hf_tlog_deltaG_curves.png",dpi=180); plt.close(fig)
 labels=[]; mae=[]; rmse=[]; tc=[]
 for n in sorted(data,key=int):
  ok=[v for v in data[n].values() if v.get("status")=="ok"]; labels.append(int(n)); mae.append(np.mean([v["heldout_mae_eV_per_atom"] for v in ok])); rmse.append(np.mean([v["heldout_rmse_eV_per_atom"] for v in ok])); tc.append(np.mean([v["crossing_error_K"] for v in ok if v["crossing_error_K"] is not None]))
 fig,ax=plt.subplots(figsize=(7,5)); ax.plot(labels,mae,"o-",label="Delta-G MAE"); ax.plot(labels,rmse,"s-",label="Delta-G RMSE"); ax.set_xlabel("Number of labels"); ax.set_ylabel("Error (eV/atom)"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(a.output/"hf_tlog_error_vs_labels.png",dpi=180); plt.close(fig)
 (a.output/"plot_manifest.json").write_text(json.dumps({"curves":str(a.output/"hf_tlog_deltaG_curves.png"),"errors":str(a.output/"hf_tlog_error_vs_labels.png"),"reference":str(a.reference),"raw_results":str(a.results)},indent=2)+"\n")
if __name__=="__main__": main()
