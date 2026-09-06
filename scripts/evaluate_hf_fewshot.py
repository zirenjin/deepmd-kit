#!/usr/bin/env python3
"""Evaluate locked static-only Hf hcp/bcc few-shot FES runs."""
import argparse, csv, glob, json, subprocess
from pathlib import Path
import numpy as np
LABELS = {1:[0], 2:[0,200], 4:[0,100,200,300], 8:[0,50,100,150,200,250,300,350]}
PHASES = {"Hf_hcp":36, "Hf_bcc":54}
def crossings(x,y,atol=1e-10):
    x=np.asarray(x,float); y=np.asarray(y,float); out=[]; i=0
    while i<len(y):
        if abs(y[i])<=atol:
            j=i
            while j+1<len(y) and abs(y[j+1])<=atol: j+=1
            out.append(float((x[i]+x[j])/2)); i=j+1; continue
        if i+1<len(y) and y[i]*y[i+1]<0:
            out.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
        i+=1
    return out
def read_property(path,natoms):
    lines=[s for s in Path(path).read_text().splitlines() if s.strip() and not s.startswith("#")]
    return float(lines[-1].split()[1])/natoms
def test_phase(model,phase,data,out):
    d=out/"test"/phase; d.mkdir(parents=True,exist_ok=True)
    subprocess.run(["dp","--pt","test","-m",str(model),"-s",str(data/phase),"-d",str(d)],check=True,stdout=(out/f"test_{phase}.log").open("w"),stderr=subprocess.STDOUT)
    files=sorted(glob.glob(str(d/f"{phase}.property.out.*")),key=lambda p:int(p.rsplit(".",1)[1]))
    if not files: raise RuntimeError(f"no outputs for {phase}: {d}")
    return np.array([read_property(p,PHASES[phase]) for p in files])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--data-root",type=Path,required=True); ap.add_argument("--reference",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--seeds",nargs="+",type=int,default=[11,23,37,51,67]); a=ap.parse_args()
    rows=list(csv.DictReader(a.reference.open())); T=np.array([float(r["T_K"]) for r in rows]); rh=np.array([float(r["G_hcp_meV_per_atom"]) for r in rows])/1000; rb=np.array([float(r["G_bcc_meV_per_atom"]) for r in rows])/1000; rd=rb-rh; rc=crossings(T,rd,1e-12); results={}
    for n,idx in LABELS.items():
        results[str(n)]={}
        for seed in a.seeds:
            run=a.root/f"hf_fes_fewshot{n}_seed{seed}"; ck=run/"model.ckpt-3000.pt"
            if not ck.exists(): results[str(n)][str(seed)]={"status":"missing_checkpoint"}; continue
            ev=run/"eval"; model=ev/"model.pth"
            if not model.exists():
                ev.mkdir(parents=True,exist_ok=True); subprocess.run(["dp","--pt","freeze","-c",str(ck),"-o",str(model)],check=True,stdout=(ev/"freeze.log").open("w"),stderr=subprocess.STDOUT)
            ph=test_phase(model,"Hf_hcp",a.data_root,ev); pb=test_phase(model,"Hf_bcc",a.data_root,ev); pd=pb-ph; mask=np.ones(T.size,dtype=bool); mask[idx]=False; err=pd[mask]-rd[mask]; rr=np.argmin(np.stack([rh,rb]),axis=0); pr=np.argmin(np.stack([ph,pb]),axis=0); pc=crossings(T,pd,1e-10)
            results[str(n)][str(seed)]={"status":"ok","label_indices":idx,"label_temperatures_K":T[idx].tolist(),"heldout_n":int(mask.sum()),"heldout_mae_eV_per_atom":float(np.mean(np.abs(err))),"heldout_rmse_eV_per_atom":float(np.sqrt(np.mean(err**2))),"heldout_sign_accuracy":float(np.mean(np.sign(pd[mask])==np.sign(rd[mask]))),"phase_ranking_accuracy":float(np.mean(pr[mask]==rr[mask])),"reference_crossings_K":rc,"predicted_crossings_K":pc,"crossing_error_K":(abs(pc[0]-rc[0]) if pc and rc else None)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(results,indent=2)+"\n"); print(json.dumps(results,indent=2))
if __name__=="__main__": main()
