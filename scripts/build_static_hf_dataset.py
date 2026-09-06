#!/usr/bin/env python3
"""Build a fixed-geometry DeepMD dataset from Hf hcp/bcc PBE references."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
from ase.io import read

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    root=Path("/GenSIvePFS/users/zirenj/fes_experiment_pbe_d3bj")
    ref=root/"reference/hf_hcp_bcc_pbe/free_energy_common.csv"
    reps=root/"reference/hf_hcp_bcc_pbe/static_representatives"
    out=root/"fes_static_only_hf_hcp_bcc_pbe"
    rows=list(csv.DictReader(ref.open()))
    t=np.array([float(r["T_K"]) for r in rows])
    g={"Hf_hcp":np.array([float(r["G_hcp_meV_per_atom"])/1000 for r in rows]),
       "Hf_bcc":np.array([float(r["G_bcc_meV_per_atom"])/1000 for r in rows])}
    out.mkdir(parents=True,exist_ok=True)
    (out/"type.raw").write_text("0\n")
    (out/"type_map.raw").write_text("Hf\n")
    meta={"protocol":"static-only","state_input":"fparam=[temperature_K, pressure_GPa]",
          "functional":"GGA-PBE","pressure_GPa":0.0001,"temperature_count":len(t),
          "temperature_range_K":[float(t[0]),float(t[-1])],"phases":{}}
    for phase,filename in [("Hf_hcp","Hf_hcp_static.vasp"),("Hf_bcc","Hf_bcc_static.vasp")]:
        a=read(str(reps/filename))
        coord=np.asarray(a.positions,dtype=np.float64)
        box=np.asarray(a.cell.array,dtype=np.float64).reshape(9)
        atype=np.zeros(len(a),dtype=np.int32)
        pdir=out/phase; pdir.mkdir(parents=True,exist_ok=True)
        size=256
        for start in range(0,len(t),size):
            stop=min(start+size,len(t)); sd=pdir/f"set.{start//size:03d}"; sd.mkdir()
            n=stop-start
            np.save(sd/"coord.npy",np.repeat(coord.reshape(1,-1),n,axis=0))
            np.save(sd/"box.npy",np.repeat(box.reshape(1,-1),n,axis=0))
            np.save(sd/"fparam.npy",np.column_stack((t[start:stop],np.full(n,0.0001))))
            np.save(sd/"free_energy.npy",(g[phase][start:stop]*len(a))[:,None])
        np.savetxt(pdir/"type.raw",atype,fmt="%d")
        meta["phases"][phase]={"source":str(reps/filename),"n_atoms":len(a),
          "label_unit":"eV_per_structure","coord_sha256":digest(coord),
          "box_sha256":digest(box),"atype_sha256":digest(atype),
          "coord_constant_across_temperature":True,"box_constant_across_temperature":True,
          "pressure_constant_across_temperature":True}
    (out/"static_protocol.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))
if __name__=="__main__":
    main()
