#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
LABELS={1:[0],2:[0,200],4:[0,100,200,300],8:[0,50,100,150,200,250,300,350]}
def cross(x,y):
 out=[];i=0
 while i<len(y):
  if abs(y[i])<1e-12:
   j=i
   while j+1<len(y) and abs(y[j+1])<1e-12:j+=1
   out.append(float((x[i]+x[j])/2));i=j+1;continue
  if i+1<len(y) and y[i]*y[i+1]<0:out.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
  i+=1
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--reference",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 rows=list(csv.DictReader(a.reference.open()));T=np.array([float(r["T_K"]) for r in rows]);tau=T/1000
 gh=np.array([float(r["G_hcp_meV_per_atom"])/1000 for r in rows]);gb=np.array([float(r["G_bcc_meV_per_atom"])/1000 for r in rows]);ref=gb-gh;rc=cross(T,ref)
 out={}
 for n,idx in LABELS.items():
  block={}
  for seed in [11,23,37,51,67]:
   torch.manual_seed(seed);np.random.seed(seed)
   ti=np.array(idx); x=[];y=[]
   for j in ti:
    for phase,g in [(0,gh[j]),(1,gb[j])]:
     x.append([tau[j],float(phase)]);y.append(g)
   x=torch.tensor(x,dtype=torch.float32);y=torch.tensor(y,dtype=torch.float32).reshape(-1,1)
   net=torch.nn.Sequential(torch.nn.Linear(2,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
   opt=torch.optim.Adam(net.parameters(),lr=2e-3)
   for _ in range(5000):
    opt.zero_grad();loss=((net(x)-y)**2).mean();loss.backward();opt.step()
   xt=[]; 
   for j in range(len(T)):
    xt.extend([[tau[j],0.],[tau[j],1.]])
   pred=net(torch.tensor(xt,dtype=torch.float32)).detach().numpy().reshape(-1,2);delta=pred[:,1]-pred[:,0];m=np.ones(len(T),bool);m[ti]=False;e=delta[m]-ref[m];pc=cross(T,delta)
   block[str(seed)]={"status":"ok","heldout_mae_eV_per_atom":float(np.mean(abs(e))),"heldout_rmse_eV_per_atom":float(np.sqrt(np.mean(e*e))),"heldout_sign_accuracy":float(np.mean(np.sign(delta[m])==np.sign(ref[m]))),"phase_ranking_accuracy":float(np.mean(np.sign(delta[m])==np.sign(ref[m]))),"reference_crossings_K":rc,"predicted_crossings_K":pc,"crossing_error_K":(abs(pc[0]-rc[0]) if pc and rc else None)}
  out[str(n)]=block
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()
