#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import h5py,numpy as np,torch
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
 ap=argparse.ArgumentParser();ap.add_argument("--embedding",type=Path,required=True);ap.add_argument("--reference",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 with h5py.File(a.embedding) as f:
  z=[]
  for name in ["Hf_hcp","Hf_bcc"]:
   d=f[name]["descriptor"][:].astype("float32"); z.append(np.concatenate([d.mean(1),d.std(1),d.max(1)],axis=1))
 z=np.stack([z[0][0],z[1][0]]); mu=z.mean(0); sd=z.std(0); sd[sd<1e-6]=1; z=(z-mu)/sd
 rows=list(csv.DictReader(a.reference.open()));T=np.array([float(r["T_K"]) for r in rows]);tau=T/1000;gh=np.array([float(r["G_hcp_meV_per_atom"])/1000 for r in rows]);gb=np.array([float(r["G_bcc_meV_per_atom"])/1000 for r in rows]);Y=np.stack([gh,gb]);ref=gb-gh;rc=cross(T,ref);out={}
 for n,idx in LABELS.items():
  block={}
  for seed in [11,23,37,51,67]:
   torch.manual_seed(seed);np.random.seed(seed); xs=[];ys=[]
   for j in idx:
    for p in [0,1]: xs.append(np.r_[z[p],tau[j],float(p)]);ys.append(Y[p,j])
   x=torch.tensor(np.asarray(xs),dtype=torch.float32);y=torch.tensor(ys,dtype=torch.float32).reshape(-1,1)
   net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,1));opt=torch.optim.Adam(net.parameters(),lr=2e-3)
   for _ in range(5000):
    opt.zero_grad();loss=((net(x)-y)**2).mean();loss.backward();opt.step()
   xt=[]
   for j in range(len(T)):
    for p in [0,1]:xt.append(np.r_[z[p],tau[j],float(p)])
   pred=net(torch.tensor(np.asarray(xt),dtype=torch.float32)).detach().numpy().reshape(-1,2);delta=pred[:,1]-pred[:,0];m=np.ones(len(T),bool);m[idx]=False;e=delta[m]-ref[m];pc=cross(T,delta)
   block[str(seed)]={"status":"ok","heldout_mae_eV_per_atom":float(np.mean(abs(e))),"heldout_rmse_eV_per_atom":float(np.sqrt(np.mean(e*e))),"heldout_sign_accuracy":float(np.mean(np.sign(delta[m])==np.sign(ref[m]))),"phase_ranking_accuracy":float(np.mean(np.sign(delta[m])==np.sign(ref[m]))),"reference_crossings_K":rc,"predicted_crossings_K":pc,"crossing_error_K":(abs(pc[0]-rc[0]) if pc and rc else None)}
  out[str(n)]=block
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()
