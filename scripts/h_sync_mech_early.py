"""EXPLORATORY (not pre-registered): early-time sigma and rate curves, sync vs fixref vs balance, A seed 0, 4 s. Output: docs/receipts/h_sync_mech/exploratory_early.txt"""
import sys; sys.path.insert(0,"/home/user/jomission/scripts")
import numpy as np, h_active_test as T, h_balance_test as HB
import jomission.dynamics.activity_balance as ab, jomission.dynamics.activity_balance_sync as sy, jomission.dynamics.activity_balance_sync_fixref as fx
DUR=4000.0
m=T.build(T.BASE_TONIC)
def hd(rule,**rp): return {"hdp_rule":rule,"hdp_rule_params":rp,"record_weight_trace":False}
def curve(sp,k=2500): # 250 ms bins
    s=sp.reshape(-1,k,sp.shape[1]).sum(axis=(1,2))/(sp.shape[1]*0.25); return np.round(s,2)
out={}
out["balance"]=HB.run(m,0,hd(ab.RULE_NAME),DUR)[0]
sp,d=HB.run(m,0,hd(sy.RULE_NAME),DUR); out["sync_s1"]=sp
aux=np.asarray(d["aux_trace"],dtype=np.float64)   # (T,N,2)
Ps=aux[:,:,0].mean(1); Pf=aux[:,:,1].mean(1)
sig=np.maximum(0,Pf-1.5*Ps)/np.maximum(Ps,1e-3)
for t in (0,10,100,500,1000,2000,3999):
    i=min(int(t*10),len(sig)-1)
    print("t %5d ms  P_s %.3f  P_f %.2f  sigma %.2f"%(t,Ps[i],Pf[i],sig[i]))
print("sigma max %.1f at t=%.0f ms; mean sigma 0-500ms %.2f, 2000-4000ms %.3f"%(sig.max(),sig.argmax()/10,sig[:5000].mean(),sig[20000:].mean()))
out["fixref"]=HB.run(m,0,hd(fx.RULE_NAME,p_ref=9.43),DUR)[0]
for k,v in out.items(): print(k,"rate/250ms:",curve(v))
