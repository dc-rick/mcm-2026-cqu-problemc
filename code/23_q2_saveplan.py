import numpy as np, json
import importlib.util, sys
sys.path.insert(0, 'code')
spec = importlib.util.spec_from_file_location("q2", "code/21_q2_opt.py")
q2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(q2)
r = q2.solve(alpha=1.0, K=60, time_limit=300, fixed_K_cost=600.0, verbose=True)
np.savez(q2.DATA + r"\q2_plan.npz", XM=r["XM"], X=r["X"], M=r["M"], I=r["I"], U=r["U"])
print("saved plan", r["XM"].shape, "bikes moved", r["XM"].sum().round(0))
