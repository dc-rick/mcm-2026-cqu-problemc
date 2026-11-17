"""Cell-level stochastic simulator with a walking radius.

A user departs from a 0.005-deg cell.  If the cell is empty the user walks to the
nearest neighbouring cell that holds a bike (<= WALK_KM); only if no bike exists
inside that radius is the trip lost.  This is the metric that matches what a
dockless user actually experiences.

T004 fix: a bicycle taken at a neighbouring cell is a real ride.  It now enters
the in-flight arrival pool exactly like a bicycle taken at the departure cell,
so walking conserves the fleet.  Before this fix the borrowed bicycle was
removed from the donor and never returned, which silently deleted 240,170
bicycles per day at R=450 m.  At R=300 m no neighbour is reachable, so the
published 13.4% baseline is bit-identical to the previous behaviour.
"""
import numpy as np
from scipy.spatial import cKDTree


class CellSim:
    def __init__(self, dep5, Pest, I0, xy, zid, rng=None, walk_km=0.30):
        self.dep5 = dep5
        self.P = Pest
        self.I0 = I0.astype(float)
        self.zid = zid
        self.C = dep5.shape[1]
        self.Z = Pest.shape[0]
        self.rng = rng or np.random.default_rng(0)
        self.T = dep5.shape[0]
        tree = cKDTree(xy)
        self.nbrs = tree.query_ball_point(xy, walk_km)
        self.zcells = [np.where(zid == z)[0] for z in range(self.Z)]

    def run(self, move5=None, record=False):
        C, T = self.C, self.T
        I = self.I0.copy()
        inflight = np.zeros((2, C))
        unmet = 0.0; served = 0.0
        traj = np.zeros((T, C)) if record else None
        for k in range(T):
            d = self.rng.poisson(self.dep5[k])
            take = np.minimum(d, np.floor(I).astype(int))
            I -= take
            resid = d - take
            served += take.sum()
            borrow = np.zeros(C, dtype=int)
            # walk to a neighbouring cell for the residual demand
            for c in np.where(resid > 0)[0]:
                need = resid[c]
                while need > 0:
                    nb = [j for j in self.nbrs[c] if I[j] >= 1]
                    if not nb:
                        break
                    j = max(nb, key=lambda q: I[q])
                    I[j] -= 1
                    borrow[c] += 1
                    need -= 1
                    served += 1
                unmet += need

            for z in range(self.Z):
                idx = self.zcells[z]
                n = int(take[idx].sum() + borrow[idx].sum())
                if n > 0:
                    dst = self.rng.choice(C, size=n, p=self.P[z])
                    np.add.at(inflight[1], dst, 1)
            I += inflight[0]
            inflight[0] = inflight[1]
            inflight[1] = 0.0

            if move5 is not None:
                mv = move5[k]
                for z in range(self.Z):
                    if mv[z] > 0:
                        I[self.zcells[z]] += mv[z] / max(len(self.zcells[z]), 1)
                    elif mv[z] < 0:
                        idx = self.zcells[z]
                        tot = I[idx].sum()
                        if tot > 0:
                            I[idx] *= max(0.0, 1 + mv[z] / tot)
            if record:
                traj[k] = I
        return dict(unmet=float(unmet), served=float(served),
                    unmet_share=float(unmet / max(unmet + served, 1)), traj=traj)
