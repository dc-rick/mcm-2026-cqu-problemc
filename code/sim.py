"""Shared stochastic simulator for a typical day at the zone level.

Runs at 5-min resolution.  Departures are Poisson, destinations multinomial from
the calibrated hour-of-origin OD matrix, bikes are unavailable for one step while
ridden.  Trucks execute an hourly relocation plan.
"""
import numpy as np


class Sim:
    def __init__(self, dep_hr, od_hr, I0, scale=1.0, seed=0, step_min=5,
                 ride_steps=1):
        """dep_hr : (Z,24) hourly departures (already at full scale)
           od_hr  : (24,Z,Z) hourly destination counts, row-normalised here
           I0     : (Z,) initial inventory (bikes)"""
        self.Z, self.T = dep_hr.shape
        self.dep_hr = dep_hr
        self.I0 = I0.astype(float)
        self.rng = np.random.default_rng(seed)
        self.sub = 60 // step_min                 # steps per hour
        self.nsteps = self.T * self.sub
        self.ride_steps = ride_steps
        P = od_hr.copy()
        s = P.sum(2, keepdims=True)
        self.P = np.divide(P, s, out=np.full_like(P, 1.0 / self.Z), where=s > 0)

    def run(self, move_plan=None, incentive=None, deadline_zone=None, record=False):
        """move_plan : (T,Z,Z) bikes moved from z to z' during hour t
           incentive  : None or dict with keys
                        'accept' (H,Z,Z) probability a trip z->j ending in j is
                        redirected to a nearby deficit zone, 'target' (H,Z,Z)
                        redistribution matrix;  see q3 module.
        """
        Z, sub = self.Z, self.sub
        I = self.I0.copy()
        inflight = np.zeros((self.ride_steps + 1, Z))
        served = np.zeros(Z); unmet = np.zeros(Z)
        I_traj = np.zeros((self.nsteps, Z)) if record else None
        dest_extra = np.zeros((self.T, Z, Z)) if record else None

        for k in range(self.nsteps):
            t = k // sub
            lam = self.dep_hr[:, t] / sub
            d = self.rng.poisson(lam)
            take = np.minimum(d, np.floor(I).astype(int))
            I -= take
            unmet += d - take
            served += take

            # destinations
            for z in range(Z):
                n = int(take[z])
                if n > 0:
                    dests = self.rng.choice(Z, size=n, p=self.P[t, z])
                    np.add.at(inflight[self.ride_steps], dests, 1)

            I += inflight[0]
            inflight = np.roll(inflight, -1, axis=0)
            inflight[-1] = 0

            if move_plan is not None:
                mv = move_plan[t] / sub
                out = mv.sum(1)
                take_m = np.minimum(out, I)
                # scale rows so that we never remove more than available
                scale = np.divide(take_m, out, out=np.zeros_like(out), where=out > 0)
                I -= take_m
                I += (mv * scale[:, None]).sum(0)

            if record:
                I_traj[k] = I
        return dict(unmet=float(unmet.sum()), served=float(served.sum()),
                    unmet_share=float(unmet.sum() / max(served.sum() + unmet.sum(), 1)),
                    I_end=I, I_traj=I_traj, unmet_zone=unmet)
