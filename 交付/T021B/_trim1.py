# T021B trim pass 1: tighten redundant prose only. No numeric token may change.
import io, re, sys
TOK = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?")

EDITS = [
("sec5_q2.tex", r"""never reaches zero from the initial-inventory proxy of 343,638 bicycles; $a$ and
$d$ are the hourly arrivals and departures of a typical day scaled by $1/p$. This is a
statement about the initial-inventory proxy and the zone aggregation, not a
resolution-free guarantee. At cell resolution the pre-plan simulation leaves unserved
shares of 5.84\% at 343,638 bicycles and 3.51\% at the closed fleet 495,540 (R=450 m).""",
r"""never reaches zero from the initial-inventory proxy of 343,638 bicycles; $a$ and
$d$ are the hourly arrivals and departures of a typical day scaled by $1/p$. This is a
statement about the proxy and the zone aggregation, not a resolution-free guarantee:
at cell resolution the pre-plan simulation leaves unserved shares of 5.84\% at 343,638
and 3.51\% at the closed fleet 495,540 (R=450 m)."""),

("sec5_q2.tex", r"""The objective has five parts. $c_{\text{km}}=6$ CNY per truck-kilometre is the running
cost; $c_{\text{load}}=0.5$ CNY per bicycle loaded and unloaded is the handling cost;
$w=6$ CNY per unmet departure is a service penalty set above the cost of moving a
bicycle; $c_{\text{dev}}=3$ CNY per bicycle of end-of-day drift penalises leaving the
system out of balance, linearised by $dp,dm$; and $f=600$ CNY per truck per day is the
ownership cost of a truck, charged on the fleet size $K$ because a truck that is bought
must be paid for whether it drives or not.""",
r"""The objective prices five things. $c_{\text{km}}=6$ CNY per truck-kilometre is the
running cost; $c_{\text{load}}=0.5$ CNY per bicycle loaded and unloaded is handling;
$w=6$ CNY per unmet departure is a service penalty set above the cost of moving a
bicycle; $c_{\text{dev}}=3$ CNY per bicycle of end-of-day drift, linearised by $dp,dm$,
penalises leaving the system out of balance; and $f=600$ CNY per truck per day is the
ownership cost of a truck, charged on $K$ because a truck that is bought must be paid
for whether it drives or not."""),

("sec5_q2.tex", r"""The constraints are the physics. \eqnref{eq:q2bal} is conservation of bicycles.
\eqnref{eq:q2serv} is the event-order constraint: a truck may take away only bicycles
that are present at the start of the hour, before that hour's departures are served.
\eqnref{eq:q2cap} couples bikes to trucks ($Q=40$ bicycles per truck), and
\eqnref{eq:q2truck}--\eqnref{eq:q2fleet} conserve trucks and return the fleet to its
starting zones by midnight. \eqnref{eq:q2dev} defines the end-of-day deviation, which
makes the plan a reset rather than a one-way transfer. Bicycles, trucks and truck
presence are integer variables; solved as a continuous program the same instance would
be cheaper but would describe fractions of a truck and fractions of a bicycle.""",
r"""The constraints are the physics. \eqnref{eq:q2bal} is conservation of bicycles;
\eqnref{eq:q2serv} is the event-order constraint: a truck may take away only bicycles
present at the start of the hour, before that hour's departures are served;
\eqnref{eq:q2cap} couples bikes to trucks ($Q=40$ bicycles per truck);
\eqnref{eq:q2truck}--\eqnref{eq:q2fleet} conserve trucks and return them to their
starting zones by midnight; \eqnref{eq:q2dev} defines the end-of-day deviation that
makes the plan a reset. Bicycles, trucks and truck presence are integer variables; as
a continuous program the instance would be cheaper but would describe fractions of a
truck and of a bicycle."""),

("sec5_q2.tex", r"""Solved as an LP over tricycle counts, the layer buys 889 tricycles, costs 507,779.67
and leaves 40,041.58 cell-level departures unserved, for a total of 591,587.42 and a
saving of 927,691.53 (61.06\%) against the no-tricycle cell-level baseline of
1,519,278.95. That is an \emph{ideal LP upper bound}, not a plan: the integer schedule
built from the same allocation leaves 534 of 13,775 trips unassigned in 15 zones,
because a fractional tricycle cannot be dispatched. Raising the fleet to the smallest
executable fixed point adds 86 tricycles across those 15 zones, for \textbf{975
tricycles} and zero unassigned trips in all 50 zones.""",
r"""Solved as an LP over tricycle counts, the layer buys 889 tricycles, costs 507,779.67
and leaves 40,041.58 cell-level departures unserved, for a total of 591,587.42 and a
saving of 927,691.53 (61.06\%) against the no-tricycle cell-level baseline of
1,519,278.95. That is an \emph{ideal LP upper bound}, not a plan: the integer schedule
built from the same allocation leaves 534 of 13,775 trips unassigned in 15 zones.
Raising the fleet to the smallest executable fixed point adds 86 tricycles across
those 15 zones, for \textbf{975 tricycles} and zero unassigned trips in all 50 zones."""),

("sec5_q2.tex", r"""leaves 2,059.49 and the integer schedule 2,085.98; the 26.50 loss is 0.43\% of the
bicycle flows the LP assigns in the zone and is bounded by 81.5. The bound is not tight, but it is small relative
to the layer cost, and the direction is known: the executable plan can only be worse
than its LP relaxation.""",
r"""leaves 2,059.49 and the integer schedule 2,085.98; the 26.50 loss is 0.43\% of the
bicycle flows the LP assigns in the zone and is bounded by 81.5. The bound is not
tight, but it is small; the executable plan can only be worse than its relaxation."""),

("sec5_q2.tex", r"""\tabref{tab:q2schemes} puts the schemes side by side. The zone-level panel and the
cell-level panel are \emph{different accounting calibers}, and the table says so: the
truck plan was optimised in the zone model, where it removes all 432.23 unmet
departures at $K=0$ and costs 83,807.74 at $K=60$. The same truck plan, priced in the
cell model, carries a cell deficit of 239,245.20 that the zone model cannot see; the
two-tier plan is the only row that removes most of it.""",
r"""\tabref{tab:q2schemes} puts the schemes side by side. The two panels are
\emph{different accounting calibers}: the truck plan was optimised in the zone model,
where it removes all 432.23 unmet departures at $K=0$ and costs 83,807.74 at $K=60$;
the same plan, priced in the cell model, carries a cell deficit of 239,245.20 that the
zone model cannot see, and the two-tier plan is the only row that removes most of it."""),

("sec5_q2.tex", r"""against 617,055.75, a saving of 902,223.20 (59.385\%). The two savings answer different
questions. 41.55\% is what trucks do for the zone model; 59.385\% is what the tricycle
layer adds once the zone model's blind spot is priced. Neither is a system-level claim
that one instrument alone would deliver the other's number.""",
r"""against 617,055.75, a saving of 902,223.20 (59.385\%). The two savings answer different
questions: 41.55\% is what trucks do for the zone model, 59.385\% what the tricycle
layer adds once the zone model's blind spot is priced, and neither is a system-level
claim."""),

("sec6_q3.tex", r"""A truck can carry a bicycle ten kilometres between zones. A rider can be persuaded to
end a trip a few hundred metres from where she intended, and no more: the reward would
have to be absurd to make someone ride across the city. The two instruments therefore
operate at different scales, and the right question is not which one replaces the other
but how much of each layer the incentive can absorb. The answer turns out to be
asymmetric, and we report it as two paragraphs rather than one headline.""",
r"""A truck can carry a bicycle ten kilometres between zones; a rider can be persuaded to
end a trip a few hundred metres from where she intended, and no more. The two
instruments operate at different scales, so the right question is not which one
replaces the other but how much of each layer the incentive can absorb. The answer is
asymmetric, and we report it as two paragraphs rather than one headline."""),

("sec6_q3.tex", r"""The truck layer of Q2 is a single-layer plan. Its own residual is tiny: at $K=0$ the
zone model leaves 432.23 departures unserved, and at $K=2$ it leaves 33.19. There is no
large pool of truck-served unmet demand for an incentive to substitute for, so the only
question is whether the incentive can pay for itself by reducing the truck objective.
It cannot. Solving the integer truck program of Section~\ref{sec:q2} with and without the incentive at
$K=0,2,5,10,20$, re-solving both sides at a 0.1\% gap, gives a payment ceiling
$\text{pure}_K-\text{hybrid}_K$ that is negative at \emph{every} $K$: $-1{,}739.22$ at
$K=0$, $-1{,}591.40$ at $K=2$, $-1{,}588.76$ at $K=5$, $-1{,}623.33$ at $K=10$ and
$-1{,}528.82$ at $K=20$. A negative ceiling means no non-negative reward can break
even.""",
r"""The truck layer of Q2 is a single-layer plan whose own residual is tiny: at $K=0$ the
zone model leaves 432.23 departures unserved, and at $K=2$ it leaves 33.19. There is no
large pool of truck-served unmet demand for an incentive to substitute for, so the only
question is whether the incentive can pay for itself by reducing the truck objective.
It cannot. Solving the integer truck program of Section~\ref{sec:q2} with and without
the incentive at $K=0,2,5,10,20$, re-solving both sides at a 0.1\% gap, gives a
payment ceiling $\text{pure}_K-\text{hybrid}_K$ that is negative at \emph{every} $K$:
$-1{,}739.22$ at $K=0$, $-1{,}591.40$ at $K=2$, $-1{,}588.76$ at $K=5$, $-1{,}623.33$
at $K=10$ and $-1{,}528.82$ at $K=20$. A negative ceiling means no non-negative reward
can break even."""),

("sec6_q3.tex", r"""The tricycle layer is different because its residual is large: 239,245.20 cell-level
departures unserved before tricycles, and 39,984.2332 after the executable 975-tricycle
plan. A reroute changes the arrival count at its target cell, which reduces the deficit
the tricycle layer must fill, so the incentive enters exactly where the tricycle layer
is paying.""",
r"""The tricycle layer is different because its residual is large: 239,245.20 cell-level
departures unserved before tricycles and 39,984.2332 after the executable
975-tricycle plan. A reroute raises the arrival count at its target cell, reducing the
deficit the tricycle layer must fill, so the incentive enters exactly where the
tricycle layer is paying."""),

("sec7_q4.tex", r"""A dockless bicycle occupies space where its trip ends and stays there until someone
rides it away. The parking demand of a cell is therefore the arrival process of
Section~\ref{sec:q1}, but the occupancy it produces is the \emph{integral} of that
process on top of whatever is already standing there. The simulation below starts each
fence at its measured standing inventory and sizes the peak of the resulting path,
because the initial-inventory proxy contains 343,638 bicycles and Q2 shows that the plan is
a reset, not an evacuation.""",
r"""A dockless bicycle occupies space where its trip ends and stays until someone rides it
away. Parking demand is therefore the arrival process of Section~\ref{sec:q1}, but the
occupancy it produces is the \emph{integral} of that process on top of the standing
stock; the simulation starts each fence at its measured inventory and sizes the peak of
the path, because the initial-inventory proxy contains 343,638 bicycles and Q2 shows
that the plan is a reset, not an evacuation."""),

("sec7_q4.tex", r"""Two conventions bound the interpretation. Per-fence percentiles are not additive: the
sum of 95th percentiles is not the city-wide 95th percentile of the sum. At the
initial-inventory proxy 343,638 the
sum of per-cell 95th percentiles is 974,923.16 while the 95th percentile of the
simultaneous system total is 630,002.78, a ratio of 0.646; at 495,540 the corresponding
figures are 1,123,702.61 and 525,987.17, a ratio of 0.468. The slot totals below are
therefore a \emph{required-capacity sizing convention} --- size each fence so that it is
adequate 95\% of the time on its own --- not a simultaneous city-wide guarantee, and
they are not existing berth counts. The inventory level is a scenario throughout: the
initial-inventory proxy 343,638 and the closed fleet 495,540 are run separately and never mixed.""",
r"""Two conventions bound the interpretation. Per-fence percentiles are not additive: the
sum of 95th percentiles is not the city-wide 95th percentile of the sum. At the
initial-inventory proxy 343,638 the sum of per-cell 95th percentiles is 974,923.16
while the 95th percentile of the simultaneous system total is 630,002.78, a ratio of
0.646; at 495,540 the figures are 1,123,702.61 and 525,987.17, a ratio of 0.468. The
slot totals below are therefore a \emph{required-capacity sizing convention} --- size
each fence so that it is adequate 95\% of the time on its own --- not a simultaneous
city-wide guarantee and not existing berth counts. The inventory level is a scenario
throughout: the initial-inventory proxy 343,638 and the closed fleet 495,540 are run
separately and never mixed."""),

("sec7_q4.tex", r"""The uncapped reference itself is retained only for comparison. It is the Kneedle knee
at $\lambda=150$ on the uncapped frontier: 705 fences, sized at the per-site 95th
percentile with no hard cap, requiring 491,692 slots at 343,638 and 621,004 at 495,540.
At $\lambda=0$ the same frontier's knee is 501 fences (64.51\% coverage) and at
$\lambda=300$ it is the same 705 fences, so the uncapped knee is stable in $\lambda$;
what is not stable is the per-site size. The 705-fence plan is therefore a
\emph{capacity reference}, not a realisable layout, and it is never compared with the
capped plan on coverage without first imposing the cap.""",
r"""The uncapped reference is retained only for comparison. It is the Kneedle knee at
$\lambda=150$ on the uncapped frontier: 705 fences, sized at the per-site 95th
percentile with no hard cap, requiring 491,692 slots at 343,638 and 621,004 at 495,540.
At $\lambda=0$ the same frontier's knee is 501 fences (64.51\% coverage) and at
$\lambda=300$ it is the same 705 fences, so the knee is stable in $\lambda$; the
per-site size is not. The 705-fence plan is therefore a \emph{capacity reference}, not
a realisable layout, and it is never compared with the capped plan without first
imposing the cap."""),

("sec7_q4.tex", r"""Two features are exact, not approximate. R=150 m and R=300 m give the
\emph{same} plan, because on the 0.005$^\circ$ grid no neighbouring cell centre lies
within 300 m of another; the radius only admits neighbour cells at 450 m. And the
nominal ranking reverses under voluntary compliance: 450 m has the highest coverage
(59.53\% against 52.42\%) but the lowest zero-incentive compliance (0.0238 against
0.1454), so effective coverage falls from 7.62\% at 150 m to 1.42\% at 450 m
(arrival-weighted 8.29\% to 1.33\%). On the arrival ruler the reversal is sharper
still: 150 m and 300 m cover 57.00\% of arrivals against 55.80\% at 450 m, because the
450 m plans leave the peak-heavy cells' excess unassigned. A capacitated fence network
is a mandatory-parking instrument; on voluntary compliance it is weakest exactly where
its nominal coverage is highest.""",
r"""Two features are exact, not approximate: R=150 m and R=300 m give the \emph{same}
plan, because on the 0.005$^\circ$ grid no neighbouring cell centre lies within 300 m of
another, so the radius only admits neighbour cells at 450 m; and the nominal ranking
reverses under voluntary compliance, since 450 m has the highest coverage (59.53\%
against 52.42\%) but the lowest zero-incentive compliance (0.0238 against 0.1454), so
effective coverage falls from 7.62\% at 150 m to 1.42\% at 450 m (arrival-weighted
8.29\% to 1.33\%). On the arrival ruler the reversal is sharper still: 150 m and 300 m
cover 57.00\% of arrivals against 55.80\% at 450 m, because the 450 m plans leave the
peak-heavy cells' excess unassigned. A capacitated fence network is a
mandatory-parking instrument; on voluntary compliance it is weakest exactly where its
nominal coverage is highest."""),

("sec8_q5.tex", r"""leaves 11.66\% of within-fence departures unserved. At $m=0$, with both platforms
proportional, the independent regime leaves 12.10\% and the shared regime 12.12\%, about
half a percentage point worse. The mechanism is not spatial: each platform's own Poisson
fluctuations no longer pool with the other's, so the same total fleet produces more
empty-fence steps.""",
r"""leaves 11.66\% of within-fence departures unserved; at $m=0$, with both platforms
proportional, the independent regime leaves 12.10\% and the shared regime 12.12\%,
about half a percentage point worse. The mechanism is not spatial: each platform's
Poisson fluctuations no longer pool with the other's, so the same total fleet produces
more empty-fence steps."""),

("sec8_q5.tex", r"""own fence sits half-empty while the other platform's riders walk past it. The
independent\_equal regime, with the same total concrete split in two, fills up to 6.23\%
of site-steps, and the shared regime sits between them at 0.35\% to 1.57\%. A low
full-site rate is therefore not evidence of adequate capacity; it is evidence that the
capacity is in the wrong hands.""",
r"""own fence sits half-empty while the other's riders walk past it; the
independent\_equal regime fills up to 6.23\% of site-steps, and the shared regime sits
between them at 0.35\% to 1.57\%. A low full-site rate is therefore not evidence of
adequate capacity but evidence that the capacity is in the wrong hands."""),

("sec9_sensitivity.tex", r"""Two cautions govern the reading. First, the reference column is the \emph{planning
midpoint}, not an expectation and not a median: the cost parameters enter
multiplicatively, so the midpoint of the parameter box is not the centre of the cost
distribution, and the Q2 midpoint of 617,056 sits well above the joint median of
484,372. Second, every Q2 and Q3 interval is an \emph{LP-proxy} interval. At 11
representative parameter points the integer chain deviates from the proxy by at most
38,986.30 CNY/day, about 0.2 standard deviations of the Q2 total; the intervals
therefore describe the shape of the parameter response, not a calibrated confidence
interval for the integer plan.""",
r"""Two cautions govern the reading. First, the reference column is the \emph{planning
midpoint}, not an expectation or median: the cost parameters enter multiplicatively,
so the parameter-box midpoint is not the centre of the cost distribution, and the Q2
midpoint of 617,056 sits well above the joint median of 484,372. Second, every Q2 and
Q3 interval is an \emph{LP-proxy} interval: at 11 representative parameter points the
integer chain deviates from the proxy by at most 38,986.30 CNY/day, about 0.2 standard
deviations of the Q2 total, so the intervals describe the shape of the parameter
response, not a calibrated confidence interval for the integer plan."""),

("sec9_sensitivity.tex", r"""The closure failures dominate, and the signs are the
opposite of what a first guess suggests. Heterogeneity in use is the largest: the
recapture rate rises monotonically from 0.4945 for one-trip bicycles to 0.8039 for
11-or-more-trip bicycles""",
r"""The closure failures dominate, and their signs are
not what a first guess suggests. Heterogeneity in use is the largest: the
recapture rate rises monotonically from 0.4945 for one-trip bicycles to 0.8039 for
11-or-more-trip bicycles"""),
]


def main():
    cache = {}
    fails = 0
    for fname, old, new in EDITS:
        path = "paper/" + fname
        if path not in cache:
            cache[path] = io.open(path, encoding="utf-8").read()
        n = cache[path].count(old)
        if n != 1:
            print("FAIL %-22s count=%d  %r" % (fname, n, old.splitlines()[0][:60]))
            fails += 1
            continue
        to, tn = TOK.findall(old), TOK.findall(new)
        assert sorted(to) == sorted(tn), (fname, to, tn)
        cache[path] = cache[path].replace(old, new, 1)
        print("ok   %-22s %r" % (fname, old.splitlines()[0][:56]))
    if fails:
        print("\n%d failed; nothing written." % fails)
        return 1
    for path, txt in cache.items():
        io.open(path, "w", encoding="utf-8", newline="\n").write(txt)
        print("wrote", path)
    print("all %d trims applied" % len(EDITS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
