# T021B trim pass 2: tighten redundant prose only. No numeric token may change.
import io, re, sys
TOK = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?")

EDITS = [
("sec5_q2.tex", r"""The hold-out replay is the external check. The plan is fixed on the five training days
and replayed on the two held-out days with their observed demand. On 12 May, at
0.9291 times the design volume, the plan leaves 704.74 departures unserved and ends
53,255.64 bicycles away from balance; on 19 May, at 1.1066 times the design volume, it
leaves 14,232.37 unserved, 0.60\% of that day's departures, and 51,718.20 away from
balance. No zone-hour hits the truck capacity in either replay. The service cost of a
volume surprise is therefore small in percentage terms, while the end-of-day balance
deteriorates sharply: the plan is built for the typical day, not for the tail.""",
r"""The hold-out replay is the external check: the plan is fixed on the five training
days and replayed on the two held-out days with their observed demand. On 12 May, at
0.9291 times the design volume, it leaves 704.74 departures unserved and ends
53,255.64 bicycles away from balance; on 19 May, at 1.1066 times the design volume, it
leaves 14,232.37 unserved, 0.60\% of that day's departures, and 51,718.20 away from
balance. No zone-hour hits the truck capacity in either replay. The service cost of a
volume surprise is small in percentage terms, but the end-of-day balance deteriorates
sharply: the plan is built for the typical day, not for the tail."""),

("sec5_q2.tex", r"""The numbers above are a planning midpoint, not an expectation or a median. Propagating
the grouped parameter draws of Section~\ref{sec:sens} through the LP proxy gives a Q2 total of \textbf{617,056} at
the midpoint, with a joint P5--P95 interval of \textbf{271,732--882,188} and a median
of 484,372; the planning midpoint sits above the distribution median because the cost
parameters enter multiplicatively and the reference is a point, not the mean.""",
r"""The numbers above are a planning midpoint, not an expectation or a median. The
grouped parameter draws of Section~\ref{sec:sens}, propagated through the LP proxy, give
a Q2 total of \textbf{617,056} at the midpoint, a joint P5--P95 interval of
\textbf{271,732--882,188} and a median of 484,372; the midpoint sits above the median
because the cost parameters enter multiplicatively and the reference is a point, not
the mean."""),

("sec5_q2.tex", r"""Two boundaries follow. The 975-tricycle fleet is an upper bound on the executable
minimum (the ideal LP uses 889), and the 59.385\% saving is a lower bound on the ideal
LP saving (61.06\%); both are reported as bounds, not as point estimates. The intervals
are LP-proxy intervals: at 11 representative parameter points the integer chain deviates
from the proxy by at most 38,986.30 CNY per day, about 0.2 standard deviations of the
Q2 total, so the interval is a shape statement, not a calibrated confidence interval.""",
r"""Two boundaries follow. The 975-tricycle fleet is an upper bound on the executable
minimum (the ideal LP uses 889), and the 59.385\% saving is a lower bound on the ideal
LP saving (61.06\%); both are bounds, not point estimates. The intervals are LP-proxy
intervals: at 11 representative parameter points the integer chain deviates from the
proxy by at most 38,986.30 CNY per day, about 0.2 standard deviations of the Q2 total,
so they are a shape statement, not a calibrated confidence interval."""),

("sec7_q4.tex", r"""The number of fences is chosen as a knee of the capacity--coverage frontier. Let $n$ be
the number of open sites and $c(n)$ the occupancy coverage of the capped $n$-site plan;
both series are normalised to $[0,1]$ over the candidate range, and the selected plan
is the point whose normalised $(n,c)$ pair lies farthest from the chord joining the
first and last frontier points (the Kneedle rule). It is the point where additional
fences stop buying coverage at the frontier's average rate. It is not the minimiser of
a cost function: with every site capped at 150 slots, slots$(n)=150n$ and
$\text{cost}(n)=\text{slots}(n)+\lambda n$ is monotone increasing, so it has no interior
minimum. The exchange rate $\lambda$ is used only for the \emph{uncapped} reference
below, where a fence has no hard size limit.""",
r"""The number of fences is chosen as a knee of the capacity--coverage frontier. Let $n$ be
the number of open sites and $c(n)$ the occupancy coverage of the capped $n$-site plan;
both are normalised to $[0,1]$, and the plan is the point whose normalised $(n,c)$ pair
lies farthest from the chord joining the first and last frontier points (the Kneedle
rule): additional fences stop buying coverage at the frontier's average rate. It is not
a cost minimiser --- with every site capped at 150 slots, slots$(n)=150n$ and
$\text{cost}(n)=\text{slots}(n)+\lambda n$ is monotone increasing --- so the exchange
rate $\lambda$ is used only for the \emph{uncapped} reference below."""),

("sec7_q4.tex", r"""0.20982). The marginal curve explains why: coverage rises by about 1.5 percentage points
per 100 sites up to 3,500, by 1.19 points between 3,500 and 4,000, then by 0.26 points
to 4,500 and 0.05 points to 5,000. Beyond 4,000, sites are expensive and buy little.""",
r"""0.20982). Coverage rises by about 1.5 percentage points per 100 sites up to 3,500, by
1.19 points between 3,500 and 4,000, then by 0.26 points to 4,500 and 0.05 points to
5,000: beyond 4,000, sites are expensive and buy little."""),

("sec8_q5.tex", r"""The Q4 plan is fixed: 4,000 sites, 150 slots each, $R=450$ m, initial-inventory proxy
343,638. Two platforms share the market equally (share $s=0.5$), each drawing its own
Poisson arrivals and departures at half rate, each holding half of every cell's
standing inventory. Three institutions are compared:""",
r"""The Q4 plan is fixed: 4,000 sites, 150 slots each, $R=450$ m, initial-inventory proxy
343,638. Two platforms share the market equally (share $s=0.5$), each drawing Poisson
arrivals and departures at half rate and holding half of every cell's inventory. Three
institutions are compared:"""),

("sec8_q5.tex", r"""\textbf{Cost.} The rebalancing plan of Q2 is computed over the whole city. A platform
with a third of the trips still faces the same 290 arcs and the same nightly window, but
fills each truck with a third as many bicycles. The fixed component of the truck fleet
--- 600 CNY per truck per day, which is \textbf{42.96\%} of the daily cost at $K=60$
(36,000 of 83,807.74 CNY) --- is therefore incurred by every platform separately. The
same argument applies to the tricycle layer's fixed cost, 300 CNY per tricycle-day, which
is 293,342.61 of the 617,055.7515 two-tier total.""",
r"""\textbf{Cost.} The rebalancing plan of Q2 is computed over the whole city; a platform
with a third of the trips still faces the same 290 arcs and nightly window but fills
each truck with a third as many bicycles. The truck fleet's fixed component --- 600 CNY
per truck per day, \textbf{42.96\%} of the daily cost at $K=60$ (36,000 of 83,807.74
CNY) --- is therefore incurred by every platform separately, and the same holds for
the tricycle layer's fixed cost, 300 CNY per tricycle-day, which is 293,342.61 of the
617,055.7515 two-tier total."""),

("sec8_q5.tex", r"""\textbf{Data.} This is the most serious loss and the least visible. Every model in this
paper is estimated from the full origin--destination matrix. A platform holding a third
of the market observes a third of the rows, so the attraction $A_j$ of
\eqnref{eq:gravity} and the cell deficit $\Delta_c$ used in \eqnref{eq:q3lp} are estimated
with a standard error that grows as $\sqrt{3}$. More decisively, \emph{no platform can
compute the plans of Q2 and Q4 at all}, because the Q2 truck plan is defined on a
city-wide zone network, the Q4 layout on a city-wide demand surface, and the Q3 reward on
a city-wide deficit list. Each is a public-good computation that only the city can run.""",
r"""\textbf{Data.} This is the most serious loss and the least visible. Every model here is
estimated from the full origin--destination matrix, and a platform holding a third of
the market observes a third of the rows, so the attraction $A_j$ of
\eqnref{eq:gravity} and the cell deficit $\Delta_c$ of \eqnref{eq:q3lp} are estimated
with a standard error that grows as $\sqrt{3}$. More decisively, \emph{no platform can
compute the plans of Q2 and Q4 at all}: the Q2 truck plan is defined on a city-wide
zone network, the Q4 layout on a city-wide demand surface, and the Q3 reward on a
city-wide deficit list, each a public-good computation that only the city can run."""),

("sec9_sensitivity.tex", r"""The zero bars are informative, not missing. They show that an input which matters in one
question may be irrelevant in another, and that a single global sensitivity ranking would
be misleading.""",
r"""The zero bars are informative, not missing: an input that matters in one question may
be irrelevant in another, so a single global sensitivity ranking would be misleading."""),

("sec9_sensitivity.tex", r"""\item \emph{systems with stations}, where the docking constraint changes the inventory
  process and the rebalancing problem becomes a routing problem with hard capacity;
\item \emph{low-density cities}, where the walking radius of Q4 covers a much smaller
  share of arrivals and the incentive pool of Q3 vanishes;
\item \emph{other layouts}, where the 4,000-site, 150-slot result is specific to this
  0.005$^\circ$ grid, this walking radius and demand surface;
\item \emph{extreme weather or holidays}, not represented in the fourteen recorded dates
  (of which five training and two held-out days drive the models; 17 May has no records);
\item \emph{the long run}, where the attraction $A_j$ itself changes with land use. The
  model is a description of an equilibrium, not a forecast of one.""",
r"""\item \emph{systems with stations}, where docking changes the inventory process and
  rebalancing becomes a routing problem with hard capacity;
\item \emph{low-density cities}, where the walking radius of Q4 covers a much smaller
  share of arrivals and the incentive pool of Q3 vanishes;
\item \emph{other layouts}: the 4,000-site, 150-slot result is specific to this
  0.005$^\circ$ grid, this walking radius and demand surface;
\item \emph{extreme weather or holidays}, not represented in the fourteen recorded
  dates (five training and two held-out days drive the models; 17 May has no records);
\item \emph{the long run}, where the attraction $A_j$ itself changes with land use; the
  model describes an equilibrium, it does not forecast one."""),
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
