"""T004 delivery validator: recompute every headline from the result files."""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "data", "runs", "T004")
D = json.load(io.open(os.path.join(ROOT, "交付", "T004_delivery.json"),
                      encoding="utf-8"))


def load(n):
    return json.load(io.open(os.path.join(RUN, n), encoding="utf-8"))


sim_old = load("sim_results.json")
sim_new = load("sim_fine.json")
neigh = load("neighbours.json")
neigh_f = load("neighbours_fine.json")
fine = load("fine_grid.json")
dry = load("dry_cells.json")
robust = load("dry_172_robustness.json")
cons = load("conservation.json")

# checks
for k, v in D["checks"].items():
    if k.startswith(tuple(str(i) for i in range(1, 11))):
        assert v is True, k
assert D["paper_integrity"]["paper"]["unchanged"] is True
assert D["paper_integrity"]["data_excluding_runs"]["unchanged"] is True
assert D["paper_integrity"]["cell_fleet_curve_unchanged"] is True
assert D["paper_integrity"]["cell_sim_inputs_unchanged"] is True
assert D["paper_integrity"]["lib_py_unchanged"] is True
assert D["paper_integrity"]["sim_cell_py_changed"] is True
assert "code/sim_cell.py" in D["code_integrity"]["t004_modified"]
for f in D["code_integrity"]["pre_existing_since_T001"]:
    assert f in D["code_integrity"]["changed"]

# headline vs files
h = D["headline"]
assert h["coarse_R300_a1_fleet343638"] == sim_old["results"][
    "coarse_343638_R300_a1.0"]["mean"]
assert h["coarse_R450_a1_fleet343638"] == sim_old["results"][
    "coarse_343638_R450_a1.0"]["mean"]
assert h["coarse_R450_a1_fleet495540"] == sim_new["runs"][
    "coarse_R450_a1.0"]["mean"]
assert h["fine_R450_a1_fleet495540"] == sim_new["runs"][
    "fine_R450_a1.0"]["mean"]
assert h["cells_with_neighbour_300_450_600"] == [
    neigh["radii"][k]["cells_with_neighbour"] for k in ("0.30", "0.45", "0.60")]

# reproduction is exact
assert sim_old["reproduce_old_curve"]["a1.0"]["delta"] == 0.0
assert sim_old["reproduce_old_curve"]["a0.8"]["delta"] == 0.0

# conservation
for grid in ("coarse", "fine"):
    for R, rec in cons[grid].items():
        assert max(abs(x) for x in rec["inventory_residual"]) == 0.0, (grid, R)
        assert min(rec["min_inventory"]) >= 0.0, (grid, R)

# dry-cell definitions
defs = dry["definitions"]
assert defs["hourly_unscaled"]["cells_negative"] == 172
assert defs["5min_scaled"]["cells_negative"] == 2306
assert defs["hourly_scaled"]["cells_negative"] == 2251
assert defs["5min_unscaled"]["cells_negative"] == 175
assert dry["stochastic_343638"]["by_radius"]["R300"][
    "mean_cells_below_one"] == 2463.4
phases = list(robust["phase_offsets_hourly_unscaled"].values())
assert min(phases) == 169 and max(phases) == 174 and 172 in phases
assert robust["arrival_lag"]["11"] == 172

# fine grid
assert fine["n_cells"] == 11269
assert fine["last_trip_end"]["cells_without_evidence"] == 25
assert fine["share_trips_mapped"] < fine["coarse_support"]["share_trips_mapped"]

# replacement text
for key, name in (("abstract_24_27", "abstract"), ("sec9_106_108", "sec9")):
    rec = D["replacement_text"][key]
    assert rec["word_count"] == len(rec["english"].split())
    assert rec["word_count"] <= 120
    assert "13.4" in rec["english"]
    assert "no zone ever runs out" not in rec["english"]

# case coverage: 6 coarse 343638 + 8 bridge/fine + 2 fine sensitivities
labels = [c["label"] for c in D["cases"]]
assert len(labels) == len(set(labels))
for need in ("coarse_343638_R300_a1.0", "coarse_343638_R450_a1.0",
             "coarse_343638_R600_a1.0", "coarse_343638_R300_a0.8",
             "fine_R450_a1.0", "fine_R300_a1.0", "fine_R600_a1.0",
             "fine_R450_a1.0_I0zero", "fine_R450_a1.0_demandmatched"):
    assert need in labels, need

# all declared outputs exist
for rel in D["outputs"].values():
    assert os.path.exists(os.path.join(ROOT, rel)), rel

print("PASS: checks, headline, conservation, dry definitions, fine grid, text, "
      "case coverage, outputs")
print("cases", len(labels), "comparisons", len(D["comparisons"]))
