"""Flatten the per-event stage1 tree to a per-jet ntuple for training.

Reads the stage1 `events` tree with uproot (vectorised) and writes the per-jet
`tree` with PyROOT. The one-hot `recojet_is{FLAVOUR}` label is inferred from the
input file name (`H{f}{f}` for Higgs samples, `_ee_{f}{f}_` for Z samples).

Usage: python stage2.py input_file output_file n_start n_final

Note: a previous per-entry PyROOT implementation read RVec<RVec<float>> branches
with nested cppyy indexing, which corrupts the heap (segfault on tau samples).
Reading the branches vectorised with uproot avoids that and is much faster.
"""
import sys
import numpy as np
import uproot
import awkward as ak
from array import array
from ROOT import TFile, TTree
from examples.FCCee.weaver.config import variables_pfcand, variables_jet, flavors

input_file, output_file = sys.argv[1], sys.argv[2]

branches_pfcand = list(variables_pfcand.keys())
branches_jet = list(variables_jet.keys())
branches_label = ["recojet_is{}".format(f.upper()) for f in flavors]

t = uproot.open(input_file)["tree"]
names = branches_label + branches_jet + branches_pfcand + ["jet_npfcand", "event_id"]
arr = t.arrays(names, library="ak")

event_id = ak.to_numpy(arr["event_id"])
njets_tot = int(event_id.size)

starts = np.flatnonzero(np.r_[True, np.diff(event_id) != 0])
nev = int(starts.size)
njets = np.diff(np.r_[starts, njets_tot])

jet_flat = {b: ak.to_numpy(arr[b]).astype(np.float32) for b in branches_jet}
jet_off = np.zeros(nev + 1, dtype=np.int64)
np.cumsum(njets, out=jet_off[1:])

npf_jet = ak.to_numpy(arr["jet_npfcand"]).astype(np.int64)
npf = np.add.reduceat(npf_jet, starts)

pf_off = np.zeros(nev + 1, dtype=np.int64)
np.cumsum(npf, out=pf_off[1:])
pf_flat = {b: ak.to_numpy(ak.flatten(arr[b], axis=1)).astype(np.float32) for b in branches_pfcand}

jet_index_flat = np.concatenate([
    np.repeat(
        np.arange(njets[i], dtype=np.int32),
        npf_jet[jet_off[i]:jet_off[i + 1]]
    )
    for i in range(nev)
])

label_ev = {b: ak.to_numpy(arr[b])[starts].astype(np.int32) for b in branches_label}

maxn = 500
maxj = int(njets.max())
n_clipped = int((npf > maxn).sum())

out_root = TFile(output_file, "RECREATE")
tr = TTree("tree", "tree with events")

lab_buf = {}
for b in branches_label:
    name = b.replace("recojet_is", "event_is")
    lab_buf[b] = array("i", [0])
    tr.Branch(name, lab_buf[b], "{}/I".format(name))

event_njets = array("i", [0])
tr.Branch("event_njets", event_njets, "event_njets/I")

event_npf = array("i", [0])
tr.Branch("event_npfcand", event_npf, "event_npfcand/I")

event_id_out = array("q", [0])
tr.Branch("event_id", event_id_out, "event_id/L")

jet_index_buf = array("i", maxn * [0])
jet_index_np = np.frombuffer(jet_index_buf, dtype=np.int32)
tr.Branch(
    "pfcand_jet_index",
    jet_index_buf,
    "pfcand_jet_index[event_npfcand]/I"
)

jet_buf, jet_np = {}, {}
for b in branches_jet:
    jet_buf[b] = array("f", maxj * [0.0])
    tr.Branch(b, jet_buf[b], "{}[event_njets]/F".format(b))
    jet_np[b] = np.frombuffer(jet_buf[b], dtype=np.float32)

pf_buf, pf_np = {}, {}
for b in branches_pfcand:
    pf_buf[b] = array("f", maxn * [0.0])
    tr.Branch(b, pf_buf[b], "{}[event_npfcand]/F".format(b))
    pf_np[b] = np.frombuffer(pf_buf[b], dtype=np.float32)

for i in range(nev):
    for b in branches_label:
        lab_buf[b][0] = int(label_ev[b][i])
    event_id_out[0] = int(event_id[starts[i]])

    nj = int(njets[i])
    event_njets[0] = nj
    jo = jet_off[i]
    for b in branches_jet:
        jet_np[b][:nj] = jet_flat[b][jo:jo + nj]

    n = int(min(npf[i], maxn))
    event_npf[0] = n
    po = pf_off[i]
    for b in branches_pfcand:
        pf_np[b][:n] = pf_flat[b][po:po + n]

    jet_index_np[:n] = jet_index_flat[po:po + n]

    tr.Fill()

tr.SetDirectory(out_root)
tr.Write()
out_root.Close()

print("wrote {} events ({} jets) -> {} ({} events exceeded maxn={})".format(
    nev, njets_tot, output_file, n_clipped, maxn))
