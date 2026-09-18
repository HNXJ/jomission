"""The whole-system checkpoint architecture, as a typed TFNE/2 expression.

    x : A : y

    x := Retina^1024,  G[x] = 32 x 32
    A := {V1^2 X[lat]} O[fffb] {V4^2 X[lat]} O[fffb] {FEF X[lat] PFC}
    y := FEF.L6.E.v(t)

Rules are declared here, never inferred. The two O adjacencies carry different mapping
policies, both decided by the owner on 2026-09-18:

    V1 bank -> V4 bank            matched instance, V1.i <-> V4.i   (a frontier override)
    V4 bank -> {FEF X[lat] PFC}   every V4 to both targets          (the derived frontier union)

The override matters: taking the derived frontiers literally would relate all four V1/V4
pairings, so matched-instance identity must be declared to be real.
"""

from __future__ import annotations

from jomission.tfne import ctx
from jomission.tfne.nf import Address, Composite, Instance, OChain, Projection

RETINA = {"name": "Retina", "N": 1024, "G": (32, 32),
          "provenance": "DEFINITION_CHOICE: the checkpoint declares a 32 x 32 retinal field"}

RULES = {
 "lat": {
  "kind": "X",
  "declaration": "X[lat] := [ $L.L3.E <>[AMPA] $R.L3.E ]",
  "associative": False,
  "clauses": [{"source": "L3.E", "target": "L3.E", "mechanism": "AMPA", "bidirectional": True}],
 },
 "fffb": {
  "kind": "O",
  "declaration": ("O[fffb] := [ $L.out >[AMPA] $R.in , $R.fb_out >[AMPA] $L.fb_in ]"),
  "mapping": "derived_frontier_union",
  "clauses": [
   {"name": "ff.L2", "direction": "forward", "source": "L2.E", "target": "L4.E", "mechanism": "AMPA"},
   {"name": "ff.L3", "direction": "forward", "source": "L3.E", "target": "L4.E", "mechanism": "AMPA"},
   {"name": "fb.L6", "direction": "backward", "source": "L6.E", "target": "L1.E", "mechanism": "AMPA"},
  ],
 },
}
# The V1 -> V4 adjacency overrides the derived frontier with matched-instance identity.
RULES["fffb_matched"] = dict(RULES["fffb"], mapping="matched_instance",
                             declaration=RULES["fffb"]["declaration"] + "  with matched-instance frontier override")

MAPPING_AUTHORITY = {
 "V1_to_V4": ("owner 2026-09-18: matched instance V1.1 <-> V4.1, V1.2 <-> V4.2. DEFINITION_CHOICE: "
              "a frontier override, since the derived union would relate all four pairings"),
 "V4_to_frontal": ("owner 2026-09-18: every V4 instance to both FEF and PFC. DERIVED: this is the "
                   "frontier union in({FEF X[lat] PFC}) = in(FEF) u in(PFC) with no override"),
}


def build(n_per_instance=ctx.N_PER_INSTANCE):
    """Return (chain, rules, external projections, declaration) for the checkpoint architecture."""
    decl = ctx.declaration(n_per_instance)

    def inst(*path):
        return Instance(tuple(path), ctx.NAME, decl)

    v1 = Composite([inst("V1", "1"), inst("V1", "2")], relation="lat", label="{V1^2 X[lat]}")
    v4 = Composite([inst("V4", "1"), inst("V4", "2")], relation="lat", label="{V4^2 X[lat]}")
    frontal = Composite([inst("FEF"), inst("PFC")], relation="lat", label="{FEF X[lat] PFC}")

    chain = OChain([v1, v4, frontal], ["fffb_matched", "fffb"])

    # x := Retina^1024 into in({V1^2 X[lat]}). Structural only: the retinal units are declared
    # and their entry points resolved, but no retinal population is constructed in this lineage.
    external = tuple(Projection(Address(("Retina",), "field", "unit"), target, "AMPA", "x->A.in")
                     for target in v1.frontier("in"))
    return chain, RULES, external, decl


OUTPUT_ADDRESS = ("FEF", "L6", "E")
OUTPUT_DECLARATION = "y := FEF.L6.E.v(t)"
