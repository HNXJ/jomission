"""Item 2 source-digest check for the laminar motif plan.

Usage: PYTHONUTF8=1 python item02_digest_check.py <dir with extracted _temp_*.txt>
Each row asserts that a value quoted in the skill or the seal proposal appears
in its source text, and prints the source page. Exit 1 if any row is unmatched.
Sources: M manuscript, D deck v4, T25 Tahvili 2025, T26 Tahvili 2026.
Text extraction: PyMuPDF page.get_text(), pages marked '=== pN ==='.
"""
import glob
import re
import sys

S = sys.argv[1]


def load(key):
    return open(glob.glob(S + "/_temp_" + key + "*.txt")[0], encoding="utf-8").read()


src = {"M": load("Laminar-Cellular"), "D": load("Spectrolaminar"), "T25": load("Tahvili2025"), "T26": load("Tahvili2026")}


def page(t, pos):
    ps = [m for m in re.finditer(r"=== p(\d+) ===", t) if m.start() <= pos]
    return ps[-1].group(1) if ps else "?"


W = r"\s*"  # PDF text breaks lines at table cells
G = r"[\s-]*"  # whitespace and private-use bracket glyphs between matrix cells
C = [
    # --- manuscript: reported values
    ("M", r"mean r = 0\.76", "motif conservation 0.76"),
    ("M", r"[Hh]igh gamma power \(91-150 Hz\) was the most conserved \(mean r = 0\.89", "high gamma conservation 0.89"),
    ("M", r"beta \(13-30 Hz; mean r = 0\.78", "beta conservation 0.78"),
    ("M", r"alpha \(8-12 Hz; mean r = 0\.63", "alpha conservation 0.63"),
    ("M", r"CR interneurons were the most conserved \(mean r = 0\.81", "CR density conservation 0.81"),
    ("M", r"CB \(mean r =\s*0\.70", "CB 0.70"),
    ("M", r"PV \(mean r = 0\.64", "PV 0.64"),
    ("M", r"NG \(mean r = 0\.53", "NG 0.53"),
    ("M", r"90\.93%", "anatomy PCA cumulative 90.93"),
    ("M", r"99\.87%", "function PCA cumulative 99.87"),
    ("M", r"62\.47%", "PC1X 62.47"),
    ("M", r"28\.46%", "PC2X 28.46"),
    ("M", r"71\.53%", "PC1Y 71.53"),
    ("M", r"28\.35%", "PC2Y 28.35"),
    ("M", r"mean Q2 = 0\.42, mean rPRED = 0\.71", "PLSR Q2 0.42, r 0.71"),
    ("M", r"mean r = 0\.992", "beta coefficient conservation 0.992"),
    ("M", r"\(p" + W + r"=" + W + r"0\.054\)", "SZ beta trend p 0.054"),
    ("M", r"d\(.{1,3}\) = 0\.756", "Procrustes cutoff X 0.756"),
    ("M", r"0\.752", "Procrustes cutoff Y 0.752"),
    ("M", r"alpha/beta \(Q2 = 0\.40.0\.75", "alpha/beta Q2 range 0.40-0.75"),
    ("M", r"rPRED =\s*0\.71.0\.88", "alpha/beta r range 0.71-0.88"),
    ("M", r"high gamma \(Q2 = 0\.37.0\.66", "high gamma Q2 range 0.37-0.66"),
    ("M", r"low-frequency gamma \(Q2 =\s*-0\.10.0\.67", "low gamma Q2 range -0.10-0.67"),
    # --- manuscript: definitions used by the seal proposal
    ("M", r"Relative Power\(c,f\)|𝑅𝑒𝑙𝑎𝑡𝑖𝑣𝑒 𝑃𝑜𝑤𝑒𝑟", "eq. 1 relative power (max over channels)"),
    ("M", r"-0\.6:0\.4 with steps of 0\.05, creating 21 cortical bins", "21 bins, -0.6:0.4, step 0.05"),
    ("M", r"60% above the middle of layer 4 and 40% below", "60/40 split about mid-L4"),
    ("M", r"z-score to the feature dimension.*not cortical bins", "z-score on feature dimension"),
    ("M", r"at least 90% of the cumulative variance", "PCs retained to >= 90% cumulative variance"),
    ("M", r"\|BSR\| ≥ 2\.5|bootstrap threshold of 2\.5", "bootstrap ratio 2.5"),
    ("M", r"N = 5,000 permutation", "5,000 permutations"),
    ("M", r"A 4-component model was selected", "4 PLSR components"),
    ("M", r"average of per-frequency correlations", "rPRED = mean of per-frequency correlations"),
    ("M", r"two-sided Wilcoxon signed-rank test against zero across areas", "Wilcoxon across areas"),
    ("M", r"Fisher.s combined probability test", "Fisher combined test"),
    ("M", r"delta \[1-3 Hz\], theta \[4-7 Hz\], alpha\s+\[8-12 Hz\], beta \[13-30 Hz\], low frequency gamma\s+\[31-90 Hz\], high frequency gamma \[91-150 Hz\]", "six bands"),
    ("M", r"peaking in L5/6, and a laminar gradient of increasing gamma power from deep layers to middle layers,\s+peaking in L3", "alpha-beta peak L5/6, gamma peak L3"),
    ("M", r"all r\s+≥ 0\.61, mean r = 0\.70", "cross-area prediction r >= 0.61, mean 0.70"),
    ("M", r"mean r = 0\.88, p < 0\.0001", "own-area prediction 0.88"),
    # --- deck v4
    ("D", r"1050 Izhikevich", "1050 neurons"),
    ("D", r"483 E, 205 PV, 167 SST, 195 VIP", "fractions 483/205/167/195"),
    ("D", r"within 0\.021", "fractions within 0.021"),
    ("D", r"E 4\.0, PV 4\.0, SST\s+0\.5, VIP 25", "drives"),
    ("D", r"E 4\.9, PV 6\.7, SST 6\.4, VIP 11\.2", "rates"),
    ("D", r"Motif r of the area means 0\.44", "motif r of area means 0.44"),
    ("D", r"Held-out r 0\.30 \(−0\.15 to 0\.52", "held-out r 0.30 (-0.15 to 0.52)"),
    ("D", r"One circuit wins 10 of 11 folds", "10 of 11 folds"),
    ("D", r"5 of 11 simulated areas \(10 of 11 observed\)", "gamma above alpha-beta 5/11 vs 10/11"),
    ("D", r"conservation r 0\.63 \(before 0\.54; observed 0\.76\)", "conservation 0.63 (0.54; 0.76)"),
    ("D", r"LOAO\s+r 0\.70 \(0\.71\)", "LOAO r 0.70 (0.71)"),
    ("D", r"Q² 0\.37 \(before 0\.16; 0\.42\)", "Q2 0.37 (0.16; 0.42)"),
    ("D", r"96/2% vs 72/28%", "PCA split 96/2 vs 72/28"),
    ("D", r"59/16% vs 72/28%", "irregular-firing PCA 59/16"),
    ("D", r"v1 0\.43.*full-width bins 0\.48, observed 0\.86", "edge/interior 0.43, 0.48, 0.86"),
    ("D", r"E CV 0\.14", "E CV 0.14"),
    ("D", r"E CV 0\.42, 16 configs", "E CV 0.42 with Poisson background, 16 configs"),
    ("D", r"VIP \(22\.6\)", "VIP rheobase 22.6"),
    ("D", r"113", "113 of 288 settings kept"),
    ("D", r"120 Hz −0\.40% vs −0\.13%", "PLSR 120 Hz -0.40% vs -0.13%"),
    ("D", r"7 of 11 areas", "knockdown outside seed range in 7 of 11 areas"),
    # --- Tahvili 2025 (Cell Reports 44:116131): neurons
    ("T25", r"−50 mV", "VT -50"),
    ("T25", r"2, 0\.5, 1\.5 mV", "Delta 2, 0.5, 1.5"),
    ("T25", r"130, 0, 25 pA", "b 130, 0, 25"),
    ("T25", r"−60, −60, −55 mV", "EL -60, -60, -55"),
    ("T25", r"4, 0, 4 nS", "a 4, 0, 4"),
    ("T25", r"500 ms", "tau_w 500"),
    ("T25", r"8,000 E neu.{0,6}rons, 800 inhibitory interneurons representing PV\+ cells, and 600", "N 8000/800/600"),
    ("T25", r"around 7 ms, or 80", "PV lead ~7 ms, 80 deg"),
    ("T25", r"no synaptic delay|deliberately do not consider any synaptic delay", "no synaptic delay in base model"),
    # --- Tahvili 2025: synapses (Table 2)
    ("T25", r"Table 2\. Synaptic parameters" + W + r"Parameter" + W + r"Value" + W + r"τPV;SOM;E" + W + r"E" + W + r"1, 2, 5 ms" + W + r"τI" + W + r"5 ms", "tauE (PV,SOM,E)=(1,2,5) ms; tauI 5 ms"),
    ("T25", r"0 mV" + W + r"Erev" + W + r"I" + W + r"−80 mV", "Erev E 0, I -80"),
    ("T25", r"QPV.{1,3}\{E;PV\}" + W + r"6 nS", "Q PV->{E,PV} 6"),
    ("T25", r"QSOM.{1,3}\{E;PV\}" + W + r"5 nS", "Q SOM->{E,PV} 5"),
    ("T25", r"QE.{1,3}E" + W + r"1\.25 nS", "Q E->E 1.25"),
    ("T25", r"QE.{1,3}PV" + W + r"3\.75 nS", "Q E->PV 3.75"),
    ("T25", r"QE.{1,3}SOM" + W + r"2\.5 nS", "Q E->SOM 2.5"),
    ("T25", r"QE.{1,3}Ext" + W + r"1\.25 nS", "Q Ext 1.25 (as printed: 'QE->Ext')"),
    # --- Tahvili 2026 (PLoS CB 22:e1014378): neurons
    ("T26", r"2, 0\.5, 2 mV", "Delta 2, 0.5, 2"),
    ("T26", r"50, 0, 80 pA", "b 50, 0, 80"),
    ("T26", r"-65, -65, -55 mV", "EL -65, -65, -55"),
    ("T26", r"VT \+ 5", "spike threshold VT + 5 Delta"),
    ("T26", r"NE = 8000.*NI = 1500.*NPV = 750.*NSOM = 750", "N 8000/750/750 (NI 1500)"),
    ("T26", r"probability\s+p\s*=\s*0\.07", "p = 0.07"),
    ("T26", r"35 Hz, with a PV.{1,5}SOM phase delay of approximately 7 ms", "35 Hz, PV-SOM delay ~7 ms"),
    # --- Tahvili 2026: synapses (Table 2) and mean field
    ("T26", r"Table 2\.\s+Synaptic parameters\." + W + r"Parameter" + W + r"Value" + W + r"τd" + W + r"2 ms" + W + r"τ E" + W + r"2 ms" + W + r"τ I" + W + r"8 ms", "tau_d 2, tauE 2, tauI 8 ms"),
    ("T26", r"EE" + W + r"rev" + W + r"0 mV" + W + r"EI" + W + r"rev" + W + r"-80 mV", "Erev E 0, I -80"),
    ("T26", r"Q\{PV,SOM\}.{1,3}\{E,PV\}" + W + r"5 nS", "Q {PV,SOM}->{E,PV} 5"),
    ("T26", r"QE.{1,3}E" + W + r"2 nS", "Q E->E 2"),
    ("T26", r"QE.{1,3}\{PV,SOM\}" + W + r"3 nS", "Q E->{PV,SOM} 3"),
    ("T26", r"QExt.{1,3}\{E,PV\}" + W + r"2\.5 nS", "Q Ext->{E,PV} 2.5"),
    ("T26", r"QPV.{1,3}SOM = 0 and\s+QSOM.{1,3}SOM = 0", "Q PV->SOM = Q SOM->SOM = 0"),
    ("T26", r"independent Poisson spike trains at 4 Hz", "external Poisson 4 Hz"),
    ("T26", r"chosen to be 15 ms", "mean-field T = 15 ms"),
    ("T26", r"τd = 2 ms", "mean-field delay 2 ms"),
    ("T26", r"QSOM.{1,3}E = 4 and so QSOM.{1,3}PV = 2\.05", "Q SOM->E 4, SOM->PV 2.05"),
    ("T26", r"τ\s+to be 20 ms", "linear-model tau 20 ms"),
    ("T26", r"0\.02" + G + r"4" + G + r"–5" + G + r"–4" + G + r"5" + G + r"–5" + G + r"–2\.05" + G + r"4" + G + r"0" + G + r"–1", "Eq. 10 matrix"),
    ("T26", r"PV has a 4 ms phase advance relative to SOM", "linear model PV lead 4 ms"),
]
bad = 0
for k, rx, lab in C:
    t = src[k]
    m = re.search(rx, t, re.S)
    if m:
        print(f"OK   {k:3} p{page(t, m.start()):>2}  {lab}")
    else:
        bad += 1
        print(f"MISS {k:3}      {lab}   <{rx[:80]}>")
print("unmatched:", bad, "of", len(C))
sys.exit(1 if bad else 0)
