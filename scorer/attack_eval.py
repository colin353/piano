import sys, numpy as np
from piano_scorer import attack
notes=[int(x) for x in (sys.argv[1] if len(sys.argv)>1 else "48,60,72,84,96").split(",")]
comps=[]; 
for n in notes:
    r=attack.analyze(n,"FF")
    comps.append(r["composite"])
    print(f"  note {r['note']:>3}: comp {r['composite']:5.2f}  cent {r['cent_div']:5.0f}c  trans r{r['transient_real']:+5.1f}/s{r['transient_syn']:+5.1f}")
print(f"MEAN composite: {np.mean(comps):.3f}")
