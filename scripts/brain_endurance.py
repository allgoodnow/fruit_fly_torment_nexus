"""Measure the standalone runtime, including bounded history during long input."""
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
os.environ.setdefault("NUMBA_CACHE_DIR",str(ROOT / ".runtime/numba"))
from nexus.brain.runtime import Brain,Connectome
from nexus.brain.targets import SUGAR,MN9

brain = Brain(Connectome.load(ROOT / "data/brain-v630"))
brain.advance(.0001)
brain.reset()
brain.stimulate(SUGAR,200)
start = time.perf_counter()
samples = []
for i in range(1000):
    if i == 800:
        brain.release()
    brain.advance(.01)
    if (i+1)%100 == 0:
        row = {"time":brain.time,"history":len(brain.history),
               "total_spikes":int(brain.counts.sum()),"mn9_spikes":int(brain.counts[brain.lookup[int(MN9)]]),
               "max_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        samples.append(row)
        print(json.dumps(row),flush=True)
result = {"sim_seconds":brain.time,"wall_seconds":time.perf_counter()-start,
          "stimulus":"200 Hz on 21 sugar cells for 8 s; released for 2 s",
          "history_capacity":brain.history.maxlen,"samples":samples}
out=ROOT / "runs/brain_endurance"
out.mkdir(parents=True,exist_ok=True)
(out / "report.json").write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
