"""Full-connectome numerical comparison and persistent runtime measurements."""
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor/brain-reference"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".runtime/numba"))
os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")
import numpy as np
from nexus.brain.runtime import Brain, Connectome

out = ROOT / "runs/persistent_brain"
out.mkdir(parents=True, exist_ok=True)
assay = json.loads((ROOT / "runs/brain_assay_630/report.json").read_text())
sugar = assay["input_ids"]
mn9 = assay["readout_id"]
started = time.perf_counter()
brain = Brain(Connectome.load(ROOT / "data/brain-v630"))
load_seconds = time.perf_counter()-started
brain.stimulate(sugar, 200)
events = np.random.default_rng(227).random((1000, len(sugar))) < .02
started = time.perf_counter()
actual = brain.advance(.1, input_events=events, trace_ids=[mn9])
compile_and_run = time.perf_counter()-started
print(f"Loaded in {load_seconds:.3f}s; first 100ms + JIT: {compile_and_run:.3f}s", flush=True)

import brian2 as b
import model
b.start_scope()
b.prefs.codegen.target = "cython"
b.prefs.codegen.runtime.cython.cache_dir = str(ROOT / ".runtime/brian-cython")
b.defaultclock.dt = .1*b.ms
source = ROOT / "vendor/brain-reference"
neu, syn, mon = model.create_model(source / "2023_03_23_completeness_630_final.csv",
                                   source / "2023_03_23_connectivity_630_final.parquet", model.default_params)
targets = brain.resolve(sugar)
neu.rfc[targets] = 0*b.ms
@b.network_operation(when='synapses', order=0)
def replay():
    tick = round(float(b.defaultclock.t/b.ms)*10)
    selected = targets[events[tick]]
    selected = selected[np.asarray(neu.not_refractory[:])[selected]]
    neu.v[selected] += 68.75*b.mV
trace = b.StateMonitor(neu, 'v', record=[brain.lookup[int(mn9)]], when='end')
net = b.Network(neu, syn, mon, trace, replay)
net.run(100*b.ms)
np.testing.assert_array_equal(actual['indices'], mon.i[:])
np.testing.assert_array_equal(actual['steps'], np.rint(mon.t[:]/b.ms*10).astype(int))
max_error = float(np.max(np.abs(brain.v-neu.v[:]/b.mV)))
np.testing.assert_allclose(brain.v, neu.v[:]/b.mV, atol=1e-8, rtol=0)
np.testing.assert_allclose(brain.g, neu.g[:]/b.mV, atol=1e-8, rtol=0)
print(f"Full-network agreement: {len(actual['indices'])} identical spikes, max voltage error {max_error:.3g} mV", flush=True)

brain.reset()
timings = []
records = []
for label, count in (("baseline",10), ("stimulated",50), ("released",50)):
    if label == "stimulated":
        brain.stimulate(sugar, 200)
    elif label == "released":
        state = [brain.v.copy(), brain.g.copy(), brain.queue_size.copy(), brain.queue.copy()]
        brain.release()
        for before, after in zip(state, (brain.v, brain.g, brain.queue_size, brain.queue)):
            np.testing.assert_array_equal(before, after)
    before = brain.counts.copy()
    for _ in range(count):
        start = time.perf_counter()
        brain.advance(.01)
        timings.append(time.perf_counter()-start)
    records.append({"condition":label,"duration_s":count*.01,"time":brain.time,
                    "spikes":int((brain.counts-before).sum()),
                    "mn9_spikes":int((brain.counts-before)[brain.lookup[int(mn9)]])})
result = {"neurons":len(brain.graph.ids),"edges":len(brain.graph.posts),
          "numerical_comparison_ms":100,"identical_spikes":len(actual['indices']),
          "max_voltage_error_mv":max_error,"load_seconds":load_seconds,
          "jit_and_first_100ms_seconds":compile_and_run,
          "chunk_ms":10,"median_chunk_wall_ms":float(np.median(timings)*1000),
          "p95_chunk_wall_ms":float(np.percentile(timings,95)*1000),
          "simulated_seconds":1.1,"wall_seconds":sum(timings),
          "realtime_factor":1.1/sum(timings),"max_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          "records":records,"release_preserves_voltages_currents_and_delay_queue":True}
(out / "report.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
