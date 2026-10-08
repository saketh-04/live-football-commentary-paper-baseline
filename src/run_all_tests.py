"""
Comprehensive Test Runner for Football Commentary Intelligence System (V1 + V2).
Executes standalone tests across all core modules and reports status.
"""
import sys
import subprocess
import time
from pathlib import Path

MODULES = [
    # V1 Modules
    ("temporal_context.py", "src/temporal_context.py"),
    ("match_state.py", "src/match_state.py"),
    ("event_importance.py", "src/event_importance.py"),
    ("context_retrieval.py", "src/context_retrieval.py"),
    ("factual_verifier.py", "src/factual_verifier.py"),
    # V2 Modules
    ("visual_verifier.py", "src/visual_verifier.py"),
    ("confidence_scorer.py", "src/confidence_scorer.py"),
    ("event_graph.py", "src/event_graph.py"),
    ("counterfactual.py", "src/counterfactual.py"),
    ("commentary_policy.py", "src/commentary_policy.py"),
    ("player_intelligence.py", "src/player_intelligence.py"),
]

def main():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE V1 + V2 TEST SUITE")
    print("=" * 70)
    
    results = {}
    for name, path in MODULES:
        print(f"\n[TESTING] {name} ({path})...")
        t0 = time.time()
        res = subprocess.run([sys.executable, path], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        dt = time.time() - t0
        passed = (res.returncode == 0)
        results[name] = {"passed": passed, "code": res.returncode, "time": dt, "stdout": res.stdout, "stderr": res.stderr}
        
        status = "PASSED" if passed else f"FAILED (code {res.returncode})"
        print(f" -> {status} in {dt:.2f}s")
        if not passed:
            print("STDERR:")
            print(res.stderr[:500])
        else:
            lines = [l for l in res.stdout.strip().split("\n") if l.strip()]
            summary_sample = "\n".join(lines[:3]) if lines else "OK"
            print(f"    Sample Output:\n{summary_sample}")
            
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, data in results.items():
        mark = "[PASS]" if data["passed"] else "[FAIL]"
        if not data["passed"]:
            all_passed = False
        print(f"  {mark:<8} | {name:<25} | {data['time']:.2f}s")
        
    print("=" * 70)
    if all_passed:
        print("ALL 10 CORE MODULE TESTS PASSED PERFECTLY!")
    else:
        print("SOME TESTS FAILED! Review logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
