#!/usr/bin/env python3
"""Run the course's evaluate_answers.py unchanged, except that a judge reply with malformed JSON
falls back to extracting the three integer scores with a regex (strict parse is tried first)."""
import re, runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.evaluation.answer_judge as aj

recovered = [0]
for name, cls in list(vars(aj).items()):
    if isinstance(cls, type) and "parse" in vars(cls):
        strict = cls.parse
        def tolerant(raw, _strict=strict):
            try:
                return _strict(raw)
            except Exception:
                obj = {}
                for k in ("correctness", "grounding", "completeness"):
                    m = re.search(rf'"{k}"\s*:\s*(\d+)', raw)
                    if not m or not 0 <= int(m.group(1)) <= 4:
                        raise
                    obj[k] = int(m.group(1))
                obj["explanation"] = "[recovered from malformed JSON]"
                recovered[0] += 1
                print(f"  (recovered scores from malformed judge JSON, total {recovered[0]})", flush=True)
                return obj
        cls.parse = staticmethod(tolerant)
        print(f"patched {name}.parse", flush=True)

script = ROOT / "scripts" / "evaluate_answers.py"
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")
