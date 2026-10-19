#!/bin/python3
# Verification battery for the flash17 2x-Spark Zig serve (localhost:8888).
# Verified run 2026-10-18: 7/7 passed; prefill 2,182 tok/s @ 30K; decode after 30K 85.6 tok/s.
import json, time, urllib.request, subprocess, tempfile, os

URL = "http://localhost:8888/v1/chat/completions"
def chat(messages, effort="low", temperature=0, max_tokens=1024, extra=None):
    p = {"model": "DeepSeek-V4.1-Flash-TF", "messages": messages, "temperature": temperature,
         "reasoning_effort": effort, "max_tokens": max_tokens}
    if extra: p.update(extra)
    t0 = time.time()
    req = urllib.request.Request(URL, data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    return body, time.time() - t0

def tf(b): return b.get("tensorfold", {})
def line(name, ok, b, wall):
    t = tf(b); u = b["usage"]
    dec = t.get("decode_s") or 0
    tps = (u["completion_tokens"] / dec) if dec else 0
    print(f"{'PASS' if ok else 'FAIL'} | {name:26s} | wall {wall:5.1f}s | {u['completion_tokens']:4d} tok | {tps:6.1f} tok/s | drafted {t.get('drafted',0)} accepted {t.get('accepted',0)}")
    return ok

results = []
# the recipe's own gate: must be 391
b, w = chat([{"role":"user","content":"What is 17 * 23?"}], extra={"max_tokens":512})
results.append(line("smoke 17*23=391", "391" in b["choices"][0]["message"]["content"], b, w))
# math reasoning
b, w = chat([{"role":"user","content":"A train travels 120 km in 1.5 hours, then 80 km in 1 hour. What is its average speed for the whole trip in km/h? Answer with just the number."}])
results.append(line("math (80 km/h)", "80" in b["choices"][0]["message"]["content"], b, w))
# code gen + REAL execution
b, w = chat([{"role":"user","content":"Write a Python function fib(n) returning the nth Fibonacci number using memoization. Reply with only a python code block."}], max_tokens=2048)
code = b["choices"][0]["message"]["content"]
if "```" in code:
    code = code.split("```")[1]
    if code.startswith("python"): code = code[6:]
test_src = code + "\nassert fib(10)==55\nassert fib(30)==832040\nassert fib(1)==1\nprint('EXEC-OK')\n"
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write(test_src); path = f.name
r = subprocess.run(["python3", path], capture_output=True, text=True, timeout=60)
os.unlink(path)
results.append(line("code gen + exec", "EXEC-OK" in r.stdout, b, w))
# decode speed
b, w = chat([{"role":"user","content":"Count from 1 to 200, numbers separated by commas only, no other text or explanation."}], max_tokens=1024)
results.append(line("decode speed (1..200)", b["choices"][0]["finish_reason"]=="stop", b, w))
# long prefill (~30K tokens) + needle recall + post-long-prompt decode
para = "The quarterly infrastructure report covers node utilization, storage trends, cooling maintenance windows, and network fabric health metrics for the serving cluster. "
mid = " Midway through the audit the auditor noted the report reference code is DS-4471 for all follow-up filings. "
ctx = para * 1200
long_prompt = ctx[:len(ctx)//2] + mid + ctx[len(ctx)//2:] + "\n\nTwo tasks: (1) What is the report reference code (format DS-XXXX)? (2) Then count from 1 to 100, comma-separated, nothing else."
b, w = chat([{"role":"user","content": long_prompt}], max_tokens=600)
t = tf(b); u = b["usage"]
ps = t.get("prefill_s") or 0; ds = t.get("decode_s") or 0
pf = u["prompt_tokens"]/ps if ps else 0; dec = u["completion_tokens"]/ds if ds else 0
print(f"     30K prefill: {u['prompt_tokens']:,} tok @ {pf:,.0f} tok/s (ref ~2,200); decode after: {dec:.1f} tok/s (ref ~64)")
results.append(line("30K needle + decode", "DS-4471" in b["choices"][0]["message"]["content"], b, w))
# JSON structured output
b, w = chat([{"role":"user","content":"Return a JSON object with key 'colors': an array of 3 objects each with 'name' and 'hex'."}],
            extra={"response_format": {"type": "json_object"}})
try:
    j = json.loads(b["choices"][0]["message"]["content"]); ok = isinstance(j.get("colors"), list) and len(j["colors"])==3
except Exception: ok = False
results.append(line("JSON mode", ok, b, w))
# knowledge
b, w = chat([{"role":"user","content":"What is the capital of Australia? Answer with just the city name."}])
results.append(line("knowledge (Canberra)", "Canberra" in b["choices"][0]["message"]["content"], b, w))
print(f"\n{sum(results)}/{len(results)} passed")
