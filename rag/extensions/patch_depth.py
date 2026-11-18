p = 'rag/extensions/depth_study.py'
s = open(p).read()
old1 = "        for K in Ks:\n            lat = ["
new1 = ("        for _ in range(50):  # warm up after the encoder so early K values are not penalised\n"
        "            retr.retrieve(vec, k=100)\n"
        "        for K in Ks:\n            lat = [")
old2 = "| {d} | {np.mean(ns):.2f} | "
new2 = ("| {d} | {np.mean(ns):.2f} (median {np.median(ns):.0f}; 10th-90th pct "
        "{np.percentile(ns, 10):.0f}-{np.percentile(ns, 90):.0f}) | ")
assert old1 in s and old2 in s, 'patch targets not found - file differs from the shipped version'
open(p, 'w').write(s.replace(old1, new1, 1).replace(old2, new2, 1))
print('patched')
