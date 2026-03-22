data = "2026-03-04 11:06:10,new-data c8,41908,5047,12.0,8851,21.1,13979,33.4,13425,32.0,606,1.4"

_, label, total, good, good_pct, slight, slight_pct, co, co_pct, fp, fp_pct, fn, fn_pct = data.split(",")
total = int(total)

print(f"\n=== Overall ({total} prediction-cases) ===")
for lbl, n, pct in [
    ("Good",           good,   good_pct),
    ("Slightly off",   slight, slight_pct),
    ("Completely off", co,     co_pct),
    ("False positive", fp,     fp_pct),
    ("False negative", fn,     fn_pct),
]:
    print(f"  {lbl:<22} {int(n):>5}  ({float(pct):.1f}%)")
