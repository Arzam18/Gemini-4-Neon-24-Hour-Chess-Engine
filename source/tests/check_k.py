import numpy as np, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'source/data/gen3a.bin'
dt = np.dtype([('occ','<u8'),('pieces','16u1'),('stm','u1'),('wdl','u1'),('score','<i2'),('ply','<u2'),('res','<u2')])
data = np.fromfile(path, dtype=dt, count=2000000)
wdl = data['wdl'].astype(np.float64) * 0.5
sc  = data['score'].astype(np.float64)
print(f"File: {path} (sample {len(data)} pos)")
print(f"WDL: W={(data['wdl']==2).mean():.3f} D={(data['wdl']==1).mean():.3f} L={(data['wdl']==0).mean():.3f}")
print(f"Score: mean={sc.mean():.1f}, std={sc.std():.1f}, p90={np.percentile(np.abs(sc),90):.0f}, |sc|<50={(np.abs(sc)<50).mean():.3f}")
best_k, best_mse = 400, 1e9
for K in range(120, 520, 20):
    p = 1.0 / (1.0 + np.exp(-np.clip(sc / K, -20, 20)))
    mse = np.mean((p - wdl) ** 2)
    if mse < best_mse:
        best_mse, best_k = mse, K
    if K in [140, 180, 200, 220, 240, 260, 280, 300, 340, 400, 460]:
        print(f"  K={K:3d}: WDL-MSE={mse:.6f}")
print(f"BEST K = {best_k} (MSE={best_mse:.6f})")
