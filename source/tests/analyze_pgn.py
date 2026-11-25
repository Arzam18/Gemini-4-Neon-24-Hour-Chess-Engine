import sys, re

pgn_path = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else "v2"

with open(pgn_path) as f:
    pgn = f.read()

games = [g for g in pgn.split('[Event ') if g.strip()]
my_d, opp_d = [], []
my_zeros_when_losing = 0
my_low_depth = 0

for g in games:
    is_w = f'[White "{target}"]' in g
    comments = re.findall(r'\{([^{}]+)\}', g)
    ply = 0
    prev_score = None
    for c in comments:
        if c == 'book':
            ply += 1
            continue
        m = re.search(r'([+-]?\d+\.\d+|[+-]?M\d+)/(\d+)\s+([0-9.]+)s', c)
        if m:
            sc_str, d, t = m.group(1), int(m.group(2)), float(m.group(3))
            is_mine = ((ply % 2 == 0) == is_w)
            if is_mine:
                my_d.append((d, t))
                if d <= 10 and t > 0.15:
                    my_low_depth += 1
                if sc_str == '0.00' and prev_score is not None and abs(prev_score) >= 2.0:
                    my_zeros_when_losing += 1
            else:
                opp_d.append((d, t))
            if 'M' not in sc_str:
                prev_score = float(sc_str)
            else:
                prev_score = 99.0 if '+' in sc_str else -99.0
        ply += 1

print(f"Games: {len(games)}")
print(f"{target:8s}: avg depth={sum(x[0] for x in my_d)/len(my_d):.2f}, avg time={sum(x[1] for x in my_d)/len(my_d):.3f}s, moves={len(my_d)}")
print(f"{'opp':8s}: avg depth={sum(x[0] for x in opp_d)/len(opp_d):.2f}, avg time={sum(x[1] for x in opp_d)/len(opp_d):.3f}s, moves={len(opp_d)}")
print(f"Low-depth (<=10 with >0.15s) moves by {target}: {my_low_depth}")
print(f"Suspicious 0.00 by {target} when opp >= 2.0: {my_zeros_when_losing}")
