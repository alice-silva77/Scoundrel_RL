import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': '#cccccc',
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10, 'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
})

PHASE_BG = {
    0: '#ddeeff',  # blue
    1: '#ddffdd',  # green
    2: '#ffffcc',  # yellow
    3: '#ffe8cc',  # orange
    4: '#ffdddd',  # red-pink
    5: '#eeddff',  # lavender
    6: '#ddf5f5',  # teal
}
PHASE_NAMES = {
    0: 'Ph.0\n32c', 1: 'Ph.1\n38c', 2: 'Ph.2\n40c',
    3: 'Ph.3\n41c', 4: 'Ph.4\n42c',
    5: 'Ph.5\n43c (A♣)', 6: 'Ph.6\n44c (full)',
}


def load(fname):
    with open(fname) as f:
        return json.load(f)


def rolling(x, w=25):
    x = np.array(x, dtype=float)
    mean_ = np.full_like(x, np.nan)
    std_  = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo, hi = max(0, i - w), min(len(x), i + w + 1)
        seg = x[lo:hi]
        valid = seg[~np.isnan(seg)]
        if len(valid) >= 3:
            mean_[i] = valid.mean()
            std_[i]  = valid.std()
    return mean_, std_


def combine_logs(log_a, log_b, offset):
    """Concatenate two logs, shifting log_b timesteps by offset."""
    keys = list(log_a.keys())
    combined = {}
    for k in keys:
        a_vals = np.array(log_a[k], dtype=float)
        b_vals = np.array(log_b[k], dtype=float)
        if k == 'timesteps':
            b_vals = b_vals + offset
        combined[k] = np.concatenate([a_vals, b_vals])
    return combined


def phase_spans(ts, ph):
    spans = []
    prev_p, prev_t = int(ph[0]), 0.0
    for p, t in zip(ph[1:], ts[1:]):
        if int(p) != prev_p:
            spans.append((prev_p, prev_t, t))
            prev_p, prev_t = int(p), t
    spans.append((prev_p, prev_t, ts[-1]))
    return spans


# ── Load and combine ────────────────────────────────────────────────
el  = load('training_log_ent_low.json')
ace = load('training_log_ent_low_ace.json')
offset = el['timesteps'][-1]
log = combine_logs(el, ace, offset)

ts  = log['timesteps'] / 1e6
ph  = log['curriculum_phase'].astype(int)
wr  = log['win_rate']
hp  = log['mean_final_hp']
rms = log['mean_rooms_cleared']

spans = phase_spans(ts, ph)
total_M = ts[-1]

# ── Figure ──────────────────────────────────────────────────────────
COLOR = '#1f77b4'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.suptitle(r'$c_\mathrm{ent}=0.01$ — complete training progression (phases 0–6)',
             fontsize=12, fontweight='bold')

for ax, (ykey, ylab, ymax) in [
    (ax1, (wr,  'Win rate (%)',       62)),
    (ax2, (rms, 'Mean rooms cleared', 15)),
]:
    vals = ykey
    # Phase backgrounds
    for phase, t0, t1 in spans:
        ax.axvspan(t0, t1, color=PHASE_BG.get(phase, '#ffffff'), alpha=0.55, zorder=0)
    # Phase labels at top of each span
    for phase, t0, t1 in spans:
        mid = (t0 + t1) / 2
        ax.text(mid, ymax * 0.97, PHASE_NAMES.get(phase, f'Ph.{phase}'),
                ha='center', va='top', fontsize=7, color='#444444', zorder=5)
    # Phase transition lines
    for i, (phase, t0, t1) in enumerate(spans):
        if i > 0:
            ax.axvline(t0, color='#777777', lw=0.9, ls='--', alpha=0.6, zorder=2)

    m, s = rolling(vals)
    ax.plot(ts, m if ax is ax2 else m * 100,
            color=COLOR, lw=1.9, zorder=4)
    ax.fill_between(ts,
                    (m - s) if ax is ax2 else (m - s) * 100,
                    (m + s) if ax is ax2 else (m + s) * 100,
                    color=COLOR, alpha=0.15, zorder=3)

    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylab, fontsize=9.5)

ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))

# Max rooms reference line
ax2.axhline(14, color='gray', lw=0.9, ls=':', alpha=0.6)
ax2.text(total_M * 0.995, 14.2, 'max (14)', color='gray',
         fontsize=7.5, ha='right', va='bottom', alpha=0.7)

ax2.set_xlabel('Environment steps (millions)', fontsize=9.5)
ax1.set_xlim(0, total_M)

fig.tight_layout()
fig.savefig('fig_entlow_full.png', dpi=150, bbox_inches='tight')
plt.close()
print('fig_entlow_full.png saved')
