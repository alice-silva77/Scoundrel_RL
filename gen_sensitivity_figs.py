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

PHASE_BG    = {0: '#ddeeff', 1: '#ddffdd', 2: '#ffffcc', 3: '#ffe8cc', 4: '#ffdddd'}
PHASE_NAMES = {0: 'Ph.0 (32c)', 1: 'Ph.1 (38c)', 2: 'Ph.2 (40c)',
               3: 'Ph.3 (41c)', 4: 'Ph.4 (42c)'}


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


def phase_spans(log):
    """Returns list of (phase_int, t_start_M, t_end_M)."""
    ph = np.array(log['curriculum_phase'])
    ts = np.array(log['timesteps']) / 1e6
    spans = []
    prev_p, prev_t = int(ph[0]), 0.0
    for p, t in zip(ph[1:], ts[1:]):
        if int(p) != prev_p:
            spans.append((prev_p, prev_t, t))
            prev_p, prev_t = int(p), t
    spans.append((prev_p, prev_t, ts[-1]))
    return spans


def draw_run(ax, log, color, subtitle, budget_M=55, ymax=60):
    ts    = np.array(log['timesteps']) / 1e6
    wr_m, wr_s = rolling(log['win_rate'])

    spans = phase_spans(log)

    # --- Phase background shading ---
    for phase, t0, t1 in spans:
        ax.axvspan(t0, t1, color=PHASE_BG.get(phase, '#ffffff'), alpha=0.55, zorder=0)

    # --- Phase transition vertical lines + labels at top ---
    for i, (phase, t0, t1) in enumerate(spans):
        if i > 0:
            ax.axvline(t0, color='#777777', lw=0.9, ls='--', alpha=0.6, zorder=2)
        mid = (t0 + t1) / 2
        ax.text(mid, ymax * 0.97, PHASE_NAMES.get(phase, f'Ph.{phase}'),
                ha='center', va='top', fontsize=7, color='#444444', zorder=5)

    # --- Win rate ---
    ax.plot(ts, wr_m * 100, color=color, lw=1.9, zorder=4)
    ax.fill_between(ts,
                    (wr_m - wr_s) * 100,
                    (wr_m + wr_s) * 100,
                    color=color, alpha=0.15, zorder=3)

    ax.set_xlim(0, budget_M)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax.set_ylabel('Win rate (%)', fontsize=9)

    # Run label inside plot
    ax.text(0.99, 0.06, subtitle, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color=color, fontweight='bold')


logs = {k: load(f'training_log_{k}.json') for k in
        ['ent_low', 'ent_med', 'ent_high', 'lr_low', 'lr_high', 'no_kwb', 'no_cc']}

# =======================================================================
# Figure A — Entropy coefficient  (3 stacked rows)
# =======================================================================
BUDGET = 55
YMAX   = 60

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
fig.suptitle('Sensitivity analysis: entropy coefficient',
             fontsize=12, fontweight='bold')

rows = [
    ('ent_low',  '#1f77b4', r'$c_\mathrm{ent}=0.01$  (interrupted at 60.5 %)'),
    ('ent_med',  '#ff7f0e', r'$c_\mathrm{ent}=0.05$  (complete)'),
    ('ent_high', '#d62728', r'$c_\mathrm{ent}=0.20$  (stuck at phase 1 after 40.9 M)'),
]
for ax, (name, col, sub) in zip(axes, rows):
    draw_run(ax, logs[name], col, sub, budget_M=BUDGET, ymax=YMAX)

axes[-1].set_xlabel('Environment steps (millions)', fontsize=9.5)
fig.tight_layout()
fig.savefig('fig_sensitivity_entropy.png', dpi=150, bbox_inches='tight')
plt.close()
print('fig_sensitivity_entropy.png saved')

# =======================================================================
# Figure B — Learning rate  (3 stacked rows)
# =======================================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
fig.suptitle('Sensitivity analysis: learning rate',
             fontsize=12, fontweight='bold')

rows_lr = [
    ('lr_low',  '#1f77b4', r'$\alpha = 10^{-5}$  (complete)'),
    ('ent_med', '#333333', r'$\alpha = 5\times10^{-5}$  (reference, complete)'),
    ('lr_high', '#d62728', r'$\alpha = 2\times10^{-4}$  (complete — stuck at phase 1 for full 55 M)'),
]
for ax, (name, col, sub) in zip(axes, rows_lr):
    draw_run(ax, logs[name], col, sub, budget_M=BUDGET, ymax=YMAX)

axes[-1].set_xlabel('Environment steps (millions)', fontsize=9.5)
fig.tight_layout()
fig.savefig('fig_sensitivity_lr.png', dpi=150, bbox_inches='tight')
plt.close()
print('fig_sensitivity_lr.png saved')

# =======================================================================
# Figure C — Reward shaping and observation ablations  (3 stacked rows)
# =======================================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
fig.suptitle('Sensitivity analysis: reward shaping and observation ablations',
             fontsize=12, fontweight='bold')

rows_abl = [
    ('ent_med', '#333333', r'Reference  ($c_\mathrm{ent}=0.05$, all features enabled)'),
    ('no_kwb',  '#d62728', r'No king weapon bonus  (kwb $= 0$)'),
    ('no_cc',   '#1f77b4', r'No card counting'),
]
for ax, (name, col, sub) in zip(axes, rows_abl):
    draw_run(ax, logs[name], col, sub, budget_M=BUDGET, ymax=YMAX)

axes[-1].set_xlabel('Environment steps (millions)', fontsize=9.5)
fig.tight_layout()
fig.savefig('fig_sensitivity_ablations.png', dpi=150, bbox_inches='tight')
plt.close()
print('fig_sensitivity_ablations.png saved')
print('All done.')
