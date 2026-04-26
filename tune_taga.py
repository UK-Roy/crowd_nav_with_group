"""
Bayesian optimization of TAGA parameters using Optuna.

Usage:
    pip install optuna
    python tune_taga.py --trials 30 --seeds 30

Optimizes TAGA hyperparameters across ORCA / Social Force / intention_rl.
Objective: maximize TAGA+variant SR, minimize GCR, penalize CR increase above base.

Artifacts:
    results/optuna_best.json         — best config found
    results/optuna_study.sqlite      — full trial history (resume with same file)
    results/optuna_log.txt           — per-trial metrics table
"""
import argparse, json, re, shutil, signal, subprocess, sys
from pathlib import Path

try:
    import optuna
except ImportError:
    print("ERROR: Optuna not installed. Run: pip install optuna", file=sys.stderr)
    sys.exit(1)

ROOT          = Path(__file__).resolve().parent
CONFIG_PATH   = ROOT / 'crowd_nav' / 'configs' / 'config.py'
CONFIG_BACKUP = ROOT / 'crowd_nav' / 'configs' / 'config.py.optuna_bak'
RESULTS_DIR   = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

POLICIES_BASE = ['orca', 'social_force', 'intention_rl']
POLICIES_TAGA = [p + '+taga' for p in POLICIES_BASE]
ALL_POLICIES  = POLICIES_BASE + POLICIES_TAGA

# ── config patching ───────────────────────────────────────────────────────────
def backup_config():
    if not CONFIG_BACKUP.exists():
        shutil.copy(CONFIG_PATH, CONFIG_BACKUP)

def restore_config():
    if CONFIG_BACKUP.exists():
        shutil.copy(CONFIG_BACKUP, CONFIG_PATH)

def patch_config(params):
    """In-place replace taga.<key> = <value> lines in config.py."""
    content = CONFIG_PATH.read_text()
    for key, value in params.items():
        pattern = re.compile(rf'(taga\.{key}\s*=\s*)[\d.]+')
        new_line = rf'\g<1>{value:.4f}'
        content, n = pattern.subn(new_line, content, count=1)
        if n == 0:
            print(f"WARN: could not patch taga.{key}", file=sys.stderr)
    CONFIG_PATH.write_text(content)

# ── output parsing ────────────────────────────────────────────────────────────
def parse_metrics(stdout):
    """Parse the summary table at the end of record_comparison.py output.
    Returns {policy_label: {sr, cr, tr, steps, gcr, reward}}.
    """
    results = {}
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('Policy') and 'SR' in line and 'GCR' in line:
            j = i + 2
            while j < len(lines):
                row = lines[j].split()
                if len(row) < 7:
                    break
                try:
                    name   = row[0]
                    sr     = float(row[1])
                    cr     = float(row[2])
                    tr     = float(row[3])
                    steps  = float(row[4])
                    gcr    = float(row[5])
                    reward = float(row[6])
                    results[name] = dict(sr=sr, cr=cr, tr=tr, steps=steps, gcr=gcr, reward=reward)
                except (ValueError, IndexError):
                    break
                j += 1
            break
    return results

# ── run one trial ─────────────────────────────────────────────────────────────
def run_comparison(n_seeds):
    """Run record_comparison.py on the current config; return parsed metrics."""
    seeds = ','.join(str(s) for s in range(n_seeds))
    cmd = [
        sys.executable, '-u', str(ROOT / 'record_comparison.py'),
        '--policies', ','.join(ALL_POLICIES),
        '--seeds', seeds,
        '--no-video',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        print(f"record_comparison.py failed (rc={proc.returncode})")
        print(proc.stderr[-2000:])
        return None
    return parse_metrics(proc.stdout)

# ── objective ─────────────────────────────────────────────────────────────────
def compute_score(metrics):
    """Higher is better. Rewards high SR and low GCR on TAGA variants;
    penalizes CR increase above each base policy's CR."""
    if not metrics:
        return -1e6, []
    score = 0.0
    details = []
    for base, taga in zip(POLICIES_BASE, POLICIES_TAGA):
        if base not in metrics or taga not in metrics:
            continue
        m_b, m_t = metrics[base], metrics[taga]
        sr_term  =  1.0 * m_t['sr']                          # higher SR better
        gcr_term = -30.0 * m_t['gcr']                        # lower GCR better
        cr_pen   = -5.0 * max(0.0, m_t['cr'] - m_b['cr'])    # penalty if TAGA CR > base CR
        contrib  = sr_term + gcr_term + cr_pen
        score   += contrib
        details.append((taga, m_t['sr'], m_t['cr'], m_t['gcr'], contrib))
    return score, details

# ── Optuna objective function ─────────────────────────────────────────────────
def objective_factory(n_seeds, log_path):
    def objective(trial):
        params = {
            'intent_lookahead': trial.suggest_float('intent_lookahead', 0.3, 1.5),
            'intent_margin'   : trial.suggest_float('intent_margin',    0.0, 0.5),
            'safe_margin'     : trial.suggest_float('safe_margin',      0.2, 0.8),
            'switch_band'     : trial.suggest_float('switch_band',      0.3, 0.8),
            'goal_threshold'  : trial.suggest_float('goal_threshold',   1.5, 3.5),
            'safety_radius'   : trial.suggest_float('safety_radius',    0.4, 0.7),
            'w_obstacle'      : trial.suggest_float('w_obstacle',       0.3, 0.7),
        }

        patch_config(params)
        metrics = run_comparison(n_seeds)

        if not metrics:
            return -1e6

        score, details = compute_score(metrics)  # always returns tuple

        # Log
        with open(log_path, 'a') as f:
            f.write(f"\n--- Trial {trial.number} (score={score:.3f}) ---\n")
            for k, v in params.items():
                f.write(f"  {k}: {v:.4f}\n")
            for name, sr, cr, gcr, contrib in details:
                f.write(f"  {name}: SR={sr:.3f} CR={cr:.3f} GCR={gcr:.4f} contrib={contrib:+.3f}\n")

        print(f"Trial {trial.number}: score={score:.3f} params={ {k:round(v,3) for k,v in params.items()} }")
        for name, sr, cr, gcr, contrib in details:
            print(f"    {name}: SR={sr:.3f} CR={cr:.3f} GCR={gcr:.4f}")

        return score
    return objective

# ── apply best params to config.py ────────────────────────────────────────────
def apply_best():
    """Read results/optuna_best.json and write those params into config.py."""
    best_path = RESULTS_DIR / 'optuna_best.json'
    if not best_path.exists():
        print(f"ERROR: {best_path} not found — run tuning first.", file=sys.stderr)
        sys.exit(1)
    data = json.loads(best_path.read_text())
    params = data.get('params', data)
    print(f"Applying best params (score={data.get('score', '?')}):")
    for k, v in params.items():
        print(f"  taga.{k} = {v:.4f}")
    patch_config(params)
    print(f"\n✓ config.py updated. Now run your 100-seed benchmark.")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials',  type=int, default=30, help='number of Optuna trials')
    ap.add_argument('--seeds',   type=int, default=30, help='seeds per trial (smaller = faster)')
    ap.add_argument('--study',   default='taga_tuning',
                    help='Optuna study name (resume with same name)')
    ap.add_argument('--apply',   action='store_true',
                    help='apply results/optuna_best.json to config.py and exit')
    args = ap.parse_args()

    if args.apply:
        apply_best()
        return

    # Install signal handlers so ctrl+c still restores config
    def cleanup(*_):
        restore_config()
        sys.exit(1)
    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    backup_config()
    log_path     = RESULTS_DIR / 'optuna_log.txt'
    storage_path = RESULTS_DIR / 'optuna_study.sqlite'
    storage_url  = f'sqlite:///{storage_path}'

    if log_path.exists():
        log_path.unlink()

    try:
        study = optuna.create_study(
            study_name    = args.study,
            storage       = storage_url,
            direction     = 'maximize',
            load_if_exists= True,
            sampler       = optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(objective_factory(args.seeds, log_path), n_trials=args.trials)

        # Persist best
        best_path = RESULTS_DIR / 'optuna_best.json'
        best_path.write_text(json.dumps({
            'score' : study.best_value,
            'params': study.best_params,
        }, indent=2))

        print("\n========== BEST TRIAL ==========")
        print(f"Score: {study.best_value:.3f}")
        print("Params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nSaved to: {best_path}")
        print(f"Full log: {log_path}")
    finally:
        restore_config()

if __name__ == '__main__':
    main()
