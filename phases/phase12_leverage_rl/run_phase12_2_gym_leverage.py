#!/usr/bin/env python3
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gymnasium import spaces
from sklearn.metrics import f1_score

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


@dataclass
class Config:
    symbol: str = 'GBPUSD'
    train_ratio: float = 0.80

    # Actor-Critic
    hidden_dim: int = 128
    lr: float = 8e-4
    epochs: int = 70
    batch_episodes: int = 40
    gamma: float = 0.98
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    max_steps_episode: int = 256

    # Trading
    cost_pips: float = 1.2


CFG = Config()
ROOT = Path(__file__).resolve().parents[2]
PHASE10_OUT_DIR = ROOT / 'phases/phase10_decision_engine/output_phase10_2_strict_expanding_GBPUSD'
PHASE11_DIR = ROOT / 'phases/phase11_offline_rl'
PHASE12_DIR = ROOT / 'phases/phase12_leverage_rl'

PHASE11_2_SCRIPT = PHASE11_DIR / 'run_phase11_2_offline_rl_reward_input.py'

IN_WINDOWS = PHASE10_OUT_DIR / 'GBPUSD_phase10_2_windows_predictions.csv'
IN_HOLDOUT = PHASE10_OUT_DIR / 'GBPUSD_phase10_2_holdout_predictions.csv'

OUT_DIR = PHASE12_DIR / 'output_phase12_2_gym_leverage_GBPUSD'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TRIALS_CSV = OUT_DIR / 'GBPUSD_phase12_2_trials.csv'
OUT_COMPARE_CSV = OUT_DIR / 'GBPUSD_phase12_2_policy_comparison.csv'
OUT_HOLDOUT_PRED_CSV = OUT_DIR / 'GBPUSD_phase12_2_holdout_predictions.csv'
OUT_HISTORY_CSV = OUT_DIR / 'GBPUSD_phase12_2_best_train_history.csv'
OUT_SUMMARY_JSON = OUT_DIR / 'GBPUSD_phase12_2_summary.json'
OUT_MODEL_PT = OUT_DIR / 'phase12_2_best_actor_critic.pt'
OUT_EQUITY_PNG = OUT_DIR / 'GBPUSD_phase12_2_holdout_equity_compare.png'
OUT_LEV_HIST_PNG = OUT_DIR / 'GBPUSD_phase12_2_holdout_leverage_hist.png'
OUT_TRAIN_PNG = OUT_DIR / 'GBPUSD_phase12_2_training_curve.png'

ACTION_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)

TRIALS = [
    {'name': 't0_neutral', 'risk_penalty': 2e-5, 'lev_bonus': 0.0, 'force_nonzero': False},
    {'name': 't1_forced_light', 'risk_penalty': 2e-5, 'lev_bonus': 2e-5, 'force_nonzero': True},
    {'name': 't2_forced_stronger', 'risk_penalty': 1e-5, 'lev_bonus': 5e-5, 'force_nonzero': True},
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load module {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith('JPY') else 0.0001


def normalize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xn = (x - mu) / sd
    xn = np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)
    return xn.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def normalize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    xn = (x - mu) / sd
    xn = np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)
    return xn.astype(np.float32)


def evaluate_levels(levels: np.ndarray, y_true_action: np.ndarray, future_ret: np.ndarray, cost_ret: float) -> Dict[str, float]:
    action = np.sign(levels).astype(np.int64)
    net = levels * future_ret - np.abs(levels) * cost_ret
    trade = (levels != 0).astype(np.float32)

    out = {
        'acc_action_3': float((action == y_true_action).mean()),
        'f1_macro_action_3': float(f1_score(y_true_action, action, labels=[-1, 0, 1], average='macro', zero_division=0)),
        'avg_net_return': float(np.mean(net)),
        'cum_net_return': float(np.sum(net)),
        'std_net_return': float(np.std(net)),
        'trade_rate': float(np.mean(trade)),
    }
    eps = 1e-12
    out['sharpe_ann'] = float((out['avg_net_return'] / (out['std_net_return'] + eps)) * np.sqrt(24.0 * 252.0))
    traded = trade > 0
    out['win_rate_trades'] = float((net[traded] > 0).mean()) if traded.any() else 0.0
    out['avg_abs_lev'] = float(np.mean(np.abs(levels)))
    out['avg_abs_lev_on_trade'] = float(np.mean(np.abs(levels[traded]))) if traded.any() else 0.0
    return out


def score_from_metrics(m: Dict[str, float]) -> float:
    # slight positive weight on leverage to "force" learning in this probe
    return float(m['avg_net_return']) * 1e4 + 0.10 * float(m['sharpe_ann']) + 0.01 * float(m['avg_abs_lev'])


def build_trajectories(df: pd.DataFrame, x_norm: np.ndarray):
    w = df.copy()
    w['__rowpos__'] = np.arange(len(w), dtype=np.int64)
    w = w.sort_values(['global_segment_id', 'step', 'timestamp']).copy()

    trajs = []
    for seg_id, g in w.groupby('global_segment_id', sort=False):
        rowpos = g['__rowpos__'].to_numpy(dtype=np.int64)
        obs = x_norm[rowpos]
        beh = g['pred_action'].to_numpy(dtype=np.int64)
        y = g['y_action'].to_numpy(dtype=np.int64)
        ret = g['future_return_h'].to_numpy(dtype=np.float32)
        ts = g['timestamp'].to_numpy()

        trajs.append({
            'global_segment_id': seg_id,
            'obs': obs,
            'behavior_action': beh,
            'y_action': y,
            'future_ret': ret,
            'timestamp': ts,
        })

    return trajs


class ForexLeverageEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(
        self,
        trajectories: List[Dict],
        action_mags: np.ndarray,
        cost_ret: float,
        risk_penalty: float,
        lev_bonus: float,
        force_nonzero: bool,
        min_force_lev: float = 0.5,
        random_reset: bool = True,
    ):
        super().__init__()
        self.trajectories = trajectories
        self.action_mags = action_mags
        self.cost_ret = float(cost_ret)
        self.risk_penalty = float(risk_penalty)
        self.lev_bonus = float(lev_bonus)
        self.force_nonzero = bool(force_nonzero)
        self.min_force_lev = float(min_force_lev)
        self.random_reset = bool(random_reset)

        obs_dim = trajectories[0]['obs'].shape[1] + 1  # + behavior sign channel
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(action_mags))

        self._traj_idx = 0
        self._step = 0

    def _get_obs(self):
        tr = self.trajectories[self._traj_idx]
        x = tr['obs'][self._step]
        b = float(tr['behavior_action'][self._step])
        return np.concatenate([x, np.array([b], dtype=np.float32)], axis=0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_reset:
            self._traj_idx = int(self.np_random.integers(0, len(self.trajectories)))
        else:
            self._traj_idx = int((self._traj_idx + 1) % len(self.trajectories))
        self._step = 0
        return self._get_obs(), {}

    def step(self, action):
        tr = self.trajectories[self._traj_idx]

        sign = int(tr['behavior_action'][self._step])
        mag = float(self.action_mags[int(action)])

        if self.force_nonzero and sign != 0 and mag <= 0.0:
            mag = self.min_force_lev

        lev = float(sign) * mag
        ret = float(tr['future_ret'][self._step])

        net = lev * ret - abs(lev) * self.cost_ret
        reward = net - self.risk_penalty * (lev ** 2) + self.lev_bonus * abs(lev) * (1.0 if sign != 0 else 0.0)

        self._step += 1
        done = self._step >= len(tr['obs'])

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        info = {'lev': lev, 'net': net}
        return obs, float(reward), bool(done), False, info


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pi = nn.Linear(hidden_dim, n_actions)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.pi(h), self.v(h).squeeze(-1)


def collect_batch(env: ForexLeverageEnv, model: ActorCritic, cfg: Config, device: torch.device):
    logps = []
    vals = []
    rews = []
    ents = []

    for _ in range(cfg.batch_episodes):
        obs, _ = env.reset()

        ep_logps = []
        ep_vals = []
        ep_rews = []
        ep_ents = []

        for _t in range(cfg.max_steps_episode):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, v = model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()

            next_obs, rew, done, _trunc, _info = env.step(int(act.item()))

            ep_logps.append(dist.log_prob(act).squeeze(0))
            ep_vals.append(v.squeeze(0))
            ep_rews.append(float(rew))
            ep_ents.append(dist.entropy().squeeze(0))

            obs = next_obs
            if done:
                break

        # discounted returns
        g = 0.0
        ep_ret = []
        for r in reversed(ep_rews):
            g = r + cfg.gamma * g
            ep_ret.append(g)
        ep_ret = list(reversed(ep_ret))

        logps.extend(ep_logps)
        vals.extend(ep_vals)
        ents.extend(ep_ents)
        rews.extend(ep_ret)

    returns = torch.tensor(rews, dtype=torch.float32, device=device)
    logps = torch.stack(logps)
    vals = torch.stack(vals)
    ents = torch.stack(ents)
    return logps, vals, returns, ents


def train_actor_critic(env_train: ForexLeverageEnv, env_val: ForexLeverageEnv, cfg: Config, device: torch.device):
    in_dim = env_train.observation_space.shape[0]
    model = ActorCritic(in_dim=in_dim, hidden_dim=cfg.hidden_dim, n_actions=env_train.action_space.n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    hist = []
    best_state = None
    best_score = -np.inf

    for ep in range(1, cfg.epochs + 1):
        model.train()
        logps, vals, rets, ents = collect_batch(env_train, model, cfg, device)

        adv = rets - vals.detach()
        pol_loss = -(logps * adv).mean()
        val_loss = nn.functional.mse_loss(vals, rets)
        ent_loss = -ents.mean()

        loss = pol_loss + cfg.value_coef * val_loss + cfg.entropy_coef * ent_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()

        # quick val metric every epoch
        m_val, _ = evaluate_policy(env_val, model, cfg, device)
        score = score_from_metrics(m_val)

        row = {
            'epoch': ep,
            'train_loss': float(loss.detach().cpu()),
            'policy_loss': float(pol_loss.detach().cpu()),
            'value_loss': float(val_loss.detach().cpu()),
            'entropy': float(ents.mean().detach().cpu()),
            'val_score': float(score),
            'val_avg_net_return': float(m_val['avg_net_return']),
            'val_sharpe_ann': float(m_val['sharpe_ann']),
            'val_avg_abs_lev': float(m_val['avg_abs_lev']),
        }
        hist.append(row)

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(hist)


def evaluate_policy(env: ForexLeverageEnv, model: ActorCritic, cfg: Config, device: torch.device):
    model.eval()

    all_levels = []
    all_y = []
    all_ret = []

    with torch.no_grad():
        for tr in env.trajectories:
            obs_seq = tr['obs']
            b_seq = tr['behavior_action']
            y_seq = tr['y_action']
            r_seq = tr['future_ret']

            levels = []
            for i in range(len(obs_seq)):
                obs = np.concatenate([obs_seq[i], np.array([float(b_seq[i])], dtype=np.float32)], axis=0)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _v = model(obs_t)
                act = int(torch.argmax(logits, dim=1).item())

                sign = int(b_seq[i])
                mag = float(ACTION_MAGS[act])
                if env.force_nonzero and sign != 0 and mag <= 0.0:
                    mag = env.min_force_lev
                lev = float(sign) * mag
                levels.append(lev)

            levels = np.asarray(levels, dtype=np.float32)
            all_levels.append(levels)
            all_y.append(y_seq.astype(np.int64))
            all_ret.append(r_seq.astype(np.float32))

    lev_all = np.concatenate(all_levels)
    y_all = np.concatenate(all_y)
    ret_all = np.concatenate(all_ret)

    metrics = evaluate_levels(lev_all, y_all, ret_all, cost_ret=env.cost_ret)
    return metrics, lev_all


def main():
    p112 = load_module(PHASE11_2_SCRIPT, name='phase11_2_for_phase12_2')

    w = pd.read_csv(IN_WINDOWS)
    h = pd.read_csv(IN_HOLDOUT)
    w['timestamp'] = pd.to_datetime(w['timestamp'], errors='coerce')
    h['timestamp'] = pd.to_datetime(h['timestamp'], errors='coerce')

    w_feat, feature_cols = p112.build_features(w)
    h_feat, _ = p112.build_features(h)

    win = w_feat.sort_values('timestamp').reset_index(drop=True)
    hold = h_feat.sort_values('timestamp').reset_index(drop=True)

    cut = int(len(win) * CFG.train_ratio)
    df_tr = win.iloc[:cut].copy()
    df_va = win.iloc[cut:].copy()
    df_ho = hold.copy()

    x_tr_raw = df_tr[feature_cols].to_numpy(dtype=np.float32)
    x_va_raw = df_va[feature_cols].to_numpy(dtype=np.float32)
    x_ho_raw = df_ho[feature_cols].to_numpy(dtype=np.float32)

    x_tr, mu, sd = normalize_fit(x_tr_raw)
    x_va = normalize_apply(x_va_raw, mu, sd)
    x_ho = normalize_apply(x_ho_raw, mu, sd)

    tr_traj = build_trajectories(df_tr, x_tr)
    va_traj = build_trajectories(df_va, x_va)
    ho_traj = build_trajectories(df_ho, x_ho)

    cost_ret = CFG.cost_pips * pip_size_for_symbol(CFG.symbol)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # baselines
    b_ho = df_ho['pred_action'].to_numpy(dtype=np.float32)
    y_ho = df_ho['y_action'].to_numpy(dtype=np.int64)
    r_ho = df_ho['future_return_h'].to_numpy(dtype=np.float32)
    b_va = df_va['pred_action'].to_numpy(dtype=np.float32)
    y_va = df_va['y_action'].to_numpy(dtype=np.int64)
    r_va = df_va['future_return_h'].to_numpy(dtype=np.float32)

    m_b1_va = evaluate_levels(b_va, y_va, r_va, cost_ret)
    m_b2_va = evaluate_levels(2.0 * b_va, y_va, r_va, cost_ret)
    m_b1_ho = evaluate_levels(b_ho, y_ho, r_ho, cost_ret)
    m_b2_ho = evaluate_levels(2.0 * b_ho, y_ho, r_ho, cost_ret)

    trial_rows = []
    best = None
    best_score = -np.inf

    for t in TRIALS:
        env_train = ForexLeverageEnv(
            trajectories=tr_traj,
            action_mags=ACTION_MAGS,
            cost_ret=cost_ret,
            risk_penalty=float(t['risk_penalty']),
            lev_bonus=float(t['lev_bonus']),
            force_nonzero=bool(t['force_nonzero']),
            random_reset=True,
        )
        env_val = ForexLeverageEnv(
            trajectories=va_traj,
            action_mags=ACTION_MAGS,
            cost_ret=cost_ret,
            risk_penalty=float(t['risk_penalty']),
            lev_bonus=float(t['lev_bonus']),
            force_nonzero=bool(t['force_nonzero']),
            random_reset=False,
        )
        env_hold = ForexLeverageEnv(
            trajectories=ho_traj,
            action_mags=ACTION_MAGS,
            cost_ret=cost_ret,
            risk_penalty=float(t['risk_penalty']),
            lev_bonus=float(t['lev_bonus']),
            force_nonzero=bool(t['force_nonzero']),
            random_reset=False,
        )

        model, hist_df = train_actor_critic(env_train, env_val, CFG, device)
        m_q_va, lev_va = evaluate_policy(env_val, model, CFG, device)
        m_q_ho, lev_ho = evaluate_policy(env_hold, model, CFG, device)

        row = {
            'trial': t['name'],
            'risk_penalty': float(t['risk_penalty']),
            'lev_bonus': float(t['lev_bonus']),
            'force_nonzero': bool(t['force_nonzero']),
            'val_score_q': float(score_from_metrics(m_q_va)),
            'val_avg_q': float(m_q_va['avg_net_return']),
            'val_sharpe_q': float(m_q_va['sharpe_ann']),
            'val_avg_abs_lev_q': float(m_q_va['avg_abs_lev']),
            'holdout_cum_q': float(m_q_ho['cum_net_return']),
            'holdout_cum_behavior_1x': float(m_b1_ho['cum_net_return']),
            'holdout_cum_behavior_2x': float(m_b2_ho['cum_net_return']),
        }
        trial_rows.append(row)

        if row['val_score_q'] > best_score:
            best_score = row['val_score_q']
            best = {
                'trial': dict(t),
                'model': model,
                'hist': hist_df,
                'm_q_va': m_q_va,
                'm_q_ho': m_q_ho,
                'lev_ho': lev_ho,
                'mu': mu,
                'sd': sd,
                'feature_cols': feature_cols,
            }

    trials_df = pd.DataFrame(trial_rows).sort_values('val_score_q', ascending=False)
    trials_df.to_csv(OUT_TRIALS_CSV, index=False)

    # Save model
    torch.save(
        {
            'state_dict': best['model'].state_dict(),
            'feature_cols': best['feature_cols'],
            'mu': best['mu'],
            'sd': best['sd'],
            'action_mags': ACTION_MAGS.tolist(),
            'best_trial': best['trial'],
            'config': asdict(CFG),
        },
        OUT_MODEL_PT,
    )

    cmp_rows = [
        {'split': 'validation', 'policy': 'behavior_1x', **m_b1_va},
        {'split': 'validation', 'policy': 'behavior_2x', **m_b2_va},
        {'split': 'validation', 'policy': 'gym_q_leverage', **best['m_q_va']},
        {'split': 'holdout', 'policy': 'behavior_1x', **m_b1_ho},
        {'split': 'holdout', 'policy': 'behavior_2x', **m_b2_ho},
        {'split': 'holdout', 'policy': 'gym_q_leverage', **best['m_q_ho']},
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT_COMPARE_CSV, index=False)

    pred = df_ho[['global_segment_id', 'timestamp', 'step', 'y_action', 'future_return_h']].copy()
    ret = pred['future_return_h'].to_numpy(dtype=np.float32)

    pred['lev_behavior_1x'] = b_ho
    pred['lev_behavior_2x'] = 2.0 * b_ho
    pred['lev_gym_q_leverage'] = best['lev_ho']

    for nm in ['behavior_1x', 'behavior_2x', 'gym_q_leverage']:
        lev = pred[f'lev_{nm}'].to_numpy(dtype=np.float32)
        pred[f'action_{nm}'] = np.sign(lev).astype(np.int64)
        pred[f'net_{nm}'] = lev * ret - np.abs(lev) * cost_ret
        pred[f'cum_{nm}'] = pred[f'net_{nm}'].cumsum()

    pred.to_csv(OUT_HOLDOUT_PRED_CSV, index=False)

    # plots
    plt.figure(figsize=(11, 4))
    for nm in ['behavior_1x', 'behavior_2x', 'gym_q_leverage']:
        plt.plot(pred['timestamp'], pred[f'cum_{nm}'], label=nm)
    plt.title('Phase12.2 holdout cumulative net return (gym leverage RL)')
    plt.xlabel('timestamp')
    plt.ylabel('cumulative return')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_EQUITY_PNG, dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(best['lev_ho'], bins=[-2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25])
    plt.title('Phase12.2 holdout leverage distribution (gym_q_leverage)')
    plt.xlabel('leverage level')
    plt.ylabel('count')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_LEV_HIST_PNG, dpi=140)
    plt.close()

    best['hist'].to_csv(OUT_HISTORY_CSV, index=False)
    plt.figure(figsize=(10, 4))
    plt.plot(best['hist']['epoch'], best['hist']['val_score'])
    plt.title('Phase12.2 training curve (val score)')
    plt.xlabel('epoch')
    plt.ylabel('val score')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_TRAIN_PNG, dpi=140)
    plt.close()

    summary = {
        'symbol': CFG.symbol,
        'model_tag': 'phase12_2_gym_leverage_rl',
        'description': 'Gym actor-critic RL for leverage magnitude with behavior-direction anchor',
        'action_mags': ACTION_MAGS.tolist(),
        'cost_pips': CFG.cost_pips,
        'dataset': {
            'n_windows_total': int(len(win)),
            'n_windows_train': int(len(df_tr)),
            'n_windows_val': int(len(df_va)),
            'n_holdout': int(len(df_ho)),
            'windows_date_min': str(win['timestamp'].min()),
            'windows_date_max': str(win['timestamp'].max()),
            'holdout_date_min': str(df_ho['timestamp'].min()),
            'holdout_date_max': str(df_ho['timestamp'].max()),
        },
        'best_trial': best['trial'],
        'best_trial_row': trials_df.head(1).to_dict(orient='records')[0],
        'validation_metrics': {
            'behavior_1x': m_b1_va,
            'behavior_2x': m_b2_va,
            'gym_q_leverage': best['m_q_va'],
        },
        'holdout_metrics': {
            'behavior_1x': m_b1_ho,
            'behavior_2x': m_b2_ho,
            'gym_q_leverage': best['m_q_ho'],
        },
        'outputs': {
            'trials_csv': str(OUT_TRIALS_CSV),
            'compare_csv': str(OUT_COMPARE_CSV),
            'holdout_predictions_csv': str(OUT_HOLDOUT_PRED_CSV),
            'history_csv': str(OUT_HISTORY_CSV),
            'model_pt': str(OUT_MODEL_PT),
            'equity_png': str(OUT_EQUITY_PNG),
            'lev_hist_png': str(OUT_LEV_HIST_PNG),
            'train_png': str(OUT_TRAIN_PNG),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print('Saved summary:', OUT_SUMMARY_JSON)
    print('Best trial:', best['trial'])
    print('Holdout metrics:', summary['holdout_metrics'])


if __name__ == '__main__':
    main()
