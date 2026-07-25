#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from twisted.internet import defer, reactor

from phase19_utils import now_utc_iso, read_env_file, safe_json_write


@dataclass
class FetchConfig:
    symbol: str
    env_file: str
    output: str
    account_id: int
    lookback_hours: int
    count: int
    demo: bool
    require_demo_account: bool
    timeout_sec: int


def _err_text(msg) -> str:
    msg = _decode_msg(msg)
    code = getattr(msg, 'errorCode', None)
    desc = getattr(msg, 'description', None)
    return f'errorCode={code} description={desc}'


def _is_error_msg(msg) -> bool:
    msg = _decode_msg(msg)
    return hasattr(msg, 'errorCode') and msg.__class__.__name__ == 'ProtoOAErrorRes'


def _decode_msg(msg):
    # ctrader_open_api returns ProtoMessage envelopes; extract protobuf payload.
    if msg.__class__.__name__ == 'ProtoMessage':
        try:
            from ctrader_open_api import Protobuf

            return Protobuf.extract(msg)
        except Exception:
            return msg
    return msg


def _pick_account(account_list_msg, preferred_account_id: int, require_demo: bool) -> int:
    accounts = list(getattr(account_list_msg, 'ctidTraderAccount', []))
    if not accounts:
        raise RuntimeError('No trading accounts returned by access token')

    if preferred_account_id > 0:
        for a in accounts:
            if int(a.ctidTraderAccountId) == int(preferred_account_id):
                if require_demo and bool(a.isLive):
                    raise RuntimeError(f'Configured account_id={preferred_account_id} is LIVE but demo required')
                return int(a.ctidTraderAccountId)
        raise RuntimeError(f'Configured account_id={preferred_account_id} not found in token account list')

    # Auto-pick first demo account if requested; otherwise first available.
    if require_demo:
        for a in accounts:
            if not bool(a.isLive):
                return int(a.ctidTraderAccountId)
        raise RuntimeError('No DEMO account found in account list (all are live)')

    return int(accounts[0].ctidTraderAccountId)


def _trendbars_to_df(trendbars_msg, digits: int) -> pd.DataFrame:
    return _trendbars_list_to_df(list(getattr(trendbars_msg, 'trendbar', [])), digits=digits)


def _trendbars_list_to_df(trendbars: list, digits: int) -> pd.DataFrame:
    rows = []
    for tb in trendbars:
        ts = datetime.fromtimestamp(int(tb.utcTimestampInMinutes) * 60, tz=timezone.utc)
        low_rel = int(tb.low)
        open_rel = low_rel + int(tb.deltaOpen)
        close_rel = low_rel + int(tb.deltaClose)
        high_rel = low_rel + int(tb.deltaHigh)

        rows.append(
            {
                'timestamp': ts,
                'open': round(open_rel / 100000.0, digits),
                'high': round(high_rel / 100000.0, digits),
                'low': round(low_rel / 100000.0, digits),
                'close': round(close_rel / 100000.0, digits),
                'volume': float(tb.volume),
            }
        )

    if not rows:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(rows).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def _write_output(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in ['.parquet', '.pq']:
        df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() in ['.csv', '.txt']:
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f'Unsupported output extension: {output_path.suffix}')


def main():
    ap = argparse.ArgumentParser(description='Phase19 - Fetch M1 bars once from cTrader Open API (demo/live).')
    ap.add_argument('--symbol', default='GBPUSD', type=str)
    ap.add_argument('--env_file', default='./phases/phase19_ctrader_paper_live/.env', type=str)
    ap.add_argument('--output', required=True, type=str, help='Output file (.parquet or .csv).')
    ap.add_argument('--account_id', default=0, type=int, help='Optional ctidTraderAccountId. 0=auto-pick.')
    ap.add_argument('--lookback_hours', default=24 * 120, type=int)
    ap.add_argument('--count', default=0, type=int, help='Optional trendbars count hint (0=auto).')
    ap.add_argument('--demo', default=1, type=int, choices=[0, 1])
    ap.add_argument('--require_demo_account', default=1, type=int, choices=[0, 1])
    ap.add_argument('--timeout_sec', default=25, type=int)
    args = ap.parse_args()

    cfg = FetchConfig(
        symbol=str(args.symbol).upper(),
        env_file=str(args.env_file),
        output=str(args.output),
        account_id=int(args.account_id),
        lookback_hours=int(args.lookback_hours),
        count=int(args.count),
        demo=bool(int(args.demo)),
        require_demo_account=bool(int(args.require_demo_account)),
        timeout_sec=int(args.timeout_sec),
    )

    env = read_env_file(Path(cfg.env_file))
    client_id = env.get('CTRADER_CLIENT_ID', '').strip()
    client_secret = env.get('CTRADER_CLIENT_SECRET', '').strip()
    access_token = env.get('CTRADER_ACCESS_TOKEN', '').strip()

    if not client_id or not client_secret:
        raise RuntimeError('Missing CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET in .env')
    if not access_token:
        raise RuntimeError('Missing CTRADER_ACCESS_TOKEN in .env. Run OAuth helper first.')

    from ctrader_open_api import Client, EndPoints, TcpProtocol
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAAccountAuthReq,
        ProtoOAApplicationAuthReq,
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAGetTrendbarsReq,
        ProtoOASymbolByIdReq,
        ProtoOASymbolsListReq,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod

    host = EndPoints.PROTOBUF_DEMO_HOST if cfg.demo else EndPoints.PROTOBUF_LIVE_HOST
    client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)

    result = {'ok': False, 'error': None, 'rows': 0, 'account_id': None, 'symbol_id': None}

    @defer.inlineCallbacks
    def flow():
        try:
            client.startService()
            yield client.whenConnected(failAfterFailures=1)

            # Step 1: App auth
            app_req = ProtoOAApplicationAuthReq()
            app_req.clientId = client_id
            app_req.clientSecret = client_secret
            app_res = yield client.send(app_req, responseTimeoutInSeconds=cfg.timeout_sec)
            app_res = _decode_msg(app_res)
            if _is_error_msg(app_res):
                raise RuntimeError(f'Application auth failed: {_err_text(app_res)}')

            # Step 2: resolve account id (explicit account_id bypasses account-list call).
            if int(cfg.account_id) > 0:
                account_id = int(cfg.account_id)
            else:
                acc_req = ProtoOAGetAccountListByAccessTokenReq()
                acc_req.accessToken = access_token
                acc_res = yield client.send(acc_req, responseTimeoutInSeconds=cfg.timeout_sec)
                acc_res = _decode_msg(acc_res)
                if _is_error_msg(acc_res):
                    raise RuntimeError(f'Account list failed: {_err_text(acc_res)}')
                account_id = _pick_account(
                    account_list_msg=acc_res,
                    preferred_account_id=cfg.account_id,
                    require_demo=cfg.require_demo_account,
                )
            result['account_id'] = int(account_id)

            # Step 3: account auth
            acc_auth = ProtoOAAccountAuthReq()
            acc_auth.ctidTraderAccountId = int(account_id)
            acc_auth.accessToken = access_token
            acc_auth_res = yield client.send(acc_auth, responseTimeoutInSeconds=cfg.timeout_sec)
            acc_auth_res = _decode_msg(acc_auth_res)
            if _is_error_msg(acc_auth_res):
                raise RuntimeError(f'Account auth failed: {_err_text(acc_auth_res)}')

            # Step 4: symbol list and symbol id
            sym_req = ProtoOASymbolsListReq()
            sym_req.ctidTraderAccountId = int(account_id)
            sym_req.includeArchivedSymbols = False
            sym_res = yield client.send(sym_req, responseTimeoutInSeconds=cfg.timeout_sec)
            sym_res = _decode_msg(sym_res)
            if _is_error_msg(sym_res):
                raise RuntimeError(f'Symbol list failed: {_err_text(sym_res)}')

            symbol_id: Optional[int] = None
            for s in list(getattr(sym_res, 'symbol', [])):
                if str(getattr(s, 'symbolName', '')).upper() == cfg.symbol:
                    symbol_id = int(s.symbolId)
                    break
            if symbol_id is None:
                raise RuntimeError(f'Symbol {cfg.symbol} not found in account symbol list')
            result['symbol_id'] = int(symbol_id)

            # Step 5: symbol details for digits
            sb_req = ProtoOASymbolByIdReq()
            sb_req.ctidTraderAccountId = int(account_id)
            # Some protobuf versions expose symbolId as a repeated field.
            try:
                sb_req.symbolId.append(int(symbol_id))
            except Exception:
                sb_req.symbolId = int(symbol_id)
            sb_res = yield client.send(sb_req, responseTimeoutInSeconds=cfg.timeout_sec)
            sb_res = _decode_msg(sb_res)
            if _is_error_msg(sb_res):
                raise RuntimeError(f'SymbolById failed: {_err_text(sb_res)}')
            sb_symbols = list(getattr(sb_res, 'symbol', []))
            if not sb_symbols:
                sb_symbols = list(getattr(sb_res, 'archivedSymbol', []))
            if not sb_symbols:
                raise RuntimeError('SymbolById returned no symbols')
            digits = int(getattr(sb_symbols[0], 'digits', 5) or 5)

            # Step 6: get trendbars (M1) with backward pagination.
            to_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            from_ts_ms = int((datetime.now(timezone.utc) - timedelta(hours=int(cfg.lookback_hours))).timestamp() * 1000)
            chunk_count = int(cfg.count) if int(cfg.count) > 0 else 10000
            cur_to_ms = int(to_ts_ms)
            all_bars = []
            n_pages = 0
            while True:
                n_pages += 1
                tb_req = ProtoOAGetTrendbarsReq()
                tb_req.ctidTraderAccountId = int(account_id)
                tb_req.symbolId = int(symbol_id)
                tb_req.period = ProtoOATrendbarPeriod.M1
                tb_req.fromTimestamp = int(from_ts_ms)
                tb_req.toTimestamp = int(cur_to_ms)
                tb_req.count = int(chunk_count)

                tb_res = yield client.send(tb_req, responseTimeoutInSeconds=max(cfg.timeout_sec, 40))
                tb_res = _decode_msg(tb_res)
                if _is_error_msg(tb_res):
                    raise RuntimeError(f'GetTrendbars failed: {_err_text(tb_res)}')

                bars = list(getattr(tb_res, 'trendbar', []))
                if not bars:
                    break

                all_bars.extend(bars)
                min_utc_min = min(int(tb.utcTimestampInMinutes) for tb in bars)
                min_utc_ms = int(min_utc_min * 60 * 1000)

                if min_utc_ms <= int(from_ts_ms + 60_000):
                    break

                next_to_ms = int(min_utc_ms - 60_000)
                if next_to_ms >= cur_to_ms:
                    break
                cur_to_ms = next_to_ms

                # Safety against malformed paging loops.
                if n_pages >= 256:
                    break

            df = _trendbars_list_to_df(all_bars, digits=digits)
            if len(df) == 0:
                raise RuntimeError('No trendbars returned')

            out_path = Path(cfg.output)
            _write_output(df, out_path)

            meta = {
                'status': 'SUCCESS',
                'created_at_utc': now_utc_iso(),
                'symbol': cfg.symbol,
                'demo': bool(cfg.demo),
                'account_id': int(account_id),
                'symbol_id': int(symbol_id),
                'rows': int(len(df)),
                'pages': int(n_pages),
                'chunk_count': int(chunk_count),
                'timestamp_min': str(df['timestamp'].min()),
                'timestamp_max': str(df['timestamp'].max()),
                'output_file': str(out_path.resolve()),
            }
            safe_json_write(out_path.with_suffix(out_path.suffix + '.meta.json'), meta)

            result['ok'] = True
            result['rows'] = int(len(df))
        except Exception as e:
            result['error'] = f'{type(e).__name__}: {e}'
        finally:
            try:
                client.stopService()
            except Exception:
                pass
            reactor.stop()

    reactor.callWhenRunning(flow)
    reactor.run()

    if not result['ok']:
        raise RuntimeError(result.get('error') or 'Unknown fetch error')

    print(
        {
            'status': 'SUCCESS',
            'symbol': cfg.symbol,
            'rows': result['rows'],
            'account_id': result['account_id'],
            'symbol_id': result['symbol_id'],
            'output': str(Path(cfg.output).resolve()),
        }
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}', file=sys.stderr)
        raise
