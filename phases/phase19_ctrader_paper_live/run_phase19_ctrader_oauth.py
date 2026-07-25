#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from phase19_utils import merge_env, read_env_file


def build_auth_uri(client_id: str, redirect_uri: str, scope: str) -> str:
    q = urlencode(
        {
            'client_id': client_id,
            'redirect_uri': redirect_uri,
            'scope': scope,
            'product': 'web',
        }
    )
    return f'https://id.ctrader.com/my/settings/openapi/grantingaccess/?{q}'


def main():
    ap = argparse.ArgumentParser(description='Phase19 OAuth helper for cTrader Open API.')
    ap.add_argument('--env_file', default='./phases/phase19_ctrader_paper_live/.env', type=str)
    ap.add_argument('--scope', default='trading', choices=['trading', 'accounts'])
    ap.add_argument('--redirect_uri', default='', type=str, help='Override redirect URI. If empty, uses CTRADER_REDIRECT_URI from .env or http://localhost/.')
    ap.add_argument('--auth_code', default='', type=str, help='Authorization code received on redirect URI.')
    ap.add_argument('--print_only', action='store_true', help='Only print auth URL and exit.')
    args = ap.parse_args()

    env_path = Path(args.env_file)
    env = read_env_file(env_path)

    client_id = env.get('CTRADER_CLIENT_ID', '').strip()
    client_secret = env.get('CTRADER_CLIENT_SECRET', '').strip()
    redirect_uri = args.redirect_uri.strip() or env.get('CTRADER_REDIRECT_URI', '').strip() or 'http://localhost/'
    if not client_id or not client_secret:
        raise RuntimeError('Missing CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET in .env')

    auth_uri = build_auth_uri(client_id=client_id, redirect_uri=redirect_uri, scope=args.scope)
    print(f'\nUsing redirect_uri: {redirect_uri}')
    print('\n[1] Open this URL in browser and allow access:')
    print(auth_uri)

    if args.print_only or not args.auth_code.strip():
        print('\n[2] Then run again with: --auth_code <code_from_redirect>')
        return

    # Lazy import so script can print URL even without optional dependency.
    from ctrader_open_api import Auth

    auth = Auth(client_id, client_secret, redirect_uri)
    token = auth.getToken(args.auth_code.strip())

    if token.get('errorCode'):
        raise RuntimeError(f"Token exchange failed: {token.get('errorCode')} | {token.get('description')}")

    expires_in = int(token.get('expiresIn', 0) or 0)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    updates = {
        'CTRADER_SCOPE': args.scope,
        'CTRADER_REDIRECT_URI': redirect_uri,
        'CTRADER_ACCESS_TOKEN': token.get('accessToken', ''),
        'CTRADER_REFRESH_TOKEN': token.get('refreshToken', ''),
        'CTRADER_TOKEN_TYPE': token.get('tokenType', ''),
        'CTRADER_ACCESS_TOKEN_EXPIRES_AT_UTC': expires_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    merge_env(env_path, updates)

    print('\nToken saved to .env successfully.')
    print('Next: run fetch or phase19 daemon and set CTRADER_ACCOUNT_ID if needed.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}', file=sys.stderr)
        raise
