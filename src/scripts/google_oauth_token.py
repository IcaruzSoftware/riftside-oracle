"""
One-off local helper: authorise riftside-oracle to use your Google Drive and print
a refresh token for the GOOGLE_OAUTH_REFRESH_TOKEN secret (the copy-to-own-Drive
fallback in oe.py).

Needs an OAuth client of type "Desktop app" (see docs/operations.md). Opens
Google's consent page in your browser, catches the redirect on
http://127.0.0.1:<free port> (PKCE S256 + state check), exchanges the code and
prints only the refresh token. Scope: https://www.googleapis.com/auth/drive
(drive.file cannot read the public OE file, so it cannot copy it).

CLI
---
  python src/scripts/google_oauth_token.py --client-id ID --client-secret SECRET
  (or set GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET)
"""

import argparse
import base64
import hashlib
import http.server
import os
import secrets
import sys
import time
import webbrowser
from urllib.parse import parse_qs, urlencode, urlsplit

import requests

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/drive"
TIMEOUT_S = 300                 # overall wait for the browser redirect


def pkce_pair() -> tuple[str, str]:
    """A PKCE code verifier and its S256 challenge."""
    verifier = secrets.token_urlsafe(64)          # 86 chars, within RFC 7636's 43..128
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def auth_url(client_id: str, redirect_uri: str, challenge: str, state: str) -> str:
    """The consent URL: offline access + forced consent, so Google returns a refresh token."""
    return AUTH_URL + "?" + urlencode({
        "client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code",
        "scope": SCOPE, "access_type": "offline", "prompt": "consent",
        "code_challenge": challenge, "code_challenge_method": "S256", "state": state})


def is_oauth_redirect(path: str) -> bool:
    """True for Google's redirect (state, code or error in the query), not a stray hit like /favicon.ico."""
    return bool({"state", "code", "error"} & parse_qs(urlsplit(path).query).keys())


def parse_callback(path: str, state: str) -> str:
    """The authorization code from the redirect's path + query; raises on a bad state or an error."""
    query = parse_qs(urlsplit(path).query)
    if query.get("state", [""])[0] != state:
        raise RuntimeError("state mismatch in the OAuth redirect; start again")
    if "error" in query:
        raise RuntimeError(f"authorisation refused: {query['error'][0]}")
    if not query.get("code"):
        raise RuntimeError("the OAuth redirect has no code")
    return query["code"][0]


def exchange_code(code: str, client_id: str, client_secret: str, verifier: str,
                  redirect_uri: str) -> str:
    """Exchange the code for tokens and return the refresh token; errors carry no credential."""
    resp = requests.post(TOKEN_URL, data={
        "code": code, "client_id": client_id, "client_secret": client_secret,
        "code_verifier": verifier, "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"}, timeout=30)
    try:
        body = resp.json()
    except ValueError:
        body = None
    body = body if isinstance(body, dict) else {}
    if resp.ok and SCOPE not in str(body.get("scope", "")).split():
        raise RuntimeError("Drive access was not granted; re-run and tick the Google Drive "
                           "checkbox on the consent page")
    if resp.ok and isinstance(body.get("refresh_token"), str):
        return body["refresh_token"]
    if resp.ok:
        raise RuntimeError("token exchange returned no refresh token (remove the app's access "
                           "at https://myaccount.google.com/permissions and run this again)")
    error = body.get("error") if isinstance(body.get("error"), str) else resp.reason
    raise RuntimeError(f"token exchange failed (HTTP {resp.status_code}: {error})")


class _Redirect(http.server.BaseHTTPRequestHandler):
    """Stores the first OAuth redirect path as server.redirect_path; ignores stray requests."""

    def do_GET(self):
        if is_oauth_redirect(self.path) and not self.server.redirect_path:
            self.server.redirect_path = self.path
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Done - you can close this tab and return to the terminal.")

    def log_message(self, *args):       # the request line would show the code
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a Google OAuth refresh token for oe.py")
    parser.add_argument("--client-id", default=os.environ.get("GOOGLE_OAUTH_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"))
    args = parser.parse_args()
    client_id, client_secret = (args.client_id or "").strip(), (args.client_secret or "").strip()
    if not client_id or not client_secret:
        parser.error("set --client-id/--client-secret or GOOGLE_OAUTH_CLIENT_ID/_SECRET")

    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(16)
    with http.server.HTTPServer(("127.0.0.1", 0), _Redirect) as server:   # port 0 = a free port
        server.redirect_path, server.timeout = None, 5    # handle_request returns after 5 s idle
        redirect_uri = f"http://127.0.0.1:{server.server_port}"
        url = auth_url(client_id, redirect_uri, challenge, state)
        print(f"Opening the Google consent page; if no browser opens, visit:\n{url}\n",
              file=sys.stderr)
        webbrowser.open(url)
        deadline = time.monotonic() + TIMEOUT_S
        while not server.redirect_path:
            if time.monotonic() > deadline:
                raise RuntimeError(f"timed out after {TIMEOUT_S // 60} min waiting for the "
                                   f"Google redirect; re-run")
            server.handle_request()
        code = parse_callback(server.redirect_path, state)
    refresh_token = exchange_code(code, client_id, client_secret, verifier, redirect_uri)
    print("Refresh token - store it as the GitHub secret GOOGLE_OAUTH_REFRESH_TOKEN "
          "(never commit or share it):", file=sys.stderr)
    print(refresh_token)


if __name__ == "__main__":
    try:
        main()
    except (requests.exceptions.RequestException, RuntimeError) as exc:
        sys.exit(f"FAILED - {exc}")
