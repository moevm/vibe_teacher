"""Фейковый Forgejo + LLM для сквозной проверки CLI.

Запуск: python3 tests/fake_server.py <port>
"""

import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PULLS = {
    11: {
        "number": 11,
        "title": "Fix login form",
        "state": "open",
        "draft": False,
        "user": {"login": "alice"},
        "assignees": [{"login": "alice"}],
        "labels": [{"name": "needs-review"}],
        "head": {"sha": "aaa111bbb222", "ref": "fix-login", "repo": {"full_name": "owner/repo"}},
        "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
        "body": "Closes #10",
        "additions": 3,
        "deletions": 1,
    },
    12: {
        "number": 12,
        "title": "Update README",
        "state": "open",
        "draft": False,
        "user": {"login": "bob"},
        "assignees": [{"login": "bob"}],
        "labels": [{"name": "docs"}],
        "head": {"sha": "ccc333ddd444", "ref": "docs", "repo": {"full_name": "owner/repo"}},
        "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
        "body": "",
        "additions": 1,
        "deletions": 0,
    },
    13: {
        "number": 13,
        "title": "WIP: refactor",
        "state": "open",
        "draft": True,
        "user": {"login": "bob"},
        "assignees": [{"login": "bob"}],
        "labels": [{"name": "docs"}, {"name": "wip"}],
        "head": {"sha": "eee555fff666", "ref": "wip", "repo": {"full_name": "owner/repo"}},
        "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
        "body": "draft",
        "additions": 9,
        "deletions": 0,
    },
}

DIFF_11 = (
    "diff --git a/src/login.py b/src/login.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/src/login.py\n"
    "+++ b/src/login.py\n"
    "@@ -1,3 +1,4 @@\n"
    " def login(user):\n"
    "-    return check(user)\n"
    "+    token = check(user)\n"
    "+    return token\n"
)
DIFF_13 = "diff --git a/wip.txt b/wip.txt\n--- /dev/null\n+++ b/wip.txt\n@@ -0,0 +1 @@\n+wip\n"
DIFF_12 = "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1,2 @@\n # Project\n+New line\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    # --- helpers -------------------------------------------------------- #
    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, text, status=200, ctype="text/plain"):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _page(self, items, qs):
        page = int(qs.get("page", ["1"])[0])
        limit = int(qs.get("limit", ["50"])[0])
        start = (page - 1) * limit
        self._json(items[start : start + limit])

    # --- GET ------------------------------------------------------------ #
    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        path = u.path
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("token "):
            return self._json({"message": "unauthorized"}, 401)

        if path == "/api/v1/repos/owner/repo":
            return self._json({"full_name": "owner/repo", "empty": False, "default_branch": "main"})
        if path == "/api/v1/repos/owner/repo/pulls":
            return self._page(list(PULLS.values()), qs)

        for num in PULLS:
            p = f"/api/v1/repos/owner/repo/pulls/{num}"
            if path == p:
                return self._json(PULLS[num])
            if path == f"{p}/commits":
                return self._page([{"sha": PULLS[num]["head"]["sha"], "commit": {"message": "change"}}], qs)
            if path == f"{p}/files":
                if num == 11:
                    files = [
                        {
                            "filename": "src/login.py",
                            "status": "modified",
                            "additions": 2,
                            "deletions": 1,
                            "changes": 3,
                            "patch": "",
                        }
                    ]  # сервер «забыл» патч
                elif num == 12:
                    files = [
                        {
                            "filename": "README.md",
                            "status": "modified",
                            "additions": 1,
                            "deletions": 0,
                            "changes": 1,
                            "patch": "@@ -1 +1,2 @@\n # Project\n+New line\n",
                        }
                    ]
                else:
                    files = [
                        {
                            "filename": "wip.txt",
                            "status": "added",
                            "additions": 1,
                            "deletions": 0,
                            "changes": 1,
                            "patch": "@@ -0,0 +1 @@\n+wip\n",
                        }
                    ]
                return self._page(files, qs)
            if path == f"{p}/reviews":
                return self._page(
                    [
                        {
                            "id": 500 + num,
                            "state": "APPROVED",
                            "user": {"login": "carol"},
                            "submitted_at": "2026-11-10T10:00:00Z",
                            "body": "ok",
                        }
                    ],
                    qs,
                )
            if path == f"{p}/reviews/{500 + num}/comments":
                return self._page(
                    [
                        {
                            "id": 900 + num,
                            "path": "src/login.py",
                            "line": 2,
                            "body": "nit: rename",
                            "diff_hunk": "@@ -1,3 +1,4 @@",
                        }
                    ],
                    qs,
                )
            if path == f"{p}.diff":
                return self._text({11: DIFF_11, 12: DIFF_12, 13: DIFF_13}[num])

        if path == "/api/v1/repos/owner/repo/issues/11/comments":
            return self._page(
                [{"id": 1, "user": {"login": "dave"}, "body": "Please rebase", "created_at": "2026-11-11T09:00:00Z"}],
                qs,
            )
        if path == "/api/v1/repos/owner/repo/issues/12/comments":
            return self._page([], qs)
        if path == "/api/v1/repos/owner/repo/issues/13/comments":
            return self._page([], qs)
        if path == "/api/v1/repos/owner/repo/issues/11/timeline":
            return self._page(
                [
                    {
                        "id": 1,
                        "type": "comment",
                        "user": {"login": "dave"},
                        "body": "Please rebase",
                        "created_at": "2026-11-11T09:00:00Z",
                    },
                    {
                        "id": 2,
                        "type": "label",
                        "user": {"login": "alice"},
                        "label": {"name": "needs-review"},
                        "created_at": "2026-11-11T09:05:00Z",
                    },
                ],
                qs,
            )
        if path == "/api/v1/repos/owner/repo/issues/12/timeline":
            # имитируем старую версию Forgejo без timeline
            return self._json({"message": "not found"}, 404)

        if path == "/api/v1/repos/owner/repo/contents/src/login.py":
            content = b"def login(user):\n    token = check(user)\n    return token\n"
            return self._json(
                {"encoding": "base64", "size": len(content), "content": base64.b64encode(content).decode()}
            )
        if path == "/api/v1/repos/owner/repo/issues/13/timeline":
            return self._page([], qs)
        if path == "/api/v1/repos/owner/repo/contents/wip.txt":
            content = b"wip\n"
            return self._json(
                {"encoding": "base64", "size": len(content), "content": base64.b64encode(content).decode()}
            )
        if path == "/api/v1/repos/owner/repo/contents/README.md":
            content = b"# Project\nNew line\n"
            return self._json(
                {"encoding": "base64", "size": len(content), "content": base64.b64encode(content).decode()}
            )

        return self._json({"message": "not found", "path": path}, 404)

    # --- POST (LLM) ----------------------------------------------------- #
    def do_POST(self):
        if urlparse(self.path).path != "/v1/chat/completions":
            return self._json({"message": "not found"}, 404)

        # Проверяем Bearer-токен, если он сконфигурирован
        expected = os.environ.get("FAKE_OPENAI_TOKEN")
        if expected:
            got = self.headers.get("Authorization", "")
            if got != f"Bearer {expected}":
                return self._json({"error": {"message": "invalid api key", "code": "invalid_api_key"}}, 401)

        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        if payload.get("model") != "fake-model":
            return self._json({"error": {"message": "unknown model"}}, 400)

        user_msg = payload["messages"][-1]["content"]
        num = "11" if '"number": 11' in user_msg or '"pr_number": 11' in user_msg else "12"
        words = [f"Анализ PR #{num}: ", "изменения ", "выглядят ", "корректными. ", "LGTM."]

        if not payload.get("stream"):
            return self._json(
                {
                    "choices": [{"message": {"content": "".join(words)}}],
                    "usage": {"prompt_tokens": 123, "completion_tokens": 12},
                }
            )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for word in words:
            chunk = {"choices": [{"delta": {"content": word}}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        final = {"choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 12}}
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
