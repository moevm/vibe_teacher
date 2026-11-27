"""Тесты forgejo_pr_llm.py на моках HTTP (без реального Forgejo и LLM)."""

import base64
import json
import os
import sys

import pytest
import requests
import responses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import forgejo_pr_llm as app

API = "https://forgejo.example.com/api/v1"
REPO_URL = "https://forgejo.example.com/owner/repo"
LLM_URL = "http://localhost:11434/v1/chat/completions"


# --------------------------------------------------------------------------- #
#  parse_repo_url
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://forgejo.example.com/owner/repo", ("https://forgejo.example.com", "owner", "repo")),
        ("https://forgejo.example.com/owner/repo/", ("https://forgejo.example.com", "owner", "repo")),
        ("https://forgejo.example.com/owner/repo.git", ("https://forgejo.example.com", "owner", "repo")),
        ("https://forgejo.example.com/owner/repo/src/branch/main", ("https://forgejo.example.com", "owner", "repo")),
        ("https://forgejo.example.com/owner/repo/pulls/12", ("https://forgejo.example.com", "owner", "repo")),
        # инсталляция на сабпасе
        ("https://host.example.com/git/owner/repo", ("https://host.example.com/git", "owner", "repo")),
        ("http://localhost:3000/owner/repo", ("http://localhost:3000", "owner", "repo")),
        (
            "https://forgejo.example.com/owner/my.repo-name_x",
            ("https://forgejo.example.com", "owner", "my.repo-name_x"),
        ),
    ],
)
def test_parse_repo_url_ok(raw, expected):
    assert app.parse_repo_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "forgejo.example.com/owner/repo",
        "https://forgejo.example.com",
        "https://forgejo.example.com/owner",
        "",
        "   ",
        "ftp://host/owner/repo",
    ],
)
def test_parse_repo_url_bad(raw):
    with pytest.raises(app.ConfigError):
        app.parse_repo_url(raw)


# --------------------------------------------------------------------------- #
#  split_unified_diff
# --------------------------------------------------------------------------- #


def test_split_unified_diff_basic():
    diff = (
        "diff --git a/src/a.py b/src/a.py\n"
        "index 111..222 100644\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/docs/b.md b/docs/b.md\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/docs/b.md\n"
        "@@ -0,0 +1 @@\n+hello\n"
    )
    parts = app.split_unified_diff(diff)
    assert set(parts) == {"src/a.py", "docs/b.md"}
    assert "+new" in parts["src/a.py"]
    assert "+hello" in parts["docs/b.md"]


def test_split_unified_diff_renamed():
    diff = (
        "diff --git a/old/name.py b/new/name.py\n"
        "similarity index 100%\n"
        "rename from old/name.py\n"
        "rename to new/name.py\n"
    )
    parts = app.split_unified_diff(diff)
    assert "new/name.py" in parts


# --------------------------------------------------------------------------- #
#  Клиент Forgejo
# --------------------------------------------------------------------------- #


def _paged_callback(total_pages, per_page):
    """Отдаёт страницы по ?page=N; после последней — пустой список."""

    def callback(request):
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(request.url).query)
        page = int(qs.get("page", ["1"])[0])
        if page > total_pages:
            body = []
        else:
            base = (page - 1) * per_page
            body = [{"number": base + i} for i in range(per_page)]
        return (200, {"Content-Type": "application/json"}, json.dumps(body))

    return callback


@responses.activate
def test_pagination():
    responses.add_callback(
        responses.GET, f"{API}/repos/owner/repo/pulls", callback=_paged_callback(total_pages=2, per_page=5)
    )
    client = app.ForgejoClient(API, "tok", page_size=5, verbose=False)
    pulls = list(client.list_open_pulls("owner", "repo"))
    assert len(pulls) == 10
    assert [p["number"] for p in pulls] == list(range(10))
    # 2 полные страницы + одна пустая, подтверждающая конец
    assert len(responses.calls) == 3


@responses.activate
def test_pagination_respects_max_pages():
    responses.add_callback(
        responses.GET, f"{API}/repos/owner/repo/pulls", callback=_paged_callback(total_pages=999, per_page=5)
    )
    client = app.ForgejoClient(API, "tok", page_size=5, max_pages=3, verbose=False)
    assert len(list(client.list_open_pulls("owner", "repo"))) == 15


@responses.activate
def test_pagination_stops_on_short_page():
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls", json=[{"number": 1}, {"number": 2}], status=200)
    client = app.ForgejoClient(API, "tok", page_size=50, verbose=False)
    assert len(list(client.list_open_pulls("owner", "repo"))) == 2
    assert len(responses.calls) == 1


@responses.activate
def test_auth_header_sent():
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo"})
    client = app.ForgejoClient(API, "secret-token", verbose=False)
    client.get_repo("owner", "repo")
    assert responses.calls[0].request.headers["Authorization"] == "token secret-token"


@responses.activate
def test_401_message_is_helpful():
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"message": "unauthorized"}, status=401)
    client = app.ForgejoClient(API, "bad", verbose=False)
    with pytest.raises(app.ForgejoError) as exc:
        client.get_repo("owner", "repo")
    assert exc.value.status == 401
    assert "токен" in str(exc.value).lower()


@responses.activate
def test_retry_on_500_then_success(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda *_: None)
    responses.add(responses.GET, f"{API}/repos/owner/repo", status=500, body="boom")
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "ok"})
    client = app.ForgejoClient(API, "tok", retries=3, verbose=False)
    assert client.get_repo("owner", "repo")["full_name"] == "ok"
    assert len(responses.calls) == 2


@responses.activate
def test_timeline_404_returns_none():
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/1/timeline", json={"message": "not found"}, status=404)
    client = app.ForgejoClient(API, "tok", verbose=False)
    assert client.list_timeline("owner", "repo", 1) is None


# --------------------------------------------------------------------------- #
#  PRCollector
# --------------------------------------------------------------------------- #


def _file_entry(**over):
    base = {
        "filename": "src/app.py",
        "status": "modified",
        "additions": 2,
        "deletions": 1,
        "changes": 3,
        "patch": "@@ -1 +1,2 @@\n-old\n+new\n+more\n",
    }
    base.update(over)
    return base


def _pull(number=7, sha="abc123def456", full_name="owner/repo"):
    return {
        "number": number,
        "title": "Test PR",
        "state": "open",
        "labels": [{"name": "needs-review"}],
        "assignees": [{"login": "alice"}],
        "head": {"sha": sha, "ref": "feature", "repo": {"full_name": full_name}},
        "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
    }


@responses.activate
def test_collect_full_snapshot():
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/commits",
        json=[{"sha": "abc123def456", "commit": {"message": "fix"}}],
    )
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files", json=[_file_entry()])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/issues/7/comments",
        json=[{"id": 1, "user": {"login": "bob"}, "body": "looks good"}],
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/reviews",
        json=[{"id": 99, "state": "APPROVED", "user": {"login": "carol"}}],
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/reviews/99/comments",
        json=[{"id": 5, "path": "src/app.py", "line": 1, "body": "nit"}],
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/issues/7/timeline",
        json=[{"id": 1, "type": "comment", "created_at": "2026-01-01T00:00:00Z"}],
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7.diff",
        body="diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n"
        "+++ b/src/app.py\n@@ -1 +1,2 @@\n-old\n+new\n+more\n",
        content_type="text/plain",
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/contents/src/app.py",
        json={"encoding": "base64", "size": 8, "content": base64.b64encode(b"new\nmore\n").decode()},
    )

    client = app.ForgejoClient(API, "tok", verbose=False)
    collector = app.PRCollector(client, max_file_bytes=1024)
    warnings: list[str] = []
    snap = collector.collect("owner", "repo", _pull(), base_url="https://forgejo.example.com", warnings=warnings)

    assert snap["schema_version"] == "1.0"
    assert snap["head_sha"] == "abc123def456"
    assert snap["pr_number"] == 7
    assert snap["counts"] == {
        "commits": 1,
        "files": 1,
        "issue_comments": 1,
        "reviews": 1,
        "review_comments": 1,
        "timeline_events": 1,
    }
    f = snap["files"][0]
    assert f["patch_source"] == "api"
    assert f["content"]["text"] == "new\nmore\n"
    assert f["content"]["binary"] is False
    assert snap["review_comments"][0]["_review_id"] == 99
    assert not snap["fetch_errors"]


@responses.activate
def test_collect_uses_head_fork_repo_for_content():
    fork_pull = _pull(full_name="otheruser/repo")
    for ep in ("commits", "reviews"):
        responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/{ep}", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files", json=[_file_entry()])
    # контент лежит в форке, не в base-репозитории
    responses.add(
        responses.GET,
        f"{API}/repos/otheruser/repo/contents/src/app.py",
        json={"encoding": "base64", "size": 4, "content": base64.b64encode(b"body").decode()},
    )

    client = app.ForgejoClient(API, "tok", verbose=False)
    snap = app.PRCollector(client, max_file_bytes=1024).collect(
        "owner", "repo", fork_pull, base_url="https://forgejo.example.com", warnings=[]
    )
    assert snap["files"][0]["content"]["text"] == "body"
    called = [c.request.url for c in responses.calls]
    assert any("otheruser/repo/contents" in u for u in called)


@responses.activate
def test_deleted_file_has_no_content():
    for ep in ("commits", "reviews"):
        responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/{ep}", json=[])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/files",
        json=[_file_entry(status="removed", filename="gone.py", patch="")],
    )
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])

    client = app.ForgejoClient(API, "tok", verbose=False)
    snap = app.PRCollector(client, max_file_bytes=1024).collect(
        "owner", "repo", _pull(), base_url="https://forgejo.example.com", warnings=[]
    )
    assert snap["files"][0]["content"]["skipped_reason"] == "deleted_in_pr"


@responses.activate
def test_large_file_skipped():
    big = app.DEFAULT_MAX_FILE_BYTES + 10
    for ep in ("commits", "reviews"):
        responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/{ep}", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files", json=[_file_entry(filename="big.bin")])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/contents/big.bin",
        json={"encoding": "base64", "size": big, "content": ""},
    )

    warnings: list[str] = []
    client = app.ForgejoClient(API, "tok", verbose=False)
    snap = app.PRCollector(client, max_file_bytes=1024).collect(
        "owner", "repo", _pull(full_name="owner/repo"), base_url="x", warnings=warnings
    )
    content = snap["files"][0]["content"]
    assert content["skipped_reason"] == "too_large"
    assert content["size"] == big
    assert any("big.bin" in w for w in warnings)


@responses.activate
def test_binary_file_detected():
    blob = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    for ep in ("commits", "reviews"):
        responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/{ep}", json=[])
    responses.add(
        responses.GET, f"{API}/repos/owner/repo/pulls/7/files", json=[_file_entry(filename="img/logo.png", patch="")]
    )
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/contents/img/logo.png",
        json={"encoding": "base64", "size": len(blob), "content": base64.b64encode(blob).decode()},
    )

    client = app.ForgejoClient(API, "tok", verbose=False)
    snap = app.PRCollector(client, max_file_bytes=4096).collect("owner", "repo", _pull(), base_url="x", warnings=[])
    content = snap["files"][0]["content"]
    assert content["binary"] is True
    assert content["encoding"] == "base64"
    assert base64.b64decode(content["text"]) == blob


@responses.activate
def test_missing_patch_taken_from_unified_diff():
    for ep in ("commits", "reviews"):
        responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/{ep}", json=[])
    responses.add(
        responses.GET, f"{API}/repos/owner/repo/pulls/7/files", json=[_file_entry(patch="")]
    )  # сервер не отдал патч
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7.diff",
        body="diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n",
        content_type="text/plain",
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/contents/src/app.py",
        json={"encoding": "base64", "size": 4, "content": base64.b64encode(b"new\n").decode()},
    )

    client = app.ForgejoClient(API, "tok", verbose=False)
    snap = app.PRCollector(client, max_file_bytes=4096).collect("owner", "repo", _pull(), base_url="x", warnings=[])
    f = snap["files"][0]
    assert f["patch_source"] == "unified_diff"
    assert "+new" in f["patch"]


@responses.activate
def test_endpoint_failure_does_not_break_pr():
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits", json={"message": "boom"}, status=500)
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits", json={"message": "boom"}, status=500)
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits", json={"message": "boom"}, status=500)
    for ep in ("reviews",):
        responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/{ep}", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])

    warnings: list[str] = []
    client = app.ForgejoClient(API, "tok", retries=3, verbose=False)
    snap = app.PRCollector(client, max_file_bytes=1024).collect(
        "owner", "repo", _pull(), base_url="x", warnings=warnings
    )
    assert snap["commits"] == []
    assert snap["fetch_errors"][0]["status"] == 500
    assert warnings


# --------------------------------------------------------------------------- #
#  LLM
# --------------------------------------------------------------------------- #


def _sse(*chunks):
    lines = []
    for ch in chunks:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": ch}}]}, ensure_ascii=False) + "\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


@responses.activate
def test_llm_streaming_collects_text(capsys):
    responses.add(responses.POST, LLM_URL, body=_sse("Hello", ", ", "world"), content_type="text/event-stream")
    client = app.LLMClient(LLM_URL, "test-model", timeout=30, show_stream=False)
    answer = client.ask(client.build_messages("prompt", '{"a":1}'))
    assert answer == "Hello, world"
    body = json.loads(responses.calls[0].request.body)
    assert body["model"] == "test-model"
    assert body["stream"] is True
    assert body["messages"][0] == {"role": "system", "content": "prompt"}
    assert '{"a":1}' in body["messages"][1]["content"]


@responses.activate
def test_llm_non_streaming_json_fallback():
    responses.add(
        responses.POST,
        LLM_URL,
        json={
            "choices": [{"message": {"content": "plain answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        content_type="application/json",
    )
    client = app.LLMClient(LLM_URL, "m", timeout=30, show_stream=False)
    assert client.ask([{"role": "user", "content": "hi"}]) == "plain answer"


@responses.activate
def test_llm_falls_back_when_stream_rejected(monkeypatch):
    """Сервер, который не понимает stream:true, но отвечает на stream:false."""
    monkeypatch.setattr(app.time, "sleep", lambda *_: None)
    seen: list[bool] = []

    def callback(request):
        payload = json.loads(request.body)
        seen.append(bool(payload.get("stream")))
        if payload.get("stream"):
            return (400, {"Content-Type": "application/json"}, '{"error": "no stream"}')
        body = {"choices": [{"message": {"content": "ok-nonstream"}}]}
        return (200, {"Content-Type": "application/json"}, json.dumps(body))

    responses.add_callback(responses.POST, LLM_URL, callback=callback)
    client = app.LLMClient(LLM_URL, "m", timeout=30, retries=3, show_stream=False)
    assert client.ask([{"role": "user", "content": "hi"}]) == "ok-nonstream"
    assert seen == [True, False]  # один стриминговый провал, затем обычный запрос


@responses.activate
def test_llm_timeout_raises(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda *_: None)

    def boom(*a, **kw):
        raise app.requests.Timeout("timed out")

    responses.add_callback(responses.POST, LLM_URL, callback=boom)
    client = app.LLMClient(LLM_URL, "m", timeout=1, retries=1, show_stream=False)
    with pytest.raises(app.LLMError) as exc:
        client.ask([{"role": "user", "content": "hi"}])
    assert "таймаут" in str(exc.value).lower()


@responses.activate
def test_llm_empty_answer_raises(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda *_: None)
    for _ in range(3):
        responses.add(responses.POST, LLM_URL, json={"choices": [{"message": {"content": ""}}]})
    client = app.LLMClient(LLM_URL, "m", timeout=5, retries=1, show_stream=False)
    with pytest.raises(app.LLMError):
        client.ask([{"role": "user", "content": "hi"}])


def test_llm_prompt_size_warning(caplog):
    client = app.LLMClient(LLM_URL, "m", max_prompt_bytes=10)
    messages = client.build_messages("p" * 100, "{}")
    size = len(json.dumps(messages, ensure_ascii=False).encode())
    assert size > 10  # предупреждение будет напечатано в ask()


# --------------------------------------------------------------------------- #
#  Оркестрация / идемпотентность
# --------------------------------------------------------------------------- #


def _wsgi_ok(*args, **kwargs):
    return (200, {"Content-Type": "application/json"}, json.dumps({"full_name": "owner/repo"}))


def test_json_and_answer_naming():
    assert app.Orchestrator.json_name(7, "abc123") == "7_abc123.json"
    assert app.Orchestrator.answer_name(7, "abc123") == "answer_7_abc123.txt"


def test_safe_filename_part():
    assert app.safe_filename_part("abc/../x y") == "abc-..-x-y"
    assert app.safe_filename_part("") == "unknown"


@pytest.fixture
def prompt_file(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("Проверь PR.", encoding="utf-8")
    return str(p)


def _ns(tmp_path, prompt_file, **over):
    args = app.build_parser().parse_args(
        [
            REPO_URL,
            "--token",
            "tok",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_validate_args_requires_token(prompt_file):
    args = app.build_parser().parse_args(
        [
            REPO_URL,
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
        ]
    )
    args.token = None
    with pytest.raises(app.ConfigError):
        app.validate_args(args)


def test_validate_args_missing_prompt(tmp_path):
    args = app.build_parser().parse_args(
        [
            REPO_URL,
            "--token",
            "tok",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            str(tmp_path / "nope.txt"),
        ]
    )
    with pytest.raises(app.ConfigError):
        app.validate_args(args)


@responses.activate
def test_skips_existing_answer_and_reuses_json(tmp_path, prompt_file):
    out = tmp_path / "out"
    out.mkdir()
    sha = "deadbeef"
    (out / f"7_{sha}.json").write_text('{"pull_request": {"number": 7}}', encoding="utf-8")
    (out / f"answer_7_{sha}.txt").write_text("готовый ответ", encoding="utf-8")

    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo"})
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls",
        json=[
            {
                "number": 7,
                "title": "T",
                "labels": [],
                "assignees": [],
                "head": {"sha": sha, "repo": {"full_name": "owner/repo"}},
            },
        ],
    )

    code = app.main(
        [
            REPO_URL,
            "--token",
            "tok",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    # к LLM обращений не было
    assert not any(c.request.url.startswith(LLM_URL) for c in responses.calls)
    assert (out / f"answer_7_{sha}.txt").read_text(encoding="utf-8") == "готовый ответ"


@responses.activate
def test_full_run_writes_json_and_answer(tmp_path, prompt_file):
    out = tmp_path / "out"
    sha = "cafe01"
    pull = {
        "number": 3,
        "title": "Add feature",
        "state": "open",
        "labels": [{"name": "needs-review"}],
        "assignees": [{"login": "alice"}],
        "head": {"sha": sha, "repo": {"full_name": "owner/repo"}},
    }

    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo"})
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls", json=[pull])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3/commits", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3/files", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/3/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3/reviews", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/3/timeline", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3.diff", body="", content_type="text/plain")
    responses.add(responses.POST, LLM_URL, body=_sse("LGTM"), content_type="text/event-stream")

    code = app.main(
        [
            REPO_URL,
            "--token",
            "tok",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
            "--output-dir",
            str(out),
            "--label",
            "needs-review",
            "--assignee",
            "alice",
        ]
    )
    assert code == 0
    assert (out / f"3_{sha}.json").exists()
    assert (out / f"answer_3_{sha}.txt").read_text(encoding="utf-8").strip() == "LGTM"

    snap = json.loads((out / f"3_{sha}.json").read_text(encoding="utf-8"))
    assert snap["pr_number"] == 3
    assert snap["head_sha"] == sha


@responses.activate
def test_label_filter_excludes(tmp_path, prompt_file):
    out = tmp_path / "out"
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo"})
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls",
        json=[
            {
                "number": 1,
                "labels": [{"name": "other"}],
                "assignees": [{"login": "alice"}],
                "head": {"sha": "aa", "repo": {"full_name": "owner/repo"}},
            },
            {
                "number": 2,
                "labels": [{"name": "needs-review"}],
                "assignees": [{"login": "bob"}],
                "head": {"sha": "bb", "repo": {"full_name": "owner/repo"}},
            },
        ],
    )
    code = app.main(
        [
            REPO_URL,
            "--token",
            "tok",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
            "--output-dir",
            str(out),
            "--label",
            "needs-review",
            "--assignee",
            "alice",
        ]
    )
    # ни один PR не подходит под И(label, assignee)
    assert code == 0
    assert not out.exists() or not any(out.iterdir())


def test_filter_requires_both():
    args = app.build_parser().parse_args(
        [
            REPO_URL,
            "--token",
            "t",
            "--llm-url",
            "u",
            "--model",
            "m",
            "--prompt-file",
            "x",
            "--label",
            "bug",
            "--assignee",
            "Alice",
        ]
    )
    orch = app.Orchestrator(args, "https://h", "o", "r")
    pull = {"labels": [{"name": "bug"}], "assignees": [{"login": "alice"}]}
    assert orch._matches_filters(pull) is True
    assert orch._matches_filters({"labels": [{"name": "bug"}], "assignees": []}) is False
    assert orch._matches_filters({"labels": [], "assignees": [{"login": "alice"}]}) is False
    # регистронезависимо
    assert orch._matches_filters({"labels": [{"name": "BUG"}], "assignees": [{"login": "ALICE"}]}) is True


def test_filter_no_criteria_matches_all():
    args = app.build_parser().parse_args(
        [
            REPO_URL,
            "--token",
            "t",
            "--llm-url",
            "u",
            "--model",
            "m",
            "--prompt-file",
            "x",
        ]
    )
    orch = app.Orchestrator(args, "https://h", "o", "r")
    assert orch._matches_filters({"labels": [], "assignees": []}) is True


# --------------------------------------------------------------------------- #
#  parse_label_filters / --label со списком и отрицаниями
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ([], [])),
        ("bug", (["bug"], [])),
        ("bug,docs", (["bug", "docs"], [])),
        (["bug", "-wip"], (["bug"], ["wip"])),
        (["bug,docs", "-wip"], (["bug", "docs"], ["wip"])),
        (["bug", "docs", "-wip", "-draft"], (["bug", "docs"], ["wip", "draft"])),
        (["  bug , , docs  "], (["bug", "docs"], [])),
        (["bug,bug"], (["bug"], [])),  # дубликаты схлопываются
        (["-wip,-wip"], ([], ["wip"])),
        (["+dash"], (["dash"], [])),  # '+' — явное обязательное
        (["\\-dash"], (["-dash"], [])),  # экранирование ведущего дефиса
    ],
)
def test_parse_label_filters(value, expected):
    assert app.parse_label_filters(value) == expected


@pytest.mark.parametrize("bad", [["bug,-bug"], ["-"], ["-,,,"], ["bug", "-"]])
def test_parse_label_filters_errors(bad):
    with pytest.raises(app.ConfigError):
        app.parse_label_filters(bad)


def test_format_label_filters():
    assert app.format_label_filters(["bug", "docs"], ["wip"]) == "bug, docs, -wip"
    assert app.format_label_filters([], ["wip"]) == "-wip"
    assert app.format_label_filters(["bug"], []) == "bug"


def _orch(label=None, assignee=None, title_contains=None):
    argv = [REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", "x"]
    if label is not None:
        for value in label:
            argv += ["--label", value]
    if assignee:
        argv += ["--assignee", assignee]
    if title_contains is not None:
        argv += ["--title-contains", title_contains]
    args = app.build_parser().parse_args(app.normalize_argv(argv))
    return app.Orchestrator(args, "https://h", "o", "r")


def _pull_labels(labels, assignees=()):
    return {"labels": [{"name": n} for n in labels], "assignees": [{"login": a} for a in assignees]}


@pytest.mark.parametrize(
    "filter_args,labels,expected",
    [
        (["bug"], ["bug"], True),
        (["bug"], ["other"], False),
        (["bug"], ["bug", "docs"], True),  # лишние метки не мешают
        (["bug,docs"], ["bug"], False),  # нужны обе (И)
        (["bug,docs"], ["bug", "docs"], True),
        (["bug,docs"], ["bug", "docs", "wip"], True),
        (["-wip"], ["bug"], True),  # только запрет
        (["-wip"], ["bug", "wip"], False),
        (["-wip"], ["WIP"], False),  # регистронезависимо
        (["bug,-wip"], ["bug"], True),
        (["bug,-wip"], ["bug", "wip"], False),
        (["bug", "-wip"], ["bug", "docs"], True),  # повторный флаг
        ([], ["anything"], True),  # без фильтра
        (["Bug,NEEDS-REVIEW"], ["bug", "needs-review"], True),
    ],
)
def test_label_filter_matching(filter_args, labels, expected):
    assert _orch(label=filter_args)._matches_filters(_pull_labels(labels)) is expected


def test_label_filter_combination_with_assignee():
    orch = _orch(label=["bug,-wip"], assignee="alice")
    assert orch._matches_filters(_pull_labels(["bug"], ["alice"])) is True
    assert orch._matches_filters(_pull_labels(["bug"], ["bob"])) is False
    assert orch._matches_filters(_pull_labels(["bug", "wip"], ["alice"])) is False


@pytest.mark.parametrize(
    "substring,title,expected",
    [
        ("fix", "Fix login form", True),
        ("FIX", "fix: typo", True),
        ("login form", "Fix login form", True),
        ("-wip", "WIP: refactor -wip", True),
        ("fix", "Update README", False),
        ("fix", "", False),
        ("fix", None, False),
    ],
)
def test_title_contains_matching(substring, title, expected):
    assert _orch(title_contains=substring)._matches_filters({"title": title}) is expected


def test_title_contains_combined_with_label_and_assignee():
    orch = _orch(label=["bug,-wip"], assignee="alice", title_contains="fix")
    pull = {"title": "Fix login", **_pull_labels(["bug"], ["alice"])}
    assert orch._matches_filters(pull) is True
    assert orch._matches_filters({**pull, "title": "Update README"}) is False
    assert orch._matches_filters({**pull, **_pull_labels(["bug", "wip"], ["alice"])}) is False
    assert orch._matches_filters({**pull, **_pull_labels(["bug"], ["bob"])}) is False


@pytest.mark.parametrize("value", ["", "   "])
def test_title_contains_rejects_blank(prompt_file, value):
    argv = [REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", prompt_file]
    args = app.build_parser().parse_args(app.normalize_argv(argv + ["--title-contains", value]))
    with pytest.raises(app.ConfigError, match="--title-contains"):
        app.validate_args(args)


@responses.activate
def test_title_contains_dry_run_lists_only_matching(prompt_file, capsys):
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo"})
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls",
        json=[
            {"number": 1, "title": "Fix login form", "head": {"sha": "aa1111"}},
            {"number": 2, "title": "Update README", "head": {"sha": "bb2222"}},
            {"number": 3, "title": "FIX typo in docs", "head": {"sha": "cc3333"}},
        ],
    )
    code = app.main(
        [
            REPO_URL,
            "--token",
            "tok",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
            "--title-contains",
            "fix",
            "--dry-run",
        ]
    )
    err = capsys.readouterr().err
    assert code == 0
    assert "#   1 [aa1111] Fix login form" in err
    assert "#   3 [cc3333] FIX typo in docs" in err
    assert "#   2" not in err
    assert "заголовок содержит «fix»" in err


def test_label_filters_parsed_once(monkeypatch):
    """Парсер вызывается один раз за прогон, а не на каждый PR."""
    calls = []
    real = app.parse_label_filters

    def spy(value):
        calls.append(value)
        return real(value)

    monkeypatch.setattr(app, "parse_label_filters", spy)
    orch = _orch(label=["bug"])
    for _ in range(5):
        orch._matches_filters(_pull(["bug"]))
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
#  --openai-api-token
# --------------------------------------------------------------------------- #


def test_openai_api_token_from_arg(monkeypatch):
    monkeypatch.delenv("OPENAI_API_TOKEN", raising=False)
    argv = [
        REPO_URL,
        "--token",
        "t",
        "--llm-url",
        "u",
        "--model",
        "m",
        "--prompt-file",
        "x",
        "--openai-api-token",
        "sk-abc",
    ]
    assert app.build_parser().parse_args(argv).openai_api_token == "sk-abc"


def test_openai_api_token_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_TOKEN", "sk-from-env")
    argv = [REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", "x"]
    assert app.build_parser().parse_args(argv).openai_api_token == "sk-from-env"


def test_openai_api_token_absent(monkeypatch):
    monkeypatch.delenv("OPENAI_API_TOKEN", raising=False)
    argv = [REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", "x"]
    assert app.build_parser().parse_args(argv).openai_api_token is None


def test_llm_client_sends_bearer_token():
    client = app.LLMClient(LLM_URL, "m", api_token="sk-secret")
    assert client.session.headers["Authorization"] == "Bearer sk-secret"


def test_llm_client_no_auth_header_without_token():
    client = app.LLMClient(LLM_URL, "m")
    assert "Authorization" not in client.session.headers

    client = app.LLMClient(LLM_URL, "m", api_token="   ")
    assert "Authorization" not in client.session.headers


@responses.activate
def test_llm_request_carries_auth_header():
    responses.add(responses.POST, LLM_URL, json={"choices": [{"message": {"content": "ok"}}]})
    client = app.LLMClient(LLM_URL, "m", timeout=5, show_stream=False, api_token="sk-live")
    client.ask([{"role": "user", "content": "hi"}])
    assert responses.calls[0].request.headers["Authorization"] == "Bearer sk-live"


@responses.activate
def test_forgejo_token_not_sent_to_llm(tmp_path, prompt_file):
    """Токен Forgejo не должен утекать в запрос к LLM."""
    out = tmp_path / "out"
    sha = "beef01"
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo"})
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls",
        json=[
            {
                "number": 1,
                "title": "T",
                "labels": [],
                "assignees": [],
                "head": {"sha": sha, "repo": {"full_name": "owner/repo"}},
            },
        ],
    )
    for ep in ("commits", "files"):
        responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/1/{ep}", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/1/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/1/reviews", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/1/timeline", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/1.diff", body="", content_type="text/plain")
    responses.add(responses.POST, LLM_URL, json={"choices": [{"message": {"content": "ok"}}]})

    app.main(
        [
            REPO_URL,
            "--token",
            "forgejo-secret",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
            "--output-dir",
            str(out),
        ]
    )

    llm_calls = [c for c in responses.calls if c.request.url.startswith(LLM_URL)]
    assert len(llm_calls) == 1
    assert "Authorization" not in llm_calls[0].request.headers
    body = llm_calls[0].request.body or b""
    body_text = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
    assert "forgejo-secret" not in body_text


# --------------------------------------------------------------------------- #
#  Краевые случаи разбора аргументов
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        (["--label", "-wip"], ["--label=-wip"]),
        (["--label=-wip"], ["--label=-wip"]),
        (["--label", "bug,docs", "--label", "-wip"], ["--label=bug,docs", "--label=-wip"]),
        (["--model", "gpt-4o-mini"], ["--model=gpt-4o-mini"]),
        (["--title-contains", "-wip"], ["--title-contains=-wip"]),
        (["--no-stream"], ["--no-stream"]),
        # следующий токен — опция, склеивать нельзя: argparse должен ругаться
        (["--label", "--dry-run"], ["--label", "--dry-run"]),
        # всё после "--" не трогаем
        (["--", "--label", "x"], ["--", "--label", "x"]),
    ],
)
def test_normalize_argv(raw, expected):
    assert app.normalize_argv(raw) == expected


def test_label_dash_value_accepted():
    """--label -wip не должен восприниматься как неизвестная опция."""
    argv = app.normalize_argv(
        [REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", "x", "--label", "-wip"]
    )
    args = app.build_parser().parse_args(argv)
    assert app.parse_label_filters(args.label) == ([], ["wip"])


@pytest.mark.parametrize("empty", [[""], [" "], [" , "], [","]])
def test_label_empty_value_is_error(empty):
    """Пустое значение --label не должно молча означать «все PR»."""
    with pytest.raises(app.ConfigError):
        app.parse_label_filters(empty)


def test_validate_args_rejects_bad_labels(prompt_file):
    args = app.build_parser().parse_args(
        app.normalize_argv(
            [
                REPO_URL,
                "--token",
                "t",
                "--llm-url",
                "u",
                "--model",
                "m",
                "--prompt-file",
                prompt_file,
                "--label",
                "bug,-bug",
            ]
        )
    )
    with pytest.raises(app.ConfigError):
        app.validate_args(args)


# --------------------------------------------------------------------------- #
#  --light-pr: облегчённый формат снимка
# --------------------------------------------------------------------------- #


@responses.activate
def test_light_pr_mode_produces_compact_snapshot():
    """light_pr=True: файлы с патчами, незакрытые review-комментарии, без content/commits/timeline."""
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files", json=[_file_entry()])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7.diff",
        body="diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n"
        "+++ b/src/app.py\n@@ -1 +1,2 @@\n-old\n+new\n+more\n",
        content_type="text/plain",
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/reviews",
        json=[{"id": 99, "state": "APPROVED", "user": {"login": "carol"}}],
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/reviews/99/comments",
        json=[
            {
                "id": 1,
                "path": "src/app.py",
                "line": 2,
                "original_line": 1,
                "body": "open comment",
                "user": {"login": "carol"},
                "created_at": "2026-01-01T00:00:00Z",
                "resolved_at": None,
            },
            {
                "id": 2,
                "path": "src/app.py",
                "line": 3,
                "original_line": 1,
                "body": "resolved comment",
                "user": {"login": "dave"},
                "created_at": "2026-01-02T00:00:00Z",
                "resolved_at": "2026-01-03T00:00:00Z",
            },
        ],
    )

    client = app.ForgejoClient(API, "tok", verbose=False)
    collector = app.PRCollector(client, max_file_bytes=1024, light_pr=True)
    snap = collector.collect("owner", "repo", _pull(), base_url="https://x", warnings=[])

    assert snap["mode"] == "light"
    assert snap["schema_version"] == "1.0-light"
    # Нет тяжёлых секций
    assert "commits" not in snap
    assert "timeline" not in snap
    assert "unified_diff" not in snap
    assert "issue_comments" not in snap
    # Файлы есть, с патчем, но без content
    assert len(snap["files"]) == 1
    f = snap["files"][0]
    assert f["filename"] == "src/app.py"
    assert f["patch"] != ""
    assert "content" not in f
    # Только незакрытый комментарий
    comments = snap["review_comments"]
    assert len(comments) == 1
    assert comments[0]["body"] == "open comment"
    assert comments[0]["path"] == "src/app.py"
    assert comments[0]["line"] == 2
    assert comments[0]["user"] == "carol"
    assert comments[0]["review_id"] == 99


@responses.activate
def test_light_pr_excludes_resolved_comments():
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/reviews",
        json=[{"id": 99, "state": "COMMENTED", "user": {"login": "x"}}],
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/reviews/99/comments",
        json=[
            {"id": 1, "path": "a.py", "line": 1, "body": "open", "user": {"login": "x"}, "resolved_at": None},
            {"id": 2, "path": "b.py", "line": 2, "body": "closed", "user": {"login": "y"}, "resolved_at": "2026-01-01"},
            {"id": 3, "path": "c.py", "line": 3, "body": "also open", "user": {"login": "z"}, "resolved_at": None},
        ],
    )

    client = app.ForgejoClient(API, "tok", verbose=False)
    snap = app.PRCollector(client, max_file_bytes=1024, light_pr=True).collect(
        "owner", "repo", _pull(), base_url="x", warnings=[]
    )
    bodies = [c["body"] for c in snap["review_comments"]]
    assert "closed" not in bodies
    assert "open" in bodies
    assert "also open" in bodies


# --------------------------------------------------------------------------- #
#  --comment-filter: фильтр комментариев по авторам
# --------------------------------------------------------------------------- #


def test_filter_comments_keeps_only_required():
    comments = [
        {"user": {"login": "alice"}, "body": "1"},
        {"user": {"login": "bob"}, "body": "2"},
        {"user": {"login": "carol"}, "body": "3"},
    ]
    result = app.PRCollector._filter_comments(comments, (["alice"], []))
    assert [c["body"] for c in result] == ["1"]


def test_filter_comments_excludes_forbidden():
    comments = [
        {"user": {"login": "alice"}, "body": "1"},
        {"user": {"login": "bot"}, "body": "2"},
        {"user": {"login": "carol"}, "body": "3"},
    ]
    result = app.PRCollector._filter_comments(comments, ([], ["bot"]))
    assert [c["body"] for c in result] == ["1", "3"]


def test_filter_comments_combined_required_and_forbidden():
    comments = [
        {"user": {"login": "alice"}, "body": "1"},
        {"user": {"login": "bob"}, "body": "2"},
        {"user": {"login": "bot"}, "body": "3"},
    ]
    result = app.PRCollector._filter_comments(comments, (["alice", "bob"], ["bot"]))
    assert [c["body"] for c in result] == ["1", "2"]


def test_filter_comments_case_insensitive():
    comments = [{"user": {"login": "Alice"}, "body": "1"}]
    result = app.PRCollector._filter_comments(comments, (["alice"], []))
    assert len(result) == 1


def test_filter_comments_empty_required_keeps_all():
    comments = [{"user": {"login": "a"}, "body": "1"}, {"user": {"login": "b"}, "body": "2"}]
    result = app.PRCollector._filter_comments(comments, ([], []))
    assert len(result) == 2


@responses.activate
def test_comment_filter_in_full_mode():
    """comment_filter применяется и в полном режиме: issue_comments + review_comments фильтруются."""
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files", json=[])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/issues/7/comments",
        json=[
            {"id": 1, "user": {"login": "alice"}, "body": "keep", "created_at": "2026-01-01"},
            {"id": 2, "user": {"login": "bot"}, "body": "drop", "created_at": "2026-01-02"},
        ],
    )
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/reviews",
        json=[{"id": 50, "state": "COMMENTED", "user": {"login": "x"}}],
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/reviews/50/comments",
        json=[
            {
                "id": 10,
                "user": {"login": "alice"},
                "path": "a.py",
                "line": 1,
                "body": "inline-keep",
                "resolved_at": None,
            },
            {"id": 11, "user": {"login": "bot"}, "path": "a.py", "line": 2, "body": "inline-drop", "resolved_at": None},
        ],
    )

    client = app.ForgejoClient(API, "tok", verbose=False)
    cf = (["alice"], ["bot"])
    collector = app.PRCollector(client, max_file_bytes=1024, comment_filter=cf)
    snap = collector.collect("owner", "repo", _pull(), base_url="x", warnings=[])

    issue_bodies = [c["body"] for c in snap["issue_comments"]]
    assert "keep" in issue_bodies
    assert "drop" not in issue_bodies

    review_bodies = [c["body"] for c in snap["review_comments"]]
    assert "inline-keep" in review_bodies
    assert "inline-drop" not in review_bodies


@responses.activate
def test_comment_filter_in_light_mode():
    """comment_filter отсекает review-комментарии и в light-режиме."""
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files", json=[])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/issues/7/comments",
        json=[{"id": 1, "user": {"login": "bot"}, "body": "noise"}],
    )
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/reviews",
        json=[{"id": 60, "state": "COMMENTED", "user": {"login": "x"}}],
    )
    responses.add(
        responses.GET,
        f"{API}/repos/owner/repo/pulls/7/reviews/60/comments",
        json=[
            {"id": 20, "user": {"login": "alice"}, "path": "a.py", "line": 1, "body": "keep", "resolved_at": None},
            {"id": 21, "user": {"login": "bot"}, "path": "a.py", "line": 2, "body": "drop", "resolved_at": None},
        ],
    )

    client = app.ForgejoClient(API, "tok", verbose=False)
    cf = (["alice"], [])
    collector = app.PRCollector(client, max_file_bytes=1024, light_pr=True, comment_filter=cf)
    snap = collector.collect("owner", "repo", _pull(), base_url="x", warnings=[])

    bodies = [c["body"] for c in snap["review_comments"]]
    assert bodies == ["keep"]


# --------------------------------------------------------------------------- #
#  CLI: --light-pr и --comment-filter
# --------------------------------------------------------------------------- #


def test_light_pr_flag_default_false():
    argv = [REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", "x"]
    assert app.build_parser().parse_args(app.normalize_argv(argv)).light_pr is False


def test_light_pr_flag_set_true():
    argv = [REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", "x", "--light-pr"]
    assert app.build_parser().parse_args(app.normalize_argv(argv)).light_pr is True


def test_comment_filter_parse():
    argv = app.normalize_argv(
        [
            REPO_URL,
            "--token",
            "t",
            "--llm-url",
            "u",
            "--model",
            "m",
            "--prompt-file",
            "x",
            "--comment-filter",
            "alice,bob,-bot",
        ]
    )
    args = app.build_parser().parse_args(argv)
    assert args.comment_filter == ["alice,bob,-bot"]
    req, forb = app.parse_label_filters(args.comment_filter)
    assert req == ["alice", "bob"]
    assert forb == ["bot"]


def test_comment_filter_repeated_flag():
    argv = app.normalize_argv(
        [
            REPO_URL,
            "--token",
            "t",
            "--llm-url",
            "u",
            "--model",
            "m",
            "--prompt-file",
            "x",
            "--comment-filter",
            "alice",
            "--comment-filter",
            "-bot",
        ]
    )
    args = app.build_parser().parse_args(argv)
    req, forb = app.parse_label_filters(args.comment_filter)
    assert req == ["alice"]
    assert forb == ["bot"]


def test_comment_filter_empty_is_none():
    """Нет --comment-filter -> None, не ошибка."""
    argv = [REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", "x"]
    args = app.build_parser().parse_args(argv)
    assert args.comment_filter is None


# --------------------------------------------------------------------------- #
#  --comments-skip-resolved-by: пропуск тредов, решённых указанными пользователями
# --------------------------------------------------------------------------- #

SKIP_FLAG = "--comments-skip-resolved-by"


def _rc(cid, body, *, review=50, path="a.py", position=3, original_position=0, user="carol", resolver=None):
    """Inline-комментарий в формате Forgejo: resolver стоит только на первом комментарии треда."""
    return {
        "id": cid,
        "body": body,
        "user": {"login": user},
        "resolver": {"login": resolver} if resolver else None,
        "pull_request_review_id": review,
        "_review_id": review,
        "path": path,
        "position": position,
        "original_position": original_position,
        "created_at": f"2026-01-01T00:00:{cid:02d}Z",
    }


def _resolved_threads():
    """Три треда одного ревью: решён alice (с ответом), решён dave, открытый."""
    return [
        _rc(10, "root, resolved by alice", user="bot", resolver="alice"),
        _rc(11, "reply in the thread resolved by alice"),
        _rc(12, "resolved by dave", position=7, resolver="dave"),
        _rc(13, "open", position=9),
    ]


def _mock_pr7(review_comments, timeline=()):
    """PR #7 с одним ревью (id 50); timeline=None — endpoint недоступен (404)."""
    base = f"{API}/repos/owner/repo"
    responses.add(responses.GET, f"{base}/pulls/7/commits", json=[])
    responses.add(responses.GET, f"{base}/pulls/7/files", json=[])
    responses.add(responses.GET, f"{base}/issues/7/comments", json=[])
    responses.add(responses.GET, f"{base}/pulls/7.diff", body="", content_type="text/plain")
    responses.add(
        responses.GET, f"{base}/pulls/7/reviews", json=[{"id": 50, "state": "COMMENTED", "user": {"login": "x"}}]
    )
    responses.add(responses.GET, f"{base}/pulls/7/reviews/50/comments", json=review_comments)
    if timeline is None:
        responses.add(responses.GET, f"{base}/issues/7/timeline", status=404)
    else:
        responses.add(responses.GET, f"{base}/issues/7/timeline", json=list(timeline))


def _collect_pr7(**collector_kwargs):
    client = app.ForgejoClient(API, "tok", verbose=False)
    collector = app.PRCollector(client, max_file_bytes=1024, **collector_kwargs)
    return collector.collect("owner", "repo", _pull(), base_url="x", warnings=[])


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, []),
        ("alice", ["alice"]),
        (["alice,bob"], ["alice", "bob"]),
        (["alice", " bob , carol "], ["alice", "bob", "carol"]),
        (["alice,,alice"], ["alice"]),
    ],
)
def test_parse_login_list(value, expected):
    assert app.parse_login_list(value, flag=SKIP_FLAG) == expected


@pytest.mark.parametrize("bad", [[""], [" , "], ["-bot"], ["alice,-bot"], ["alice", ""]])
def test_parse_login_list_errors(bad):
    with pytest.raises(app.ConfigError, match=SKIP_FLAG):
        app.parse_login_list(bad, flag=SKIP_FLAG)


def test_skip_resolved_flag_default_is_none():
    argv = [REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", "x"]
    assert app.build_parser().parse_args(argv).comments_skip_resolved_by is None


def test_skip_resolved_flag_parse():
    argv = app.normalize_argv(
        [
            REPO_URL,
            "--token",
            "t",
            "--llm-url",
            "u",
            "--model",
            "m",
            "--prompt-file",
            "x",
            SKIP_FLAG,
            "alice,bob",
            f"{SKIP_FLAG}=carol",
        ]
    )
    args = app.build_parser().parse_args(argv)
    assert args.comments_skip_resolved_by == ["alice,bob", "carol"]
    assert app.parse_login_list(args.comments_skip_resolved_by, flag=SKIP_FLAG) == ["alice", "bob", "carol"]


@pytest.mark.parametrize("value", ["", "-bot"])
def test_validate_args_rejects_bad_skip_resolved_by(tmp_path, prompt_file, value):
    args = _ns(tmp_path, prompt_file, comments_skip_resolved_by=[value])
    with pytest.raises(app.ConfigError, match=SKIP_FLAG):
        app.validate_args(args)


def test_skip_resolved_dash_value_is_config_error_not_argparse_error(prompt_file, capsys):
    """`--comments-skip-resolved-by -bot` доходит до проверки значения и даёт код 2 с понятным текстом."""
    code = app.main(
        [
            REPO_URL,
            "--token",
            "t",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
            SKIP_FLAG,
            "-bot",
        ]
    )
    assert code == 2
    assert SKIP_FLAG in capsys.readouterr().err


def test_skip_resolved_threads_drops_whole_thread():
    comments = [
        *_resolved_threads(),
        _rc(14, "same line, another review", review=51),
        _rc(15, "same number, old side of the diff", position=0, original_position=3),
        _rc(16, "same line, another file", path="b.py"),
    ]
    # логин во флаге сравнивается без учёта регистра: PRCollector приводит его к нижнему
    kept, skipped = app.PRCollector._skip_resolved_threads(comments, {"alice"})
    assert [c["id"] for c in skipped] == [10, 11]
    assert [c["id"] for c in kept] == [12, 13, 14, 15, 16]


def test_skip_resolved_threads_resolver_login_case_insensitive():
    kept, skipped = app.PRCollector._skip_resolved_threads([_rc(1, "x", resolver="Alice")], {"alice"})
    assert kept == []
    assert len(skipped) == 1


def test_skip_resolved_threads_without_coordinates_are_not_merged():
    """Если сервер не отдал position/original_position, соседние комментарии не считаются одним тредом."""
    comments = [
        {"id": 1, "_review_id": 50, "path": "a.py", "body": "resolved", "resolver": {"login": "alice"}},
        {"id": 2, "_review_id": 50, "path": "a.py", "body": "unrelated"},
    ]
    kept, skipped = app.PRCollector._skip_resolved_threads(comments, {"alice"})
    assert [c["id"] for c in skipped] == [1]
    assert [c["id"] for c in kept] == [2]


@responses.activate
def test_skip_resolved_by_in_full_mode():
    _mock_pr7(
        _resolved_threads(),
        timeline=[
            # релизные версии Forgejo отдают inline-комментарии и в timeline (type=code)
            {"id": 10, "type": "code", "body": "root, resolved by alice", "resolve_doer": {"login": "alice"}},
            {"id": 11, "type": "code", "body": "reply in the thread resolved by alice", "resolve_doer": None},
            {"id": 12, "type": "code", "body": "resolved by dave", "resolve_doer": {"login": "dave"}},
            {"id": 77, "type": "comment", "body": "plain comment", "resolve_doer": None},
            {"id": 78, "type": "label", "body": "", "resolve_doer": None},
        ],
    )
    snap = _collect_pr7(skip_resolved_by=["Alice"])

    assert [c["id"] for c in snap["review_comments"]] == [12, 13]
    assert [e["id"] for e in snap["timeline"]] == [12, 77, 78]
    assert snap["counts"]["review_comments"] == 2
    assert snap["counts"]["timeline_events"] == 3
    assert snap["counts"]["resolved_review_comments_skipped"] == 2
    assert "resolved by alice" not in json.dumps(snap)


@responses.activate
def test_skip_resolved_by_several_users():
    _mock_pr7(_resolved_threads())
    snap = _collect_pr7(skip_resolved_by=["alice", "dave"])
    assert [c["id"] for c in snap["review_comments"]] == [13]
    assert snap["counts"]["resolved_review_comments_skipped"] == 3


@responses.activate
def test_resolved_comments_kept_without_flag():
    """Без флага снимок не меняется: решённые треды на месте, счётчика пропущенных нет."""
    _mock_pr7(_resolved_threads())
    snap = _collect_pr7()
    assert [c["id"] for c in snap["review_comments"]] == [10, 11, 12, 13]
    assert "resolved_review_comments_skipped" not in snap["counts"]


@responses.activate
def test_skip_resolved_by_no_match_reports_zero():
    _mock_pr7(_resolved_threads())
    snap = _collect_pr7(skip_resolved_by=["nobody"])
    assert [c["id"] for c in snap["review_comments"]] == [10, 11, 12, 13]
    assert snap["counts"]["resolved_review_comments_skipped"] == 0


@responses.activate
def test_skip_resolved_by_applied_before_comment_filter():
    """Автор первого комментария отсечён --comment-filter, но ответы решённого треда всё равно пропускаются."""
    _mock_pr7(_resolved_threads())
    snap = _collect_pr7(skip_resolved_by=["alice"], comment_filter=([], ["bot"]))
    assert [c["id"] for c in snap["review_comments"]] == [12, 13]


@responses.activate
def test_skip_resolved_by_with_synthetic_timeline():
    _mock_pr7(_resolved_threads(), timeline=None)
    snap = _collect_pr7(skip_resolved_by=["alice"])
    assert snap["source"]["history_unavailable"] is True
    assert [e["id"] for e in snap["timeline"] if e["type"] == "review_comment"] == [12, 13]


@responses.activate
def test_skip_resolved_by_in_light_mode():
    _mock_pr7(_resolved_threads())
    snap = _collect_pr7(light_pr=True, skip_resolved_by=["alice"])
    assert snap["mode"] == "light"
    assert [c["id"] for c in snap["review_comments"]] == [12, 13]
    assert snap["counts"] == {"files": 0, "open_review_comments": 2, "resolved_review_comments_skipped": 2}


@responses.activate
def test_e2e_skip_resolved_by_reaches_snapshot_and_llm(tmp_path, prompt_file, capsys):
    """Сквозной прогон через main(): флаг доходит до JSON на диске и до запроса к LLM."""
    out = tmp_path / "out"
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo", "empty": False})
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls", json=[_pull()])
    _mock_pr7(_resolved_threads())
    responses.add(responses.POST, LLM_URL, json={"choices": [{"message": {"content": "ok"}}]})

    code = app.main(
        [
            REPO_URL,
            "--token",
            "tok",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
            "--output-dir",
            str(out),
            "--no-stream",
            SKIP_FLAG,
            "Alice",
        ]
    )

    assert code == 0
    snap = json.loads((out / "7_abc123def456.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in snap["review_comments"]] == [12, 13]
    assert snap["counts"]["resolved_review_comments_skipped"] == 2

    llm_request = next(c.request for c in responses.calls if c.request.url == LLM_URL)
    assert "resolved by alice" not in json.loads(llm_request.body)["messages"][1]["content"]

    logged = capsys.readouterr().err
    assert "пометили решёнными: Alice" in logged
    assert "пропущено решённых комментариев 2" in logged


# --------------------------------------------------------------------------- #
#  Регрессия: логирование counts в light-режиме (KeyError 'commits')
# --------------------------------------------------------------------------- #


def test_format_counts_full_mode():
    counts = {
        "commits": 3,
        "files": 5,
        "issue_comments": 2,
        "reviews": 1,
        "review_comments": 4,
        "timeline_events": 17,
    }
    text = app.format_counts(counts)
    assert "коммитов 3" in text
    assert "файлов 5" in text
    assert "комментариев 2" in text
    assert "ревью 1" in text
    assert "событий 17" in text


def test_format_counts_light_mode():
    """light-режим отдаёт другой набор ключей — сводка не должна падать."""
    counts = {"files": 2, "open_review_comments": 3}
    text = app.format_counts(counts)
    assert "файлов 2" in text
    assert "незакрытых комментариев 3" in text
    # ключей полного режима в этом наборе нет — и это нормально
    assert "коммитов" not in text


@pytest.mark.parametrize("counts", [None, {}])
def test_format_counts_empty(counts):
    assert app.format_counts(counts) == ""


def test_format_counts_unknown_key_not_lost():
    text = app.format_counts({"files": 1, "brand_new_metric": 7})
    assert "файлов 1" in text
    assert "brand_new_metric 7" in text


@responses.activate
def test_e2e_light_pr_does_not_crash_on_counts(tmp_path, prompt_file):
    """Сквозной прогон через main() с --light-pr: раньше падал с KeyError 'commits'.

    Проверяем именно то, что упало у пользователя: логирование после записи
    JSON-файла в облегчённом режиме.
    """
    out = tmp_path / "out"
    sha = "0dca9c8787b8"
    pull = {
        "number": 54,
        "title": "Luchkin_MA_1",
        "state": "open",
        "labels": [],
        "assignees": [],
        "head": {"sha": sha, "repo": {"full_name": "owner/repo"}},
        "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
    }
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo", "empty": False})
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls", json=[pull])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/54/commits", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/54/files", json=[_file_entry()])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/54/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/54/reviews", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/54/timeline", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/54.diff", body="", content_type="text/plain")
    responses.add(responses.POST, LLM_URL, json={"choices": [{"message": {"content": "ok"}}]})

    code = app.main(
        [
            REPO_URL,
            "--token",
            "tok",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
            "--output-dir",
            str(out),
            "--light-pr",
        ]
    )

    assert code == 0
    json_path = out / f"54_{sha}.json"
    assert json_path.exists()

    snap = json.loads(json_path.read_text(encoding="utf-8"))
    assert snap["mode"] == "light"
    assert snap["counts"] == {"files": 1, "open_review_comments": 0}
    # ключей полного режима в light-снимке нет — на них и падало
    assert "commits" not in snap["counts"]


@responses.activate
def test_e2e_full_mode_still_logs_all_counts(tmp_path, prompt_file, capsys):
    """Полный режим по-прежнему показывает коммиты и события в сводке."""
    out = tmp_path / "out"
    sha = "beef99"
    pull = {
        "number": 3,
        "title": "Full",
        "state": "open",
        "labels": [],
        "assignees": [],
        "head": {"sha": sha, "repo": {"full_name": "owner/repo"}},
        "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
    }
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo", "empty": False})
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls", json=[pull])
    responses.add(
        responses.GET, f"{API}/repos/owner/repo/pulls/3/commits", json=[{"sha": sha, "commit": {"message": "x"}}]
    )
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3/files", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/3/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3/reviews", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/3/timeline", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3.diff", body="", content_type="text/plain")
    responses.add(responses.POST, LLM_URL, json={"choices": [{"message": {"content": "ok"}}]})

    code = app.main(
        [
            REPO_URL,
            "--token",
            "tok",
            "--llm-url",
            LLM_URL,
            "--model",
            "m",
            "--prompt-file",
            prompt_file,
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / f"3_{sha}.json").exists()


# --------------------------------------------------------------------------- #
#  --post-draft-review / --post-review: размещение замечаний LLM в PR
# --------------------------------------------------------------------------- #

REVIEW_SHA = "cafe0123456789"
REVIEW_FILE = "Pilipenko_Vladimir_lb3/src/main.c"
REVIEWS_URL = f"{API}/repos/owner/repo/pulls/3/reviews"

LLM_REVIEW_ANSWER = """[
  {
    "file": "Pilipenko_Vladimir_lb3/src/main.c",
    "line": "18",
    "comment_text": "Логика динамического расширения буфера повторяется здесь (чтение `text`) и позже в строках 75–85 (расширение `new`). Такой дублирующий код сложнее сопровождать: при изменении правил распределения легко ошибиться только в одном месте. Вынесите общий механизм выделения/перевыделения памяти в отдельную функцию."
  },
  {
    "file": "Pilipenko_Vladimir_lb3/src/main.c",
    "line": "56",
    "comment_text": "Здесь и в строках 61–67 предложение, содержащее `555`, помечается флагом и полностью пропускается. В постановке лабораторной работы такого правила нет, из-за этого такие предложения исчезают из вывода и счётчик `m` может стать неверным. Уберите эту фильтрацию, если она явно не требуется."
  }
]"""


def test_parse_review_comments_sample_answer():
    comments, problems = app.parse_review_comments(LLM_REVIEW_ANSWER)
    assert problems == []
    assert [(c["path"], c["new_position"]) for c in comments] == [(REVIEW_FILE, 18), (REVIEW_FILE, 56)]
    assert comments[0]["body"].startswith("Логика динамического расширения буфера")
    assert comments[1]["body"].endswith("если она явно не требуется.")


def test_parse_review_comments_skips_invalid_entries():
    answer = json.dumps(
        [
            {"file": "a.c", "line": 7, "comment_text": "ok"},
            {"line": "3", "comment_text": "no file"},
            {"file": "a.c", "line": "abc", "comment_text": "bad line"},
            {"file": "a.c", "line": 0, "comment_text": "zero line"},
            {"file": "a.c", "line": True, "comment_text": "bool line"},
            {"file": "a.c", "line": "4", "comment_text": "   "},
            "not an object",
        ]
    )
    comments, problems = app.parse_review_comments(answer)
    assert comments == [{"path": "a.c", "body": "ok", "new_position": 7}]
    assert len(problems) == 6


@pytest.mark.parametrize("answer", ["Всё хорошо!", "", '{"file": "a.c"}', '"text"'])
def test_parse_review_comments_rejects_non_list(answer):
    with pytest.raises(app.ReviewParseError):
        app.parse_review_comments(answer)


def test_parse_review_comments_empty_list():
    assert app.parse_review_comments("[]") == ([], [])


def test_review_marker_naming():
    assert app.Orchestrator.review_name(7, "abc123") == "review_7_abc123.json"


@responses.activate
def test_create_pull_review_sends_json_with_token():
    responses.add(responses.POST, REVIEWS_URL, json={"id": 42, "state": "COMMENT"})
    client = app.ForgejoClient(API, "tok")
    payload = {"event": "COMMENT", "body": "b", "commit_id": "abc", "comments": []}
    assert client.create_pull_review("owner", "repo", 3, payload)["id"] == 42
    request = responses.calls[0].request
    assert json.loads(request.body) == payload
    assert request.headers["Authorization"] == "token tok"
    assert request.headers["Content-Type"] == "application/json"


@responses.activate
def test_post_is_not_retried_on_server_error():
    responses.add(responses.POST, REVIEWS_URL, status=500, body="boom")
    client = app.ForgejoClient(API, "tok", retries=3)
    with pytest.raises(app.ForgejoError) as exc_info:
        client.create_pull_review("owner", "repo", 3, {"event": "COMMENT"})
    assert exc_info.value.status == 500
    assert len(responses.calls) == 1


def _review_argv(prompt_file, out, *extra):
    return [
        REPO_URL,
        "--token",
        "tok",
        "--llm-url",
        LLM_URL,
        "--model",
        "m",
        "--prompt-file",
        prompt_file,
        "--output-dir",
        str(out),
        *extra,
    ]


def _seed_review_run(out, answer=LLM_REVIEW_ANSWER, files=(REVIEW_FILE,)):
    """Снимок PR и ответ LLM уже лежат на диске: LLM и сбор данных в тесте не участвуют."""
    out.mkdir(exist_ok=True)
    snapshot = {"pr_number": 3, "head_sha": REVIEW_SHA, "files": [{"filename": f} for f in files]}
    (out / f"3_{REVIEW_SHA}.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (out / f"answer_3_{REVIEW_SHA}.txt").write_text(answer, encoding="utf-8")


def _mock_review_forgejo(existing_reviews=()):
    pull = {
        "number": 3,
        "title": "lb3",
        "labels": [],
        "assignees": [],
        "head": {"sha": REVIEW_SHA, "repo": {"full_name": "owner/repo"}},
    }
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo"})
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls", json=[pull])
    responses.add(responses.GET, REVIEWS_URL, json=list(existing_reviews))


def _review_posts():
    return [c for c in responses.calls if c.request.method == "POST" and c.request.url == REVIEWS_URL]


@responses.activate
def test_post_review_publishes_comment_review(tmp_path, prompt_file, capsys):
    out = tmp_path / "out"
    _seed_review_run(out)
    _mock_review_forgejo()
    responses.add(responses.POST, REVIEWS_URL, json={"id": 42, "state": "COMMENT"})

    code = app.main(_review_argv(prompt_file, out, "--post-review"))

    assert code == 0
    assert len(_review_posts()) == 1
    body = json.loads(_review_posts()[0].request.body)
    assert body["event"] == "COMMENT"
    assert body["commit_id"] == REVIEW_SHA
    assert body["body"].startswith("Автоматическая проверка (LLM: m)")
    assert [(c["path"], c["new_position"]) for c in body["comments"]] == [(REVIEW_FILE, 18), (REVIEW_FILE, 56)]
    assert body["comments"][0]["body"].startswith("Логика динамического расширения буфера")
    assert not any(c.request.url.startswith(LLM_URL) for c in responses.calls)

    marker = json.loads((out / f"review_3_{REVIEW_SHA}.json").read_text(encoding="utf-8"))
    assert marker["mode"] == "review"
    assert marker["review_id"] == 42
    assert marker["comments"] == 2
    assert "review размещено" in capsys.readouterr().err


@responses.activate
def test_post_draft_review_uses_pending_event(tmp_path, prompt_file):
    out = tmp_path / "out"
    _seed_review_run(out)
    _mock_review_forgejo()
    responses.add(responses.POST, REVIEWS_URL, json={"id": 43, "state": "PENDING"})

    code = app.main(_review_argv(prompt_file, out, "--post-draft-review"))

    assert code == 0
    assert json.loads(_review_posts()[0].request.body)["event"] == "PENDING"
    marker = json.loads((out / f"review_3_{REVIEW_SHA}.json").read_text(encoding="utf-8"))
    assert marker["mode"] == "draft"
    assert marker["review_id"] == 43


@responses.activate
def test_review_not_posted_without_flags(tmp_path, prompt_file, capsys):
    out = tmp_path / "out"
    _seed_review_run(out)
    _mock_review_forgejo()

    assert app.main(_review_argv(prompt_file, out)) == 0
    assert _review_posts() == []
    assert not (out / f"review_3_{REVIEW_SHA}.json").exists()
    assert "review размещено" not in capsys.readouterr().err


@responses.activate
def test_review_posted_after_fresh_llm_answer(tmp_path, prompt_file):
    out = tmp_path / "out"
    pull = {
        "number": 3,
        "title": "lb3",
        "state": "open",
        "labels": [],
        "assignees": [],
        "head": {"sha": REVIEW_SHA, "repo": {"full_name": "owner/repo"}},
    }
    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo"})
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls", json=[pull])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3/commits", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3/files", json=[_file_entry(filename=REVIEW_FILE)])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/3/comments", json=[])
    responses.add(responses.GET, REVIEWS_URL, json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/3/timeline", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3.diff", body="", content_type="text/plain")
    responses.add(responses.POST, LLM_URL, body=_sse(LLM_REVIEW_ANSWER), content_type="text/event-stream")
    responses.add(responses.POST, REVIEWS_URL, json={"id": 44, "state": "COMMENT"})

    code = app.main(_review_argv(prompt_file, out, "--light-pr", "--post-review"))

    assert code == 0
    assert len(_review_posts()) == 1
    assert len(json.loads(_review_posts()[0].request.body)["comments"]) == 2
    assert (out / f"review_3_{REVIEW_SHA}.json").exists()


@responses.activate
def test_review_skipped_when_marker_exists(tmp_path, prompt_file):
    out = tmp_path / "out"
    _seed_review_run(out)
    (out / f"review_3_{REVIEW_SHA}.json").write_text('{"mode": "draft", "review_id": 1}', encoding="utf-8")
    _mock_review_forgejo()

    assert app.main(_review_argv(prompt_file, out, "--post-review")) == 0
    assert _review_posts() == []


@responses.activate
def test_force_review_posts_again(tmp_path, prompt_file):
    out = tmp_path / "out"
    _seed_review_run(out)
    (out / f"review_3_{REVIEW_SHA}.json").write_text('{"mode": "review", "review_id": 1}', encoding="utf-8")
    _mock_review_forgejo()
    responses.add(responses.POST, REVIEWS_URL, json={"id": 2, "state": "COMMENT"})

    assert app.main(_review_argv(prompt_file, out, "--post-review", "--force-review")) == 0
    assert len(_review_posts()) == 1
    assert json.loads((out / f"review_3_{REVIEW_SHA}.json").read_text(encoding="utf-8"))["review_id"] == 2


@responses.activate
def test_review_drops_comments_for_files_outside_pr(tmp_path, prompt_file, capsys):
    out = tmp_path / "out"
    answer = json.dumps(
        [
            {"file": REVIEW_FILE, "line": "18", "comment_text": "реальный файл"},
            {"file": "ghost/file.c", "line": "5", "comment_text": "выдуманный файл"},
        ],
        ensure_ascii=False,
    )
    _seed_review_run(out, answer=answer)
    _mock_review_forgejo()
    responses.add(responses.POST, REVIEWS_URL, json={"id": 5, "state": "COMMENT"})

    assert app.main(_review_argv(prompt_file, out, "--post-review")) == 0
    comments = json.loads(_review_posts()[0].request.body)["comments"]
    assert [c["path"] for c in comments] == [REVIEW_FILE]
    assert "ghost/file.c" in capsys.readouterr().err


@responses.activate
def test_review_all_comments_invalid_is_failure(tmp_path, prompt_file):
    out = tmp_path / "out"
    answer = json.dumps([{"file": "ghost/file.c", "line": "5", "comment_text": "x"}])
    _seed_review_run(out, answer=answer)
    _mock_review_forgejo()

    assert app.main(_review_argv(prompt_file, out, "--post-review")) == 1
    assert _review_posts() == []
    assert not (out / f"review_3_{REVIEW_SHA}.json").exists()


@responses.activate
def test_review_empty_answer_creates_nothing(tmp_path, prompt_file):
    out = tmp_path / "out"
    _seed_review_run(out, answer="[]")
    _mock_review_forgejo()

    assert app.main(_review_argv(prompt_file, out, "--post-review")) == 0
    assert _review_posts() == []
    assert not (out / f"review_3_{REVIEW_SHA}.json").exists()


@responses.activate
def test_review_malformed_answer_is_failure(tmp_path, prompt_file):
    out = tmp_path / "out"
    _seed_review_run(out, answer="Замечаний нет, всё отлично.")
    _mock_review_forgejo()

    assert app.main(_review_argv(prompt_file, out, "--post-review")) == 1
    assert _review_posts() == []


@responses.activate
@pytest.mark.parametrize("flag", ["--post-review", "--post-draft-review"])
def test_review_blocked_by_existing_pending_review(tmp_path, prompt_file, capsys, flag):
    out = tmp_path / "out"
    _seed_review_run(out)
    _mock_review_forgejo(existing_reviews=[{"id": 77, "state": "PENDING", "user": {"login": "teacher"}}])

    assert app.main(_review_argv(prompt_file, out, flag)) == 1
    assert _review_posts() == []
    assert not (out / f"review_3_{REVIEW_SHA}.json").exists()
    err = capsys.readouterr().err
    assert "#77" in err
    assert "teacher" in err


@responses.activate
def test_review_forbidden_mentions_write_scope(tmp_path, prompt_file, capsys):
    out = tmp_path / "out"
    _seed_review_run(out)
    _mock_review_forgejo()
    responses.add(responses.POST, REVIEWS_URL, status=403, json={"message": "token scope"})

    assert app.main(_review_argv(prompt_file, out, "--post-review")) == 1
    assert len(_review_posts()) == 1
    assert "write:repository" in capsys.readouterr().err
    assert not (out / f"review_3_{REVIEW_SHA}.json").exists()


@responses.activate
def test_review_server_error_is_not_retried(tmp_path, prompt_file):
    out = tmp_path / "out"
    _seed_review_run(out)
    _mock_review_forgejo()
    responses.add(responses.POST, REVIEWS_URL, status=500, body="boom")

    assert app.main(_review_argv(prompt_file, out, "--post-review")) == 1
    assert len(_review_posts()) == 1
    assert not (out / f"review_3_{REVIEW_SHA}.json").exists()


@responses.activate
def test_review_network_error_warns_result_unknown(tmp_path, prompt_file, capsys):
    out = tmp_path / "out"
    _seed_review_run(out)
    _mock_review_forgejo()
    responses.add(responses.POST, REVIEWS_URL, body=requests.ConnectionError("timed out"))

    assert app.main(_review_argv(prompt_file, out, "--post-review")) == 1
    assert len(_review_posts()) == 1
    assert "Результат неизвестен" in capsys.readouterr().err


@responses.activate
def test_dry_run_never_posts_review(tmp_path, prompt_file):
    out = tmp_path / "out"
    _seed_review_run(out)
    _mock_review_forgejo()

    assert app.main(_review_argv(prompt_file, out, "--post-review", "--dry-run")) == 0
    assert _review_posts() == []


def test_post_flags_are_mutually_exclusive(prompt_file):
    with pytest.raises(SystemExit) as exc_info:
        app.build_parser().parse_args(_review_argv(prompt_file, ".", "--post-review", "--post-draft-review"))
    assert exc_info.value.code == 2


def test_force_review_requires_post_flag(prompt_file):
    assert app.main(_review_argv(prompt_file, ".", "--force-review")) == 2
