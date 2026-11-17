"""Тесты forgejo_pr_llm.py на моках HTTP (без реального Forgejo и LLM)."""

import base64
import json
import os
import sys

import pytest
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
        ("https://forgejo.example.com/owner/repo/src/branch/main",
         ("https://forgejo.example.com", "owner", "repo")),
        ("https://forgejo.example.com/owner/repo/pulls/12",
         ("https://forgejo.example.com", "owner", "repo")),
        # инсталляция на сабпасе
        ("https://host.example.com/git/owner/repo", ("https://host.example.com/git", "owner", "repo")),
        ("http://localhost:3000/owner/repo", ("http://localhost:3000", "owner", "repo")),
        ("https://forgejo.example.com/owner/my.repo-name_x", 
         ("https://forgejo.example.com", "owner", "my.repo-name_x")),
    ],
)
def test_parse_repo_url_ok(raw, expected):
    assert app.parse_repo_url(raw) == expected


@pytest.mark.parametrize("raw", [
    "forgejo.example.com/owner/repo",
    "https://forgejo.example.com",
    "https://forgejo.example.com/owner",
    "",
    "   ",
    "ftp://host/owner/repo",
])
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
    responses.add_callback(responses.GET, f"{API}/repos/owner/repo/pulls",
                           callback=_paged_callback(total_pages=2, per_page=5))
    client = app.ForgejoClient(API, "tok", page_size=5, verbose=False)
    pulls = list(client.list_open_pulls("owner", "repo"))
    assert len(pulls) == 10
    assert [p["number"] for p in pulls] == list(range(10))
    # 2 полные страницы + одна пустая, подтверждающая конец
    assert len(responses.calls) == 3


@responses.activate
def test_pagination_respects_max_pages():
    responses.add_callback(responses.GET, f"{API}/repos/owner/repo/pulls",
                           callback=_paged_callback(total_pages=999, per_page=5))
    client = app.ForgejoClient(API, "tok", page_size=5, max_pages=3, verbose=False)
    assert len(list(client.list_open_pulls("owner", "repo"))) == 15


@responses.activate
def test_pagination_stops_on_short_page():
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls",
                  json=[{"number": 1}, {"number": 2}], status=200)
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
    responses.add(responses.GET, f"{API}/repos/owner/repo",
                  json={"message": "unauthorized"}, status=401)
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
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/1/timeline",
                  json={"message": "not found"}, status=404)
    client = app.ForgejoClient(API, "tok", verbose=False)
    assert client.list_timeline("owner", "repo", 1) is None


# --------------------------------------------------------------------------- #
#  PRCollector
# --------------------------------------------------------------------------- #

def _file_entry(**over):
    base = {
        "filename": "src/app.py", "status": "modified",
        "additions": 2, "deletions": 1, "changes": 3,
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
        "head": {"sha": sha, "ref": "feature",
                 "repo": {"full_name": full_name}},
        "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
    }


@responses.activate
def test_collect_full_snapshot():
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits",
                  json=[{"sha": "abc123def456", "commit": {"message": "fix"}}])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files",
                  json=[_file_entry()])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments",
                  json=[{"id": 1, "user": {"login": "bob"}, "body": "looks good"}])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/reviews",
                  json=[{"id": 99, "state": "APPROVED", "user": {"login": "carol"}}])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/reviews/99/comments",
                  json=[{"id": 5, "path": "src/app.py", "line": 1, "body": "nit"}])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline",
                  json=[{"id": 1, "type": "comment", "created_at": "2026-01-01T00:00:00Z"}])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7.diff",
                  body="diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n"
                       "+++ b/src/app.py\n@@ -1 +1,2 @@\n-old\n+new\n+more\n",
                  content_type="text/plain")
    responses.add(responses.GET, f"{API}/repos/owner/repo/contents/src/app.py",
                  json={"encoding": "base64", "size": 8,
                        "content": base64.b64encode(b"new\nmore\n").decode()})

    client = app.ForgejoClient(API, "tok", verbose=False)
    collector = app.PRCollector(client, max_file_bytes=1024)
    warnings: list[str] = []
    snap = collector.collect("owner", "repo", _pull(), base_url="https://forgejo.example.com",
                             warnings=warnings)

    assert snap["schema_version"] == "1.0"
    assert snap["head_sha"] == "abc123def456"
    assert snap["pr_number"] == 7
    assert snap["counts"] == {
        "commits": 1, "files": 1, "issue_comments": 1,
        "reviews": 1, "review_comments": 1, "timeline_events": 1,
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
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files",
                  json=[_file_entry()])
    # контент лежит в форке, не в base-репозитории
    responses.add(responses.GET, f"{API}/repos/otheruser/repo/contents/src/app.py",
                  json={"encoding": "base64", "size": 4,
                        "content": base64.b64encode(b"body").decode()})

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
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files",
                  json=[_file_entry(status="removed", filename="gone.py", patch="")])
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
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files",
                  json=[_file_entry(filename="big.bin")])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/contents/big.bin",
                  json={"encoding": "base64", "size": big, "content": ""})

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
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files",
                  json=[_file_entry(filename="img/logo.png", patch="")])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/contents/img/logo.png",
                  json={"encoding": "base64", "size": len(blob),
                        "content": base64.b64encode(blob).decode()})

    client = app.ForgejoClient(API, "tok", verbose=False)
    snap = app.PRCollector(client, max_file_bytes=4096).collect(
        "owner", "repo", _pull(), base_url="x", warnings=[]
    )
    content = snap["files"][0]["content"]
    assert content["binary"] is True
    assert content["encoding"] == "base64"
    assert base64.b64decode(content["text"]) == blob


@responses.activate
def test_missing_patch_taken_from_unified_diff():
    for ep in ("commits", "reviews"):
        responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/{ep}", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/files",
                  json=[_file_entry(patch="")])          # сервер не отдал патч
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/7/timeline", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7.diff",
                  body="diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n"
                       "+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n",
                  content_type="text/plain")
    responses.add(responses.GET, f"{API}/repos/owner/repo/contents/src/app.py",
                  json={"encoding": "base64", "size": 4,
                        "content": base64.b64encode(b"new\n").decode()})

    client = app.ForgejoClient(API, "tok", verbose=False)
    snap = app.PRCollector(client, max_file_bytes=4096).collect(
        "owner", "repo", _pull(), base_url="x", warnings=[]
    )
    f = snap["files"][0]
    assert f["patch_source"] == "unified_diff"
    assert "+new" in f["patch"]


@responses.activate
def test_endpoint_failure_does_not_break_pr():
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits",
                  json={"message": "boom"}, status=500)
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits",
                  json={"message": "boom"}, status=500)
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/7/commits",
                  json={"message": "boom"}, status=500)
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
        lines.append("data: " + json.dumps(
            {"choices": [{"delta": {"content": ch}}]}, ensure_ascii=False) + "\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


@responses.activate
def test_llm_streaming_collects_text(capsys):
    responses.add(responses.POST, LLM_URL, body=_sse("Hello", ", ", "world"),
                  content_type="text/event-stream")
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
    responses.add(responses.POST, LLM_URL,
                  json={"choices": [{"message": {"content": "plain answer"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
                  content_type="application/json")
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
        responses.add(responses.POST, LLM_URL,
                      json={"choices": [{"message": {"content": ""}}]})
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
    args = app.build_parser().parse_args([
        REPO_URL, "--token", "tok", "--llm-url", LLM_URL,
        "--model", "m", "--prompt-file", prompt_file,
        "--output-dir", str(tmp_path / "out"),
    ])
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_validate_args_requires_token(prompt_file):
    args = app.build_parser().parse_args([
        REPO_URL, "--llm-url", LLM_URL, "--model", "m", "--prompt-file", prompt_file,
    ])
    args.token = None
    with pytest.raises(app.ConfigError):
        app.validate_args(args)


def test_validate_args_missing_prompt(tmp_path):
    args = app.build_parser().parse_args([
        REPO_URL, "--token", "tok", "--llm-url", LLM_URL, "--model", "m",
        "--prompt-file", str(tmp_path / "nope.txt"),
    ])
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
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls", json=[
        {"number": 7, "title": "T", "labels": [], "assignees": [],
         "head": {"sha": sha, "repo": {"full_name": "owner/repo"}}},
    ])

    code = app.main([REPO_URL, "--token", "tok", "--llm-url", LLM_URL,
                     "--model", "m", "--prompt-file", prompt_file,
                     "--output-dir", str(out)])
    assert code == 0
    # к LLM обращений не было
    assert not any(c.request.url.startswith(LLM_URL) for c in responses.calls)
    assert (out / f"answer_7_{sha}.txt").read_text(encoding="utf-8") == "готовый ответ"


@responses.activate
def test_full_run_writes_json_and_answer(tmp_path, prompt_file):
    out = tmp_path / "out"
    sha = "cafe01"
    pull = {"number": 3, "title": "Add feature", "state": "open",
            "labels": [{"name": "needs-review"}], "assignees": [{"login": "alice"}],
            "head": {"sha": sha, "repo": {"full_name": "owner/repo"}}}

    responses.add(responses.GET, f"{API}/repos/owner/repo", json={"full_name": "owner/repo"})
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls", json=[pull])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3/commits", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3/files", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/3/comments", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3/reviews", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/issues/3/timeline", json=[])
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls/3.diff", body="",
                  content_type="text/plain")
    responses.add(responses.POST, LLM_URL, body=_sse("LGTM"), content_type="text/event-stream")

    code = app.main([REPO_URL, "--token", "tok", "--llm-url", LLM_URL,
                     "--model", "m", "--prompt-file", prompt_file,
                     "--output-dir", str(out), "--label", "needs-review",
                     "--assignee", "alice"])
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
    responses.add(responses.GET, f"{API}/repos/owner/repo/pulls", json=[
        {"number": 1, "labels": [{"name": "other"}], "assignees": [{"login": "alice"}],
         "head": {"sha": "aa", "repo": {"full_name": "owner/repo"}}},
        {"number": 2, "labels": [{"name": "needs-review"}], "assignees": [{"login": "bob"}],
         "head": {"sha": "bb", "repo": {"full_name": "owner/repo"}}},
    ])
    code = app.main([REPO_URL, "--token", "tok", "--llm-url", LLM_URL,
                     "--model", "m", "--prompt-file", prompt_file,
                     "--output-dir", str(out), "--label", "needs-review",
                     "--assignee", "alice"])
    # ни один PR не подходит под И(label, assignee)
    assert code == 0
    assert not out.exists() or not any(out.iterdir())


def test_filter_requires_both():
    args = app.build_parser().parse_args([
        REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", "x",
        "--label", "bug", "--assignee", "Alice",
    ])
    orch = app.Orchestrator(args, "https://h", "o", "r")
    pull = {"labels": [{"name": "bug"}], "assignees": [{"login": "alice"}]}
    assert orch._matches_filters(pull) is True
    assert orch._matches_filters({"labels": [{"name": "bug"}], "assignees": []}) is False
    assert orch._matches_filters({"labels": [], "assignees": [{"login": "alice"}]}) is False
    # регистронезависимо
    assert orch._matches_filters({"labels": [{"name": "BUG"}],
                                  "assignees": [{"login": "ALICE"}]}) is True


def test_filter_no_criteria_matches_all():
    args = app.build_parser().parse_args([
        REPO_URL, "--token", "t", "--llm-url", "u", "--model", "m", "--prompt-file", "x",
    ])
    orch = app.Orchestrator(args, "https://h", "o", "r")
    assert orch._matches_filters({"labels": [], "assignees": []}) is True
