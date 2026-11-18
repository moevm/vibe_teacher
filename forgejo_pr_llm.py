#!/usr/bin/env python3
"""forgejo_pr_llm.py

Выгрузка открытых Pull Request из Forgejo и проверка их локальной LLM
(OpenAI-совместимый endpoint).

Использование:
    python3 forgejo_pr_llm.py https://forgejo.example.com/owner/repo \
        --token $FORGEJO_TOKEN \
        --llm-url http://localhost:11434/v1/chat/completions \
        --model qwen2.5-coder:14b \
        --prompt-file prompt.txt

Выходные файлы (в --output-dir, по умолчанию текущая директория):
    <номерPR>_<sha>.json        — полный снимок PR
    answer_<номерPR>_<sha>.txt  — ответ LLM

Коды возврата: 0 — успех, 1 — были ошибки по отдельным PR, 2 — ошибка конфигурации.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator
from urllib.parse import quote, urlparse

import requests

__version__ = "1.2.0"

DEFAULT_LLM_TIMEOUT = 900  # 15 минут
DEFAULT_PAGE_SIZE = 50
DEFAULT_MAX_FILE_BYTES = 512 * 1024
DEFAULT_MAX_PROMPT_BYTES = 1024 * 1024
MAX_UNIFIED_DIFF_BYTES = 8 * 1024 * 1024

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

# Сегменты веб-путей, по которым обрезаем «хвост» URL (…/owner/repo/src/branch/main)
_TRAILING_MARKERS = {
    "src",
    "issues",
    "pulls",
    "wiki",
    "settings",
    "commits",
    "releases",
    "activity",
    "branches",
    "tags",
    "projects",
}


# --------------------------------------------------------------------------- #
#  Вспомогательное
# --------------------------------------------------------------------------- #


class ConfigError(Exception):
    """Ошибка конфигурации/аргументов — код возврата 2."""


class ForgejoError(Exception):
    def __init__(self, message: str, status: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status = status
        self.url = url


class LLMError(Exception):
    def __init__(self, message: str, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


def log(msg: str = "") -> None:
    """Служебные сообщения — в stderr, чтобы stdout оставался чистым для ответа LLM."""
    print(msg, file=sys.stderr, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human_bytes(n: int) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} ГБ"


def parse_repo_url(raw: str) -> tuple[str, str, str]:
    """https://host/[subpath/]owner/repo[/...] -> (base_url, owner, repo).

    base_url — корень инсталляции вместе с возможным сабпасом (https://host/git),
    owner/repo — два последних значимых сегмента пути.
    """
    value = raw.strip()
    if not value:
        raise ConfigError("Пустой URL репозитория.")

    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ConfigError(f"URL репозитория должен начинаться с http:// или https:// — получено: {raw!r}")

    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    segments = [s for s in path.split("/") if s]

    # Обрезаем хвост веб-страницы: /owner/repo/src/branch/main и подобное
    for i, seg in enumerate(segments):
        if seg in _TRAILING_MARKERS and i >= 2:
            segments = segments[:i]
            break

    if len(segments) < 2:
        raise ConfigError(f"В URL не найдены owner/repo. Ожидается https://host/owner/repo — получено: {raw!r}")

    owner, repo = segments[-2], segments[-1]
    prefix = "/".join(segments[:-2])
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    if prefix:
        base_url = f"{base_url}/{prefix}"

    if not re.fullmatch(r"[A-Za-z0-9._-]+", repo):
        raise ConfigError(f"Некорректное имя репозитория: {repo!r}")

    return base_url.rstrip("/"), owner, repo


def atomic_write_text(path: str, text: str) -> None:
    """Пишем через временный файл, чтобы прерванный запуск не оставил огрызок."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False))


def safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", value or "unknown")[:120]


def parse_label_filters(expressions: Iterable[str] | str | None) -> tuple[list[str], list[str]]:
    """Разбирает выражения --label в (обязательные, запрещённые) метки.

    Несколько меток можно задать через запятую в одном значении и/или
    повторив флаг. Префикс ``-`` означает «метки быть не должно»::

        "needs-review,bug,-wip"  ->  (["needs-review", "bug"], ["wip"])

    Экранирование: ``\\-label`` или ``+label`` задают метку, начинающуюся
    с дефиса. Пустые сегменты игнорируются.
    """
    if expressions is None:
        raw_values: list[str] = []
    elif isinstance(expressions, str):
        raw_values = [expressions]
    else:
        raw_values = [e for e in expressions if e is not None]

    required: list[str] = []
    forbidden: list[str] = []

    for value in raw_values:
        if not value.strip():
            raise ConfigError("Пустое значение --label: укажите метки, например --label=bug,-wip.")
        for raw in value.split(","):
            token = raw.strip()
            if not token:
                continue  # пустые сегменты между запятыми допустимы

            if token.startswith("\\") or token.startswith("+"):
                token, target = token[1:], required
            elif token.startswith("-"):
                token, target = token[1:], forbidden
            else:
                target = required

            token = token.strip()
            if not token:
                raise ConfigError(
                    f"Пустое имя метки в --label: {value!r}. Метку, начинающуюся с дефиса, экранируйте так: '\\-имя'."
                )
            target.append(token)

    if raw_values and not (required or forbidden):
        raise ConfigError("В --label не найдено ни одной метки. Пример корректного значения: --label=bug,-wip.")

    # Дубликаты схлопываем, сохраняя порядок ввода
    required = list(dict.fromkeys(required))
    forbidden = list(dict.fromkeys(forbidden))

    both = sorted(set(required) & set(forbidden), key=str.lower)
    if both:
        raise ConfigError("Метка одновременно требуется и запрещена: " + ", ".join(both))
    return required, forbidden


def format_label_filters(required: list[str], forbidden: list[str]) -> str:
    return ", ".join(list(required) + [f"-{name}" for name in forbidden])


def split_unified_diff(diff_text: str) -> dict[str, str]:
    """Режем общий diff на фрагменты по файлам.

    Ключ — путь после правки (b/...), т.к. именно он совпадает с полем
    ``filename`` в ответе /pulls/{index}/files. Заголовок ``+++`` точнее
    строки ``diff --git`` (там путь может быть усечён), поэтому при наличии
    ``+++`` переключаемся на него.
    """
    chunks: dict[str, str] = {}
    key: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if key is not None and buffer:
            chunks.setdefault(key, "".join(buffer))

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            flush()
            buffer = [line]
            key = _key_from_diff_git(line)
            continue
        if key is None:
            continue
        buffer.append(line)
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target and target != "/dev/null":
                new_key = target[2:] if target.startswith("b/") else target
                new_key = new_key.strip('"')
                if new_key != key:
                    flush()
                    buffer = []
                    key = new_key
    flush()
    return chunks


def _key_from_diff_git(line: str) -> str | None:
    m = re.match(r'^diff --git (?:"?a/(?P<a>.+?)"? "?b/(?P<b>.+?)"?)\s*$', line.rstrip("\n"))
    if not m:
        return None
    return m.group("b")


def _shorten(text: str | None, limit: int = 400) -> str:
    if not text:
        return "<пустой ответ>"
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}…"


# --------------------------------------------------------------------------- #
#  Клиент Forgejo API
# --------------------------------------------------------------------------- #


class ForgejoClient:
    """Тонкая обёртка над REST API Forgejo (унаследован от Gitea)."""

    def __init__(
        self,
        api_base: str,
        token: str,
        *,
        timeout: float = 60.0,
        retries: int = 3,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int = 500,
        verbose: bool = True,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.retries = max(1, retries)
        self.page_size = max(1, page_size)
        self.max_pages = max(1, max_pages)
        self.timeout = timeout
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {token}",
                "Accept": "application/json",
                "User-Agent": f"forgejo-pr-llm/{__version__}",
            }
        )

    # -- низкий уровень ----------------------------------------------------- #

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        allow_404: bool = False,
        raw: bool = False,
    ) -> Any:
        url = f"{self.api_base}{path}"
        last_error: str = "неизвестная ошибка"

        for attempt in range(self.retries):
            try:
                resp = self.session.request(method, url, params=params, timeout=(15, self.timeout))
            except requests.RequestException as exc:
                last_error = f"сетевая ошибка: {exc}"
                if attempt + 1 < self.retries:
                    self._backoff(attempt, f"{last_error}; повтор")
                    continue
                raise ForgejoError(last_error, url=url) from exc

            if resp.status_code in RETRY_STATUS:
                last_error = f"HTTP {resp.status_code}"
                if attempt + 1 < self.retries:
                    self._backoff(attempt, f"HTTP {resp.status_code}; повтор", resp)
                    continue
                raise ForgejoError(
                    f"Сервер вернул {resp.status_code} после {self.retries} попыток: {_shorten(resp.text)}",
                    status=resp.status_code,
                    url=url,
                )

            if resp.status_code == 404 and allow_404:
                return None

            if resp.status_code >= 400:
                raise ForgejoError(self._explain(resp), status=resp.status_code, url=url)

            if raw:
                return resp.text
            if not resp.content:
                return None
            try:
                return resp.json()
            except ValueError as exc:
                raise ForgejoError(f"Ответ не является JSON: {_shorten(resp.text)}", url=url) from exc

        raise ForgejoError(last_error, url=url)

    @staticmethod
    def _explain(resp: requests.Response) -> str:
        body = _shorten(resp.text)
        if resp.status_code == 401:
            return (
                f"401 Unauthorized — токен не принят. Проверьте значение --token "
                f"и срок его действия. Ответ сервера: {body}"
            )
        if resp.status_code == 403:
            return (
                f"403 Forbidden — токен принят, но прав недостаточно. "
                f"Нужны scope 'read:repository' (и 'read:issue' для комментариев). "
                f"Ответ сервера: {body}"
            )
        return f"HTTP {resp.status_code} {resp.reason}: {body}"

    def _backoff(self, attempt: int, why: str, resp: requests.Response | None = None) -> None:
        delay = min(2 ** (attempt + 1), 30)
        retry_after = resp.headers.get("Retry-After") if resp is not None else None
        if retry_after and retry_after.strip().isdigit():
            delay = min(int(retry_after.strip()), 60)
        if self.verbose:
            log(f"    ! {why} через {delay} с")
        time.sleep(delay)

    def _paginate(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> Iterator[dict[str, Any]]:
        page = 1
        while page <= self.max_pages:
            page_params = dict(params or {})
            page_params.update({"page": page, "limit": self.page_size})
            chunk = self._request("GET", path, params=page_params, allow_404=allow_404)
            if allow_404 and chunk is None:
                return
            if not chunk:
                return
            if isinstance(chunk, dict):
                chunk = [chunk]
            for item in chunk:
                yield item
            if len(chunk) < self.page_size:
                return
            page += 1

    # -- репозиторий и PR --------------------------------------------------- #

    def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        return self._request("GET", f"/repos/{owner}/{repo}")

    def list_open_pulls(self, owner: str, repo: str) -> Iterator[dict[str, Any]]:
        return self._paginate(f"/repos/{owner}/{repo}/pulls", params={"state": "open"})

    def get_pull(self, owner: str, repo: str, index: int) -> dict[str, Any]:
        return self._request("GET", f"/repos/{owner}/{repo}/pulls/{index}")

    def list_pull_commits(self, owner: str, repo: str, index: int) -> list[dict[str, Any]]:
        return list(self._paginate(f"/repos/{owner}/{repo}/pulls/{index}/commits"))

    def list_pull_files(self, owner: str, repo: str, index: int) -> list[dict[str, Any]]:
        return list(self._paginate(f"/repos/{owner}/{repo}/pulls/{index}/files"))

    def get_pull_diff(self, owner: str, repo: str, index: int) -> str | None:
        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{index}.diff",
            allow_404=True,
            raw=True,
        )

    def list_issue_comments(self, owner: str, repo: str, index: int) -> list[dict[str, Any]]:
        return list(self._paginate(f"/repos/{owner}/{repo}/issues/{index}/comments"))

    def list_pull_reviews(self, owner: str, repo: str, index: int) -> list[dict[str, Any]]:
        return list(self._paginate(f"/repos/{owner}/{repo}/pulls/{index}/reviews"))

    def list_review_comments(self, owner: str, repo: str, index: int, review_id: int) -> list[dict[str, Any]]:
        return list(self._paginate(f"/repos/{owner}/{repo}/pulls/{index}/reviews/{review_id}/comments"))

    def list_timeline(self, owner: str, repo: str, index: int) -> list[dict[str, Any]] | None:
        """Timeline есть не во всех версиях Forgejo — None означает «эндпоинт недоступен»."""
        path = f"/repos/{owner}/{repo}/issues/{index}/timeline"
        collected: list[dict[str, Any]] = []
        page = 1
        while page <= self.max_pages:
            chunk = self._request(
                "GET",
                path,
                params={"page": page, "limit": self.page_size},
                allow_404=True,
            )
            if chunk is None:
                # 404 на первой странице — эндпоинта нет; на следующих — просто конец
                return None if page == 1 else collected
            if not chunk:
                return collected
            if isinstance(chunk, dict):
                chunk = [chunk]
            collected.extend(chunk)
            if len(chunk) < self.page_size:
                return collected
            page += 1
        return collected

    # -- файлы -------------------------------------------------------------- #

    def get_file_content(self, repo_full_name: str, path: str, ref: str) -> dict[str, Any] | None:
        """Содержимое файла на конкретном коммите. None — файл не найден (404)."""
        quoted = quote(path, safe="/")
        data = self._request(
            "GET",
            f"/repos/{repo_full_name}/contents/{quoted}",
            params={"ref": ref},
            allow_404=True,
        )
        if data is None:
            return None
        if isinstance(data, list):
            return None  # путь оказался каталогом
        return data

    def get_raw_file(self, repo_full_name: str, path: str, ref: str) -> bytes | None:
        quoted = quote(path, safe="/")
        return self._request(
            "GET",
            f"/repos/{repo_full_name}/raw/{quoted}",
            params={"ref": ref},
            allow_404=True,
            raw=True,
        )


# --------------------------------------------------------------------------- #
#  Сборка снимка PR
# --------------------------------------------------------------------------- #

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".tiff",
    ".psd",
    ".pdf",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".tar",
    ".jar",
    ".war",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".o",
    ".a",
    ".class",
    ".pyc",
    ".wasm",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".wav",
    ".flac",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".sqlite",
    ".db",
}


def _looks_binary(path: str, data: bytes) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext in BINARY_EXTENSIONS:
        return True
    if b"\x00" in data[:8192]:
        return True
    try:
        data[:8192].decode("utf-8")
    except UnicodeDecodeError as exc:
        # Обрывок многобайтового символа на границе буфера — не признак бинарника
        if exc.start >= 8192 - 4 or exc.reason == "unexpected end of data":
            return False
        return True
    return False


class PRCollector:
    """Собирает полный снимок одного PR в один JSON-документ."""

    def __init__(
        self,
        client: ForgejoClient,
        *,
        max_file_bytes: int,
        max_diff_bytes: int = MAX_UNIFIED_DIFF_BYTES,
        light_pr: bool = False,
        comment_filter: tuple[list[str], list[str]] | None = None,
    ) -> None:
        self.client = client
        self.max_file_bytes = max_file_bytes
        self.max_diff_bytes = max_diff_bytes
        self.light_pr = light_pr
        self.comment_filter = comment_filter

    def collect(
        self,
        owner: str,
        repo: str,
        pull: dict[str, Any],
        *,
        base_url: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        index = pull["number"]
        errors: list[dict[str, Any]] = []

        def guard(label: str, fn, default):
            try:
                return fn()
            except ForgejoError as exc:
                errors.append(
                    {
                        "endpoint": label,
                        "status": exc.status,
                        "message": str(exc),
                    }
                )
                warnings.append(f"PR #{index}: не удалось получить {label} ({exc.status})")
                return default

        commits = guard("commits", lambda: self.client.list_pull_commits(owner, repo, index), [])
        files = guard("files", lambda: self.client.list_pull_files(owner, repo, index), [])
        issue_comments = guard(
            "issue_comments",
            lambda: self.client.list_issue_comments(owner, repo, index),
            [],
        )
        reviews = guard("reviews", lambda: self.client.list_pull_reviews(owner, repo, index), [])
        timeline = guard("timeline", lambda: self.client.list_timeline(owner, repo, index), None)

        review_comments = self._collect_review_comments(owner, repo, index, reviews, errors, warnings)

        # Фильтрация комментариев по авторам (--comment-filter)
        if self.comment_filter is not None:
            issue_comments = self._filter_comments(issue_comments, self.comment_filter)
            review_comments = self._filter_comments(review_comments, self.comment_filter)

        head_sha = (pull.get("head") or {}).get("sha") or ""
        head_repo = (pull.get("head") or {}).get("repo") or {}
        head_full_name = head_repo.get("full_name") or f"{owner}/{repo}"

        # Облегчённый режим: только файлы и незакрытые review-комментарии
        if self.light_pr:
            return self._collect_light(
                pull,
                index,
                owner,
                repo,
                base_url,
                head_sha,
                head_full_name,
                files,
                review_comments,
                errors,
                warnings,
            )

        # Патчи, обрезанные сервером, добираем из общего .diff
        diff_text = guard("diff", lambda: self.client.get_pull_diff(owner, repo, index), None)
        diff_parts: dict[str, str] = {}
        if diff_text:
            diff_parts = split_unified_diff(diff_text)
            if len(diff_text.encode("utf-8", "replace")) > self.max_diff_bytes:
                warnings.append(f"PR #{index}: дифф больше {human_bytes(self.max_diff_bytes)}, сохранён частично")
                diff_text = diff_text[: self.max_diff_bytes]
                warnings.append(f"PR #{index}: unified_diff обрезан до {human_bytes(len(diff_text))}")

        enriched_files = [
            self._enrich_file(owner, repo, index, head_full_name, head_sha, f, diff_parts, warnings) for f in files
        ]

        history_unavailable = timeline is None
        if history_unavailable:
            timeline = self._synthetic_timeline(issue_comments, reviews, review_comments)
            warnings.append(f"PR #{index}: эндпоинт timeline недоступен, собрана синтетическая история")

        return {
            "schema_version": "1.0",
            "fetched_at": utc_now(),
            "generator": f"forgejo-pr-llm/{__version__}",
            "source": {
                "base_url": base_url,
                "owner": owner,
                "repo": repo,
                "history_unavailable": history_unavailable,
            },
            "pr_number": index,
            "head_sha": head_sha,
            "counts": {
                "commits": len(commits),
                "files": len(enriched_files),
                "issue_comments": len(issue_comments),
                "reviews": len(reviews),
                "review_comments": len(review_comments),
                "timeline_events": len(timeline),
            },
            "pull_request": pull,
            "commits": commits,
            "files": enriched_files,
            "unified_diff": {
                "text": diff_text,
                "truncated": bool(diff_text) and len(diff_text.encode("utf-8", "replace")) >= self.max_diff_bytes,
            },
            "issue_comments": issue_comments,
            "reviews": reviews,
            "review_comments": review_comments,
            "timeline": timeline,
            "fetch_errors": errors,
        }

    # -- детали ------------------------------------------------------------- #

    def _collect_review_comments(
        self,
        owner: str,
        repo: str,
        index: int,
        reviews: list[dict[str, Any]],
        errors: list[dict[str, Any]],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for review in reviews:
            rid = review.get("id")
            if rid is None:
                continue
            try:
                comments = self.client.list_review_comments(owner, repo, index, rid)
            except ForgejoError as exc:
                errors.append(
                    {
                        "endpoint": f"reviews/{rid}/comments",
                        "status": exc.status,
                        "message": str(exc),
                    }
                )
                warnings.append(f"PR #{index}: не удалось получить комментарии ревью {rid}")
                continue
            for c in comments:
                c.setdefault("_review_id", rid)
                collected.append(c)
        return collected

    @staticmethod
    def _filter_comments(
        comments: list[dict[str, Any]],
        filter_spec: tuple[list[str], list[str]],
    ) -> list[dict[str, Any]]:
        """Оставляет только комментарии от заданных авторов.

        required — оставить только этих (ИЛИ); forbidden — исключить этих.
        """
        required, forbidden = filter_spec
        req_set = {u.lower() for u in required}
        forb_set = {u.lower() for u in forbidden}
        result: list[dict[str, Any]] = []
        for c in comments:
            user = c.get("user") or {}
            login = (user.get("login") or "").lower() if isinstance(user, dict) else ""
            if forb_set and login in forb_set:
                continue
            if req_set and login not in req_set:
                continue
            result.append(c)
        return result

    def _collect_light(
        self,
        pull: dict[str, Any],
        index: int,
        owner: str,
        repo: str,
        base_url: str,
        head_sha: str,
        head_full_name: str,
        files: list[dict[str, Any]],
        review_comments: list[dict[str, Any]],
        errors: list[dict[str, Any]],
        warnings: list[str],
    ) -> dict[str, Any]:
        """Облегчённый снимок: файлы с патчами + незакрытые inline-комментарии.

        Не включает: полное содержимое файлов, коммиты, timeline, общий дифф.
        Незакрытые = review-комментарии у которых resolved_at is None.
        """
        diff_text: str | None = None
        try:
            diff_text = self.client.get_pull_diff(owner, repo, index)
        except ForgejoError as exc:
            errors.append({"endpoint": "diff", "status": exc.status, "message": str(exc)})
            warnings.append(f"PR #{index}: не удалось получить дифф ({exc.status})")

        diff_parts: dict[str, str] = {}
        if diff_text:
            diff_parts = split_unified_diff(diff_text)

        light_files: list[dict[str, Any]] = []
        for f in files:
            fname = f.get("filename") or ""
            patch = f.get("patch") or ""
            patch_source = "api" if patch else None
            if not patch and fname in diff_parts:
                patch = diff_parts[fname]
                patch_source = "unified_diff"
            light_files.append(
                {
                    "filename": fname,
                    "status": f.get("status"),
                    "additions": f.get("additions"),
                    "deletions": f.get("deletions"),
                    "changes": f.get("changes"),
                    "patch": patch,
                    "patch_source": patch_source,
                }
            )

        # Оставляем только незакрытые (resolved_at is None) review-комментарии
        open_review_comments: list[dict[str, Any]] = []
        for rc in review_comments:
            if rc.get("resolved_at") is not None:
                continue
            user_obj = rc.get("user") or {}
            open_review_comments.append(
                {
                    "id": rc.get("id"),
                    "user": user_obj.get("login") if isinstance(user_obj, dict) else None,
                    "path": rc.get("path"),
                    "line": rc.get("line"),
                    "original_line": rc.get("original_line"),
                    "diff_hunk": rc.get("diff_hunk"),
                    "body": rc.get("body"),
                    "created_at": rc.get("created_at"),
                    "review_id": rc.get("_review_id"),
                }
            )

        user_obj = pull.get("user") or {}
        return {
            "schema_version": "1.0-light",
            "mode": "light",
            "fetched_at": utc_now(),
            "generator": f"forgejo-pr-llm/{__version__}",
            "source": {"base_url": base_url, "owner": owner, "repo": repo},
            "pr_number": index,
            "head_sha": head_sha,
            "counts": {
                "files": len(light_files),
                "open_review_comments": len(open_review_comments),
            },
            "pull_request": {
                "number": pull.get("number"),
                "title": pull.get("title"),
                "state": pull.get("state"),
                "user": user_obj.get("login") if isinstance(user_obj, dict) else None,
                "labels": [lbl.get("name") for lbl in (pull.get("labels") or []) if isinstance(lbl, dict)],
                "head": pull.get("head"),
                "base": pull.get("base"),
            },
            "files": light_files,
            "review_comments": open_review_comments,
            "fetch_errors": errors,
        }

    def _enrich_file(
        self,
        owner: str,
        repo: str,
        index: int,
        head_full_name: str,
        head_sha: str,
        entry: dict[str, Any],
        diff_parts: dict[str, str],
        warnings: list[str],
    ) -> dict[str, Any]:
        item = dict(entry)
        path = item.get("filename") or ""
        status = item.get("status") or "modified"

        # Патч: если сервер обрезал или не отдал — берём из общего diff
        if not item.get("patch") and path in diff_parts:
            item["patch"] = diff_parts[path]
            item["patch_source"] = "unified_diff"
        elif item.get("patch"):
            item["patch_source"] = "api"

        item["content"] = self._fetch_content(owner, repo, index, head_full_name, head_sha, path, status, warnings)
        return item

    def _fetch_content(
        self,
        owner: str,
        repo: str,
        index: int,
        head_full_name: str,
        head_sha: str,
        path: str,
        status: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        base = {
            "encoding": None,
            "text": None,
            "size": None,
            "binary": False,
            "truncated": False,
            "skipped_reason": None,
        }
        if status == "removed":
            return {**base, "skipped_reason": "deleted_in_pr"}
        if not path or not head_sha:
            return {**base, "skipped_reason": "no_head_sha"}

        # У форков объекты лежат в head-репозитории, а не в base
        for repo_name in (head_full_name, f"{owner}/{repo}"):
            try:
                meta = self.client.get_file_content(repo_name, path, head_sha)
            except ForgejoError as exc:
                if exc.status == 403:
                    return {**base, "skipped_reason": "forbidden"}
                continue
            if meta is None:
                continue

            size = meta.get("size")
            if isinstance(size, int) and size > self.max_file_bytes:
                warnings.append(
                    f"PR #{index}: файл {path} ({human_bytes(size)}) больше лимита "
                    f"{human_bytes(self.max_file_bytes)} — содержимое не сохранено"
                )
                return {**base, "size": size, "skipped_reason": "too_large", "encoding": meta.get("encoding")}

            payload = meta.get("content")
            if payload is None:
                raw = None
                try:
                    raw = self.client.get_raw_file(repo_name, path, head_sha)
                except ForgejoError:
                    raw = None
                if raw is None:
                    return {**base, "size": size, "skipped_reason": "no_content"}
                data = raw
            else:
                try:
                    data = base64.b64decode(payload)
                except (binascii.Error, ValueError):
                    return {**base, "size": size, "skipped_reason": "bad_base64"}

            if _looks_binary(path, data):
                return {
                    **base,
                    "encoding": "base64",
                    "text": base64.b64encode(data).decode("ascii"),
                    "size": len(data),
                    "binary": True,
                }

            truncated = len(data) > self.max_file_bytes
            if truncated:
                data = data[: self.max_file_bytes]
            text = data.decode("utf-8", errors="replace")
            return {
                **base,
                "encoding": "utf-8",
                "text": text,
                "size": len(data),
                "binary": False,
                "truncated": truncated,
            }

        return {**base, "skipped_reason": "not_found"}

    @staticmethod
    def _synthetic_timeline(
        issue_comments: list[dict[str, Any]],
        reviews: list[dict[str, Any]],
        review_comments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for c in issue_comments:
            events.append(
                {
                    "type": "comment",
                    "id": c.get("id"),
                    "user": c.get("user"),
                    "created_at": c.get("created_at"),
                    "body": c.get("body"),
                    "_synthetic": True,
                }
            )
        for r in reviews:
            events.append(
                {
                    "type": "review",
                    "id": r.get("id"),
                    "user": r.get("user"),
                    "created_at": r.get("submitted_at") or r.get("created_at"),
                    "body": r.get("body"),
                    "state": r.get("state"),
                    "commit_id": r.get("commit_id"),
                    "_synthetic": True,
                }
            )
        for rc in review_comments:
            events.append(
                {
                    "type": "review_comment",
                    "id": rc.get("id"),
                    "user": rc.get("user"),
                    "created_at": rc.get("created_at"),
                    "body": rc.get("body"),
                    "path": rc.get("path"),
                    "line": rc.get("line"),
                    "_synthetic": True,
                }
            )
        events.sort(key=lambda e: e.get("created_at") or "")
        return events


# --------------------------------------------------------------------------- #
#  Клиент LLM (OpenAI-совместимый chat/completions)
# --------------------------------------------------------------------------- #


class LLMClient:
    """Запрос к локальной модели с ожиданием завершения и выводом хода работы."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: float = DEFAULT_LLM_TIMEOUT,
        retries: int = 3,
        show_stream: bool = True,
        max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
        api_token: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.retries = max(1, retries)
        self.show_stream = show_stream
        self.max_prompt_bytes = max_prompt_bytes
        self.api_token = (api_token or "").strip() or None
        self.session = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
            "User-Agent": f"forgejo-pr-llm/{__version__}",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        self.session.headers.update(headers)

    def build_messages(self, system_prompt: str, pr_json: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Ниже — данные Pull Request в формате JSON. "
                    "Проанализируй их согласно инструкции выше.\n\n"
                    f"```json\n{pr_json}\n```"
                ),
            },
        ]

    def ask(self, messages: list[dict[str, str]], *, label: str = "") -> str:
        payload_size = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        if payload_size > self.max_prompt_bytes:
            log(
                f"    ! ВНИМАНИЕ: запрос {human_bytes(payload_size)} превышает лимит "
                f"{human_bytes(self.max_prompt_bytes)} — модель может обрезать контекст "
                f"или вернуть ошибку"
            )

        last_error: Exception | None = None
        stream_rejected = False
        for attempt in range(self.retries):
            try:
                return self._ask_stream(messages, label=label)
            except LLMError as exc:
                if not exc.retryable:
                    # Сервер явно не умеет stream:true — это не сбой, а несовместимость
                    if exc.status in (400, 404, 405, 422, 501):
                        stream_rejected = True
                        break
                    raise
                last_error = exc
                if attempt + 1 < self.retries:
                    delay = min(2 ** (attempt + 1), 30)
                    log(f"    ! {exc} — повтор через {delay} с")
                    time.sleep(delay)
                continue

        # Стриминг недоступен — пробуем обычный режим
        if stream_rejected:
            log("    · сервер не поддерживает stream:true, запрашиваю целиком")
        else:
            log("    · стриминг не удался, переключаюсь на обычный режим")
        for attempt in range(self.retries):
            try:
                return self._ask_blocking(messages)
            except LLMError as exc:
                last_error = exc
                if not exc.retryable or attempt + 1 >= self.retries:
                    raise
                delay = min(2 ** (attempt + 1), 30)
                log(f"    ! {exc} — повтор через {delay} с")
                time.sleep(delay)

        raise last_error or LLMError("не удалось получить ответ модели")

    # -- реализация --------------------------------------------------------- #

    def _ask_stream(self, messages: list[dict[str, str]], *, label: str = "") -> str:
        body = {"model": self.model, "messages": messages, "stream": True}
        started = time.monotonic()
        chunks: list[str] = []
        usage: dict[str, Any] | None = None

        try:
            resp = self.session.post(self.endpoint, json=body, timeout=(10, self.timeout), stream=True)
        except requests.Timeout as exc:
            where = f" ({label})" if label else ""
            raise LLMError(f"таймаут подключения к LLM{where}: {exc}", retryable=True) from exc
        except requests.RequestException as exc:
            raise LLMError(f"сетевая ошибка при обращении к LLM: {exc}", retryable=True) from exc

        with resp:
            if resp.status_code in (400, 404, 405, 422, 501):
                detail = _shorten(resp.text, 300)
                raise LLMError(
                    f"LLM отклонила стриминг (HTTP {resp.status_code}): {detail}",
                    retryable=False,
                    status=resp.status_code,
                )
            if resp.status_code in RETRY_STATUS:
                raise LLMError(f"LLM вернула HTTP {resp.status_code}", retryable=True, status=resp.status_code)
            if resp.status_code >= 400:
                raise LLMError(
                    f"LLM вернула HTTP {resp.status_code}: {_shorten(resp.text, 300)}",
                    retryable=False,
                    status=resp.status_code,
                )

            if "text/event-stream" not in resp.headers.get("Content-Type", ""):
                # Сервер проигнорировал stream:true и вернул целый JSON
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise LLMError("LLM вернула не JSON", retryable=False) from exc
                text, usage = self._extract_message(data)
                if not text.strip():
                    raise LLMError("LLM вернула пустой ответ", retryable=True)
                self._print_answer(text)
                self._report_finish(text, usage, time.monotonic() - started)
                return text

            stream_iter = resp.iter_lines(decode_unicode=False)
            while True:
                try:
                    raw_line = next(stream_iter)
                except StopIteration:
                    break
                except requests.Timeout as exc:
                    where = f" ({label})" if label else ""
                    raise LLMError(
                        f"LLM не присылала данные дольше {int(self.timeout)} с{where}",
                        retryable=True,
                    ) from exc
                except requests.RequestException as exc:
                    raise LLMError(f"обрыв соединения с LLM: {exc}", retryable=True) from exc

                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str in ("[DONE]", "DONE"):
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("error"), dict):
                    msg = chunk["error"].get("message") or "ошибка модели"
                    raise LLMError(f"LLM сообщила об ошибке: {msg}", retryable=False)
                usage = chunk.get("usage") or usage
                for piece in self._stream_deltas(chunk):
                    chunks.append(piece)
                    if self.show_stream:
                        sys.stdout.write(piece)
                        sys.stdout.flush()

            text = "".join(chunks)
            if self.show_stream and text:
                sys.stdout.write("\n")
                sys.stdout.flush()
            if not text.strip():
                raise LLMError("LLM вернула пустой ответ", retryable=True)
            self._report_finish(text, usage, time.monotonic() - started)
            return text

    @staticmethod
    def _stream_deltas(chunk: dict[str, Any]) -> Iterator[str]:
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                yield content
            elif choice.get("text"):
                yield choice["text"]

    def _ask_blocking(self, messages: list[dict[str, str]]) -> str:
        body = {"model": self.model, "messages": messages, "stream": False}
        started = time.monotonic()
        stop_flag = threading.Event()

        def progress() -> None:
            while not stop_flag.wait(1.0):
                elapsed = int(time.monotonic() - started)
                log(f"\r    · ожидание ответа LLM: {elapsed // 60}м {elapsed % 60:02d}с   ")

        thread = threading.Thread(target=progress, daemon=True)
        thread.start()
        try:
            resp = self.session.post(self.endpoint, json=body, timeout=(10, self.timeout))
        except requests.Timeout as exc:
            raise LLMError(f"таймаут ожидания LLM ({int(self.timeout)} с)", retryable=True) from exc
        except requests.RequestException as exc:
            raise LLMError(f"сетевая ошибка при обращении к LLM: {exc}", retryable=True) from exc
        finally:
            stop_flag.set()
            thread.join(timeout=1.5)
            if self.show_stream:
                log("\r" + " " * 60 + "\r")

        if resp.status_code in RETRY_STATUS:
            raise LLMError(f"LLM вернула HTTP {resp.status_code}", retryable=True, status=resp.status_code)
        if resp.status_code >= 400:
            raise LLMError(
                f"LLM вернула HTTP {resp.status_code}: {_shorten(resp.text, 300)}",
                retryable=False,
                status=resp.status_code,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError("LLM вернула не JSON", retryable=False) from exc

        text, usage = self._extract_message(data)
        if not text.strip():
            raise LLMError("LLM вернула пустой ответ", retryable=True)
        self._print_answer(text)
        self._report_finish(text, usage, time.monotonic() - started)
        return text

    @staticmethod
    def _extract_message(data: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        usage = data.get("usage")
        choices = data.get("choices") or []
        if not choices:
            message = data.get("message")
            if isinstance(message, dict) and message.get("content"):
                return str(message["content"]), usage
            raise LLMError("в ответе LLM нет поля choices")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # некоторые серверы отдают список блоков
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content or ""), usage

    def _print_answer(self, text: str) -> None:
        if self.show_stream:
            sys.stdout.write(text)
            if not text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()

    @staticmethod
    def _report_finish(text: str, usage: dict[str, Any] | None, elapsed: float) -> None:
        minutes, seconds = divmod(int(elapsed), 60)
        details = f"{minutes}м {seconds:02d}с, ответ {human_bytes(len(text.encode('utf-8')))}"
        if usage:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if prompt_tokens is not None:
                details += f", prompt {prompt_tokens} ток."
            if completion_tokens is not None:
                details += f", completion {completion_tokens} ток."
        log(f"    · LLM завершила за {details}")


# --------------------------------------------------------------------------- #
#  Оркестрация
# --------------------------------------------------------------------------- #


class Orchestrator:
    def __init__(self, args: argparse.Namespace, base_url: str, owner: str, repo: str) -> None:
        self.args = args
        self.base_url = base_url
        self.owner = owner
        self.repo = repo
        self.output_dir = os.path.abspath(args.output_dir)
        self.warnings: list[str] = []
        self.stats = {
            "prs_total": 0,
            "json_written": 0,
            "json_reused": 0,
            "llm_called": 0,
            "llm_skipped": 0,
            "failures": 0,
        }

    # -- имена файлов ------------------------------------------------------- #

    @staticmethod
    def json_name(index: int, sha: str) -> str:
        return f"{index}_{safe_filename_part(sha) or 'nosha'}.json"

    @staticmethod
    def answer_name(index: int, sha: str) -> str:
        return f"answer_{index}_{safe_filename_part(sha) or 'nosha'}.txt"

    # -- фильтры ------------------------------------------------------------ #

    def _label_filters(self) -> tuple[list[str], list[str]]:
        """Разобранные label-фильтры; парсим один раз на прогон."""
        cached = getattr(self, "_label_cache", None)
        if cached is None:
            cached = parse_label_filters(getattr(self.args, "label", None))
            self._label_cache = cached
        return cached

    def _comment_filter(self) -> tuple[list[str], list[str]] | None:
        """Разобранный comment-filter; None если флаг не задан."""
        raw = getattr(self.args, "comment_filter", None)
        if not raw:
            return None
        cached = getattr(self, "_comment_filter_cache", None)
        if cached is None:
            required, forbidden = parse_label_filters(raw)
            cached = (required, forbidden) if (required or forbidden) else None
            self._comment_filter_cache = cached
        return cached

    def _matches_filters(self, pull: dict[str, Any]) -> bool:
        required, forbidden = self._label_filters()
        if required or forbidden:
            labels = {
                (lbl.get("name") or "").strip().lower() for lbl in (pull.get("labels") or []) if isinstance(lbl, dict)
            }
            # все обязательные метки должны присутствовать (И)
            for name in required:
                if name.lower() not in labels:
                    return False
            # ни одной запрещённой быть не должно
            for name in forbidden:
                if name.lower() in labels:
                    return False

        if self.args.assignee:
            wanted = self.args.assignee.lower()
            logins = {(a.get("login") or "").lower() for a in (pull.get("assignees") or []) if isinstance(a, dict)}
            single = pull.get("assignee") or {}
            if isinstance(single, dict) and single.get("login"):
                logins.add(single["login"].lower())
            if wanted not in logins:
                return False

        return True

    # -- основной поток ----------------------------------------------------- #

    def run(self) -> int:
        client = ForgejoClient(
            f"{self.base_url}/api/v1",
            self.args.token,
            page_size=self.args.page_size,
            retries=self.args.http_retries,
            timeout=self.args.http_timeout,
        )

        log(f"Репозиторий: {self.owner}/{self.repo} @ {self.base_url}")
        try:
            repo_info = client.get_repo(self.owner, self.repo)
        except ForgejoError as exc:
            log(f"ОШИБКА: не удалось получить репозиторий — {exc}")
            return 2

        if repo_info.get("empty"):
            log("Репозиторий пуст.")
            return 0
        log(f"Доступ подтверждён: {repo_info.get('full_name')} (default branch: {repo_info.get('default_branch')})")

        log("Загружаю список открытых PR…")
        try:
            pulls = list(client.list_open_pulls(self.owner, self.repo))
        except ForgejoError as exc:
            log(f"ОШИБКА: не удалось получить список PR — {exc}")
            return 1

        log(f"Открытых PR: {len(pulls)}")
        selected = [p for p in pulls if self._matches_filters(p)]
        filtered_out = len(pulls) - len(selected)
        required, forbidden = self._label_filters()
        if required or forbidden or self.args.assignee:
            criteria = []
            if required or forbidden:
                criteria.append(f"метки «{format_label_filters(required, forbidden)}»")
            if self.args.assignee:
                criteria.append(f"assignee «{self.args.assignee}»")
            log(f"После фильтра (И: {', '.join(criteria)}): {len(selected)} (отсеяно {filtered_out})")

        if not selected:
            log("Нечего обрабатывать.")
            return 0

        if self.args.dry_run:
            for pull in selected:
                sha = ((pull.get("head") or {}).get("sha") or "")[:12]
                log(f"  #{pull['number']:>4} [{sha}] {pull.get('title')}")
            log("Режим --dry-run: файлы не создавались.")
            return 0

        os.makedirs(self.output_dir, exist_ok=True)
        system_prompt = self._read_prompt()
        llm = LLMClient(
            self.args.llm_url,
            self.args.model,
            timeout=self.args.llm_timeout,
            retries=self.args.llm_retries,
            show_stream=not self.args.no_stream,
            max_prompt_bytes=self.args.max_prompt_bytes,
            api_token=getattr(self.args, "openai_api_token", None),
        )

        collector = PRCollector(
            client,
            max_file_bytes=self.args.max_file_bytes,
            light_pr=getattr(self.args, "light_pr", False),
            comment_filter=self._comment_filter(),
        )
        self.stats["prs_total"] = len(selected)

        for position, pull in enumerate(selected, start=1):
            index = pull.get("number")
            prefix = f"[{position}/{len(selected)}] PR #{index}"
            sha = self._resolve_head_sha(client, pull)
            if not sha:
                log(f"{prefix}: не удалось определить head-sha — пропускаю")
                self.stats["failures"] += 1
                continue

            json_path = os.path.join(self.output_dir, self.json_name(index, sha))
            answer_path = os.path.join(self.output_dir, self.answer_name(index, sha))

            mode_tag = " [light]" if getattr(self.args, "light_pr", False) else ""
            log(f"\n{prefix} (sha {sha[:12]}){mode_tag} — {pull.get('title')}")

            pr_payload: dict[str, Any] | None = None
            if os.path.exists(json_path) and not self.args.refresh:
                log(f"    · JSON уже есть: {os.path.basename(json_path)}")
                self.stats["json_reused"] += 1
            else:
                log("    · собираю данные PR…")
                try:
                    pr_payload = collector.collect(
                        self.owner,
                        self.repo,
                        pull,
                        base_url=self.base_url,
                        warnings=self.warnings,
                    )
                except ForgejoError as exc:
                    log(f"    ! ошибка сбора данных: {exc}")
                    self.stats["failures"] += 1
                    continue
                try:
                    atomic_write_json(json_path, pr_payload)
                except OSError as exc:
                    log(f"    ! не удалось записать {json_path}: {exc}")
                    self.stats["failures"] += 1
                    continue
                self.stats["json_written"] += 1
                counts = pr_payload["counts"]
                log(
                    f"    · JSON сохранён: {os.path.basename(json_path)} "
                    f"({human_bytes(os.path.getsize(json_path))}; "
                    f"коммитов {counts['commits']}, файлов {counts['files']}, "
                    f"комментариев {counts['issue_comments']}, "
                    f"ревью {counts['reviews']}, событий {counts['timeline_events']})"
                )

            if os.path.exists(answer_path) and not self.args.force_llm:
                log(f"    · ответ LLM уже есть: {os.path.basename(answer_path)}")
                self.stats["llm_skipped"] += 1
                continue

            try:
                pr_json_text = self._load_pr_json(json_path, pr_payload)
            except (OSError, ValueError) as exc:
                log(f"    ! не удалось прочитать JSON PR: {exc}")
                self.stats["failures"] += 1
                continue

            log(f"    · запрос к LLM ({self.args.model})…")
            try:
                answer = llm.ask(
                    llm.build_messages(system_prompt, pr_json_text),
                    label=f"PR #{index}",
                )
            except LLMError as exc:
                log(f"    ! ошибка LLM: {exc}")
                self.stats["failures"] += 1
                continue

            try:
                atomic_write_text(answer_path, answer if answer.endswith("\n") else answer + "\n")
            except OSError as exc:
                log(f"    ! не удалось записать {answer_path}: {exc}")
                self.stats["failures"] += 1
                continue
            self.stats["llm_called"] += 1
            log(f"    · ответ сохранён: {os.path.basename(answer_path)}")

        self._print_summary()
        return 0 if self.stats["failures"] == 0 else 1

    # -- вспомогательное ---------------------------------------------------- #

    def _resolve_head_sha(self, client: ForgejoClient, pull: dict[str, Any]) -> str:
        sha = ((pull.get("head") or {}).get("sha") or "").strip()
        if sha:
            return sha
        try:
            fresh = client.get_pull(self.owner, self.repo, pull["number"])
        except ForgejoError:
            return ""
        return ((fresh.get("head") or {}).get("sha") or "").strip()

    def _read_prompt(self) -> str:
        try:
            with open(self.args.prompt_file, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            raise ConfigError(f"Не удалось прочитать prompt-file: {exc}") from exc
        if not text.strip():
            raise ConfigError(f"Файл промпта пуст: {self.args.prompt_file}")
        return text

    @staticmethod
    def _load_pr_json(path: str, payload: dict[str, Any] | None) -> str:
        if payload is not None:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def _print_summary(self) -> None:
        s = self.stats
        log("\n" + "=" * 62)
        log("ИТОГО")
        log(f"  PR обработано:        {s['prs_total']}")
        log(f"  JSON скачано:         {s['json_written']}")
        log(f"  JSON переиспользовано:{s['json_reused']:>3}")
        log(f"  запросов к LLM:       {s['llm_called']}")
        log(f"  ответов уже было:     {s['llm_skipped']}")
        log(f"  ошибок:               {s['failures']}")

        if self.warnings:
            unique = list(dict.fromkeys(self.warnings))
            log(f"\n  Предупреждений: {len(self.warnings)} (уникальных {len(unique)})")
            for warning in unique[:30]:
                log(f"    - {warning}")
            if len(unique) > 30:
                log(f"    … и ещё {len(unique) - 30}")
        log(f"\n  Каталог вывода: {self.output_dir}")
        log("=" * 62)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

_VALUE_FLAGS = {
    "--label",
    "--assignee",
    "--token",
    "--llm-url",
    "--model",
    "--prompt-file",
    "--output-dir",
    "--openai-api-token",
    "--comment-filter",
}


def normalize_argv(argv: list[str]) -> list[str]:
    """Склеивает ``--flag value`` в ``--flag=value`` для флагов со значением.

    Нужно, чтобы принимать значения, начинающиеся с дефиса
    (``--label -wip``), которые argparse иначе считает опциями. Явная
    форма ``--label=-wip`` работает и без этой функции.
    """
    result: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            result.extend(argv[i:])
            break
        if token in _VALUE_FLAGS and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            result.append(f"{token}={argv[i + 1]}")
            i += 2
            continue
        result.append(token)
        i += 1
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgejo_pr_llm.py",
        description=(
            "Выгружает открытые Pull Request из Forgejo в JSON и прогоняет каждый "
            "через локальную LLM (OpenAI-совместимый API)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  %(prog)s https://forgejo.example.com/owner/repo \\\n"
            "      --token $FORGEJO_TOKEN \\\n"
            "      --llm-url http://localhost:11434/v1/chat/completions \\\n"
            "      --model qwen2.5-coder:14b --prompt-file prompt.txt\n\n"
            "  # только PR с меткой и assignee, в отдельный каталог\n"
            "  %(prog)s https://forgejo.example.com/owner/repo --token $TOK \\\n"
            "      --llm-url http://localhost:11434/v1/chat/completions \\\n"
            "      --model qwen2.5-coder:14b --prompt-file prompt.txt \\\n"
            "      --label needs-review --assignee alice --output-dir ./out\n\n"
            "  # несколько меток: нужны 'bug' и 'needs-review', не должно быть 'wip'\n"
            "  %(prog)s https://forgejo.example.com/owner/repo --token $TOK \\\n"
            "      --llm-url http://localhost:11434/v1/chat/completions \\\n"
            "      --model qwen2.5-coder:14b --prompt-file prompt.txt \\\n"
            "      --label bug,needs-review,-wip\n\n"
            "  # то же самое отдельными флагами + токен для OpenAI-совместимого API\n"
            "  %(prog)s https://forgejo.example.com/owner/repo --token $TOK \\\n"
            "      --llm-url https://api.openai.com/v1/chat/completions \\\n"
            "      --openai-api-token $OPENAI_API_TOKEN \\\n"
            "      --model gpt-4o-mini --prompt-file prompt.txt \\\n"
            "      --label bug --label needs-review --label=-wip\n\n"
            "  # облегчённый режим + фильтр комментариев\n"
            "  %(prog)s https://forgejo.example.com/owner/repo --token $TOK \\\n"
            "      --llm-url http://localhost:11434/v1/chat/completions \\\n"
            "      --model qwen2.5-coder:14b --prompt-file prompt.txt \\\n"
            "      --light-pr --comment-filter alice,bob,-bot\n\n"
            "Файлы: <N>_<sha>.json и answer_<N>_<sha>.txt\n"
            "Существующие файлы не перекачиваются (--refresh / --force-llm для повтора)."
        ),
    )
    parser.add_argument("repo_url", metavar="REPO_URL", help="Полный URL репозитория: https://host/owner/repo")
    parser.add_argument(
        "--token", default=os.environ.get("FORGEJO_TOKEN"), help="Токен доступа Forgejo (по умолчанию $FORGEJO_TOKEN)"
    )
    parser.add_argument(
        "--llm-url",
        required=True,
        metavar="LLM_URL",
        help="Полный endpoint, напр. http://localhost:11434/v1/chat/completions",
    )
    parser.add_argument("--model", required=True, help="Имя модели для LLM endpoint")
    parser.add_argument(
        "--prompt-file", required=True, metavar="PATH", help="Файл с текстом промпта (статическая инструкция)"
    )
    parser.add_argument(
        "--label",
        default=None,
        action="append",
        metavar="МЕТКИ",
        help=(
            "Фильтр по меткам. Несколько меток — через запятую или повторным "
            "флагом. Метка без префикса обязательна (все условия по И), "
            "префикс '-' исключает PR с этой меткой: --label=bug,-wip"
        ),
    )
    parser.add_argument(
        "--openai-api-token",
        default=os.environ.get("OPENAI_API_TOKEN"),
        metavar="TOKEN",
        help=(
            "Bearer-токен для LLM endpoint (заголовок Authorization). "
            "По умолчанию $OPENAI_API_TOKEN; если пусто — не отправляется"
        ),
    )
    parser.add_argument(
        "--light-pr",
        action="store_true",
        help=(
            "Облегчённый формат снимка: только затронутые файлы (с патчами) "
            "и незакрытые review-комментарии с указанием строк. "
            "Без содержимого файлов, коммитов, timeline и общего диффа"
        ),
    )
    parser.add_argument(
        "--comment-filter",
        default=None,
        action="append",
        metavar="ПОЛЬЗОВАТЕЛИ",
        help=(
            "Фильтр комментариев по авторам. Синтаксис как у --label: "
            "список через запятую или повторный флаг, префикс '-' исключает автора. "
            "Применяется в любом режиме сохранения PR"
        ),
    )
    parser.add_argument("--assignee", default=None, help="Выгружать только PR с этим assignee (логин)")

    group = parser.add_argument_group("вывод")
    group.add_argument("--output-dir", default=".", help="Каталог для результатов (по умолчанию текущий)")
    group.add_argument("--no-stream", action="store_true", help="Не печатать ответ LLM в консоль по мере генерации")
    group.add_argument("--refresh", action="store_true", help="Перекачать JSON даже если файл уже существует")
    group.add_argument("--force-llm", action="store_true", help="Повторить запрос к LLM даже если ответ уже есть")
    group.add_argument("--dry-run", action="store_true", help="Только показать список PR, ничего не записывать")

    tuning = parser.add_argument_group("таймауты и лимиты")
    tuning.add_argument(
        "--llm-timeout",
        type=float,
        default=DEFAULT_LLM_TIMEOUT,
        metavar="СЕК",
        help=f"Ожидание ответа LLM (по умолчанию {DEFAULT_LLM_TIMEOUT})",
    )
    tuning.add_argument("--llm-retries", type=int, default=3, help="Повторы запросов к LLM")
    tuning.add_argument(
        "--http-timeout", type=float, default=60.0, metavar="СЕК", help="Таймаут запросов к Forgejo (по умолчанию 60)"
    )
    tuning.add_argument("--http-retries", type=int, default=3, help="Повторы запросов к Forgejo")
    tuning.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Размер страницы API (по умолчанию {DEFAULT_PAGE_SIZE}, максимум 50)",
    )
    tuning.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        metavar="БАЙТ",
        help="Лимит содержимого одного файла",
    )
    tuning.add_argument(
        "--max-prompt-bytes",
        type=int,
        default=DEFAULT_MAX_PROMPT_BYTES,
        metavar="БАЙТ",
        help="Порог предупреждения о размере запроса к LLM",
    )
    tuning.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.token:
        raise ConfigError("Не задан токен: укажите --token или переменную окружения FORGEJO_TOKEN.")
    if args.page_size < 1 or args.page_size > 50:
        raise ConfigError("--page-size должен быть от 1 до 50 (ограничение API Forgejo).")
    if args.llm_timeout <= 0:
        raise ConfigError("--llm-timeout должен быть больше нуля.")
    if args.max_file_bytes < 0:
        raise ConfigError("--max-file-bytes не может быть отрицательным.")
    if not os.path.isfile(args.prompt_file):
        raise ConfigError(f"--prompt-file не найден: {args.prompt_file}")
    parse_label_filters(args.label)  # ранняя проверка синтаксиса
    if getattr(args, "comment_filter", None) is not None:
        parse_label_filters(args.comment_filter)  # ранняя проверка синтаксиса


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalize_argv(raw))

    try:
        validate_args(args)
        base_url, owner, repo = parse_repo_url(args.repo_url)
    except ConfigError as exc:
        log(f"ОШИБКА КОНФИГУРАЦИИ: {exc}")
        return 2

    orchestrator = Orchestrator(args, base_url, owner, repo)
    try:
        return orchestrator.run()
    except KeyboardInterrupt:
        log("\n\nПрервано пользователем. Уже записанные файлы сохранены; повторный запуск продолжит с места остановки.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
