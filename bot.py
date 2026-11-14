"""Бот прогнозов для турниров Metaculus FutureEval.

Режимы:
  python bot.py                 - прогноз по всем открытым вопросам турниров без нашего прогноза
  python bot.py --check         - только посчитать такие вопросы (для быстрой проверки в расписании)
  python bot.py --dry-run       - сделать прогнозы, но ничего не отправлять
  python bot.py --question-file вопрос.json [--runs 1]
                                - прогноз по вопросу из файла, без Metaculus (числовые и с вариантами тоже)
  python bot.py --ask "Вопрос?" --criteria "что считается «да» и к какой дате" [--background "..."]
                                - свой вопрос с ответом да/нет, без Metaculus

Правила турнира, которые соблюдает код: один прогноз на вопрос, прогноз и комментарий
отправляет сам бот через API, человек в прогнозе не участвует.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from forecaster import Forecast, forecast_question
from metaculus import SUPPORTED_TYPES, Metaculus, Question, forecast_payload, question_from_json

# Осенний сезон FutureEval 2026 и текущий раунд MiniBench. Новый сезон - поменять первый номер.
DEFAULT_TOURNAMENTS = "33121,minibench"
QUESTION_CONCURRENCY = int(os.environ.get("QUESTION_CONCURRENCY", "2"))
MAX_COMMENT_CHARS = 9000
LOG_FILE = Path(os.environ.get("FORECAST_LOG", "forecast-log.jsonl"))
# Комментарии, которые не удалось отправить после принятого прогноза. В GitHub Actions файл
# переносится между запусками через кеш; следующий запуск отправляет их первыми.
PENDING_COMMENTS = Path(os.environ.get("PENDING_COMMENTS", "pending-comments.jsonl"))
_files_lock = threading.Lock()


def tournaments() -> list[str]:
    raw = os.environ.get("TOURNAMENTS", DEFAULT_TOURNAMENTS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def pending_questions(client: Metaculus) -> list[Question]:
    """Открытые вопросы всех турниров, по которым у бота ещё нет прогноза. Ближайшие к закрытию - первыми."""
    seen: set[int] = set()
    pending: list[Question] = []
    failed: list[str] = []
    names = tournaments()
    for tournament in names:
        try:
            questions = client.open_questions(tournament)
        except Exception as error:  # один недоступный турнир не должен останавливать остальные
            print(f"[ошибка] турнир {tournament}: {error}", flush=True)
            failed.append(f"{tournament}: {error}")
            continue
        for question in questions:
            if question.question_id in seen or question.already_forecast:
                continue
            if question.type not in SUPPORTED_TYPES:
                print(f"[пропуск] {question.url}: тип {question.type} не поддерживается", flush=True)
                continue
            seen.add(question.question_id)
            pending.append(question)
    if names and len(failed) == len(names):
        # Например, отозван токен: без исключения это выглядело бы как «вопросов нет», и запуск прошёл бы зелёным.
        raise RuntimeError("ни один турнир не прочитан: " + "; ".join(failed))
    pending.sort(key=lambda q: q.close_time or "9999")
    return pending


def save_pending_comment(question: Question, text: str) -> None:
    entry = {"post_id": question.post_id, "question_id": question.question_id, "url": question.url, "text": text}
    with _files_lock, PENDING_COMMENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def retry_pending_comments(client: Metaculus) -> int:
    """Отправляет отложенные комментарии; неотправленные остаются в файле. Возвращает число неотправленных."""
    if not PENDING_COMMENTS.exists():
        return 0
    lines = [line for line in PENDING_COMMENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    still: list[str] = []
    for line in lines:
        entry = json.loads(line)
        question = Question(question_id=entry["question_id"], post_id=entry["post_id"], type="", title="")
        try:
            client.post_comment(question, entry["text"])
            print(f"[комментарий] отправлен отложенный: {entry['url']}", flush=True)
        except Exception as error:
            print(f"[ошибка] отложенный комментарий {entry['url']}: {error}", flush=True)
            still.append(line)
    if still:
        PENDING_COMMENTS.write_text("\n".join(still) + "\n", encoding="utf-8")
    else:
        PENDING_COMMENTS.unlink()
    return len(still)


def log_entry(question: Question, forecast: Forecast | None, status: str, error: str = "") -> None:
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "tournament": question.tournament,
        "question_id": question.question_id,
        "url": question.url,
        "type": question.type,
        "title": question.title,
        "error": error,
    }
    if forecast is not None:
        entry["summary"] = forecast.summary
        entry["runs"] = [
            {"ok": r.ok, "error": r.error, "cost_usd": r.cost_usd, "duration_s": round(r.duration_s), "turns": r.turns, "tokens": r.tokens, "web_searches": r.web_searches}
            for r in forecast.runs
        ]
    with _files_lock, LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ask_locally(question: Question, runs: int | None) -> int:
    """Прогноз по своему вопросу: в Metaculus ничего не уходит, ответ с обоснованием печатается по-русски."""
    try:
        forecast = forecast_question(question, runs_count=runs, language="Russian")
    except Exception as error:
        print(f"[ошибка] {error}", flush=True)
        log_entry(question, None, "forecast_failed", str(error))
        return 1
    log_entry(question, forecast, "local")
    print(forecast.comment, flush=True)
    return 0


def handle_question(client: Metaculus, question: Question, dry_run: bool, runs: int | None) -> bool:
    """Прогноз по одному вопросу турнира и отправка. Возвращает True при успехе."""
    started = time.time()
    try:
        # Перед работой перечитываем вопрос: вдруг прогноз уже отправлен соседним запуском.
        question = client.question_details(question)
        if question.already_forecast:
            print(f"[пропуск] {question.url}: прогноз уже есть", flush=True)
            return True
        forecast = forecast_question(question, runs_count=runs)
    except Exception as error:
        print(f"[ошибка] {question.url}: {error}", flush=True)
        log_entry(question, None, "forecast_failed", str(error))
        return False

    minutes = (time.time() - started) / 60
    print(f"[прогноз] {question.url} | {question.title[:90]} | {forecast.summary} | {minutes:.1f} мин", flush=True)
    if dry_run:
        log_entry(question, forecast, "dry_run")
        return True

    try:
        client.submit_forecast(question, forecast_payload(question.type, forecast.value))
    except Exception as error:
        print(f"[ошибка] отправка прогноза {question.url}: {error}", flush=True)
        log_entry(question, forecast, "submit_failed", str(error))
        return False
    try:
        client.post_comment(question, forecast.comment[:MAX_COMMENT_CHARS])
    except Exception as error:
        # Прогноз уже принят, и следующий запуск вопрос пропустит; комментарий обязателен по правилам,
        # поэтому он откладывается и отправляется в начале следующего запуска.
        print(f"[ошибка] комментарий {question.url}: {error}; отложен до следующего запуска", flush=True)
        save_pending_comment(question, forecast.comment[:MAX_COMMENT_CHARS])
        log_entry(question, forecast, "comment_failed", str(error))
        return False
    log_entry(question, forecast, "submitted")
    return True


def load_question_file(path: str) -> Question:
    """Вопрос из локального JSON: либо блок `question` из API, либо упрощённый набор полей."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "question" in data and isinstance(data["question"], dict):
        return question_from_json(data["question"], data.get("id", 0), "local")
    if "scaling" in data or "id" in data:
        return question_from_json(data, data.get("post_id", 0), "local")
    known = {name for name in Question.__dataclass_fields__}
    fields = {key: value for key, value in data.items() if key in known}
    fields.setdefault("question_id", 0)
    fields.setdefault("post_id", 0)
    return Question(**fields)


def main() -> int:
    # До разбора аргументов: справка и сообщения об ошибках тоже печатаются по-русски.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Бот прогнозов Metaculus FutureEval")
    parser.add_argument("--check", action="store_true", help="только посчитать вопросы без прогноза")
    parser.add_argument("--dry-run", action="store_true", help="не отправлять прогнозы")
    parser.add_argument("--question-file", help="прогноз по вопросу из файла JSON, без Metaculus")
    parser.add_argument("--ask", help="свой вопрос с ответом да/нет: вернётся вероятность «да»")
    parser.add_argument("--criteria", default="", help="для --ask: что именно считается ответом «да» и к какой дате")
    parser.add_argument("--background", default="", help="для --ask: что модели нужно знать, чего нет в интернете")
    parser.add_argument("--runs", type=int, help="число прогонов модели на вопрос")
    parser.add_argument("--limit", type=int, help="не больше стольких вопросов за запуск")
    args = parser.parse_args()

    if args.ask:
        question = Question(
            question_id=0,
            post_id=0,
            type="binary",
            title=args.ask,
            description=args.background,
            resolution_criteria=args.criteria,
            tournament="local",
        )
        return ask_locally(question, args.runs)
    if args.question_file:
        return ask_locally(load_question_file(args.question_file), args.runs)

    client = Metaculus()
    unsent_comments = 0 if args.dry_run else retry_pending_comments(client)
    pending = pending_questions(client)
    if args.limit:
        pending = pending[: args.limit]
    print(f"Вопросов без прогноза: {len(pending)}", flush=True)
    if args.check:
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a", encoding="utf-8") as handle:
                handle.write(f"pending={len(pending)}\nunsent_comments={unsent_comments}\n")
        # Проверка не падает из-за комментариев, чтобы не мешать прогнозам по новым вопросам;
        # о неотправленных сообщает отдельный последний шаг расписания.
        return 0
    if not pending:
        return 1 if unsent_comments else 0

    with ThreadPoolExecutor(max_workers=QUESTION_CONCURRENCY) as pool:
        results = list(pool.map(lambda q: handle_question(client, q, args.dry_run, args.runs), pending))
    failed = results.count(False)
    print(f"Готово: {results.count(True)} из {len(results)}, сбоев {failed}", flush=True)
    # Сбой помечает запуск как неуспешный, чтобы GitHub прислал письмо; вопрос повторится в следующем запуске.
    return 1 if failed or unsent_comments else 0


if __name__ == "__main__":
    sys.exit(main())
