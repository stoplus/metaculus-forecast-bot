"""Клиент API Metaculus: список открытых вопросов турнира, отправка прогноза и комментария."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

API_BASE_URL = "https://www.metaculus.com/api"
SUPPORTED_TYPES = ("binary", "multiple_choice", "numeric", "discrete")
REQUEST_TIMEOUT = 60


@dataclass
class Question:
    """Один вопрос, на который бот даёт прогноз."""

    question_id: int
    post_id: int
    type: str
    title: str
    description: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    unit: str = ""
    options: list[str] = field(default_factory=list)
    range_min: float | None = None
    range_max: float | None = None
    zero_point: float | None = None
    open_lower_bound: bool = False
    open_upper_bound: bool = False
    cdf_size: int = 201
    close_time: str = ""
    resolve_time: str = ""
    already_forecast: bool = False
    tournament: str = ""

    @property
    def url(self) -> str:
        return f"https://www.metaculus.com/questions/{self.post_id}/"


def question_from_json(q: dict[str, Any], post_id: int, tournament: str = "") -> Question:
    """Собирает Question из блока `question` ответа API."""
    scaling = q.get("scaling") or {}
    inbound = scaling.get("inbound_outcome_count")
    my_forecasts = q.get("my_forecasts") or {}
    latest = my_forecasts.get("latest") or {}
    return Question(
        question_id=q["id"],
        post_id=post_id,
        type=q.get("type", ""),
        title=q.get("title", ""),
        description=q.get("description") or "",
        resolution_criteria=q.get("resolution_criteria") or "",
        fine_print=q.get("fine_print") or "",
        unit=q.get("unit") or "",
        options=list(q.get("options") or []),
        range_min=scaling.get("range_min"),
        range_max=scaling.get("range_max"),
        zero_point=scaling.get("zero_point"),
        open_lower_bound=bool(q.get("open_lower_bound")),
        open_upper_bound=bool(q.get("open_upper_bound")),
        cdf_size=(int(inbound) + 1) if inbound else 201,
        close_time=q.get("scheduled_close_time") or "",
        resolve_time=q.get("scheduled_resolve_time") or "",
        already_forecast=latest.get("forecast_values") is not None,
        tournament=tournament,
    )


def questions_from_post(post: dict[str, Any], tournament: str = "") -> list[Question]:
    """Вопросы поста: одиночный вопрос или все вопросы группы. Условные пары пропускаются."""
    post_id = post["id"]
    found: list[dict[str, Any]] = []
    if post.get("question"):
        found.append(post["question"])
    group = post.get("group_of_questions") or {}
    found.extend(group.get("questions") or [])
    return [question_from_json(q, post_id, tournament) for q in found]


class Metaculus:
    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("METACULUS_TOKEN", "")
        if not self.token:
            raise RuntimeError("не задан METACULUS_TOKEN")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Token {self.token}"

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        """Запрос с повторами при временных сбоях и ограничении частоты."""
        url = f"{API_BASE_URL}{path}"
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            except requests.RequestException as error:
                last_error = error
            else:
                if response.status_code < 400:
                    return response
                last_error = RuntimeError(
                    f"{method} {path}: HTTP {response.status_code} {response.text[:500]}"
                )
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
            time.sleep(5 * (attempt + 1))
        raise last_error  # type: ignore[misc]

    def open_questions(self, tournament: str | int) -> list[Question]:
        """Открытые вопросы турнира поддерживаемых типов (с постраничной выборкой)."""
        result: list[Question] = []
        offset = 0
        limit = 50
        while True:
            params = {
                "limit": limit,
                "offset": offset,
                "order_by": "-hotness",
                "forecast_type": ",".join(SUPPORTED_TYPES),
                "tournaments": [tournament],
                "statuses": "open",
                "include_description": "true",
            }
            data = self._request("GET", "/posts/", params=params).json()
            posts = data.get("results") or []
            for post in posts:
                for question in questions_from_post(post, str(tournament)):
                    result.append(question)
            if len(posts) < limit or not data.get("next"):
                break
            offset += limit
        return result

    def question_details(self, question: Question) -> Question:
        """Перечитывает вопрос по его посту: полный текст и признак «уже есть прогноз»."""
        post = self._request("GET", f"/posts/{question.post_id}/").json()
        for fresh in questions_from_post(post, question.tournament):
            if fresh.question_id == question.question_id:
                return fresh
        return question

    def submit_forecast(self, question: Question, payload: dict[str, Any]) -> None:
        body = [{"question": question.question_id, "source": "api", **payload}]
        self._request("POST", "/questions/forecast/", json=body)

    def post_comment(self, question: Question, text: str) -> None:
        body = {
            "text": text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": question.post_id,
        }
        self._request("POST", "/comments/create/", json=body)


def forecast_payload(question_type: str, forecast: Any) -> dict[str, Any]:
    """Тело прогноза в формате API для каждого типа вопроса."""
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }
