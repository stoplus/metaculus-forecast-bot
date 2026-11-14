"""Проверки без сети: распределения, сведение прогонов, разбор вопросов. Запуск: python -m unittest -v"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdf import build_cdf, check_cdf, scale_locations  # noqa: E402
from forecaster import (  # noqa: E402
    PERCENTILES,
    aggregate_binary,
    aggregate_multiple_choice,
    aggregate_percentiles,
    answer_instructions,
    build_prompt,
    schema_for,
)
from metaculus import forecast_payload, question_from_json, questions_from_post  # noqa: E402


def normal_percentiles(center: float, spread: float) -> dict[float, float]:
    z = {1: -2.326, 5: -1.645, 10: -1.282, 20: -0.842, 30: -0.524, 40: -0.253, 50: 0.0,
         60: 0.253, 70: 0.524, 80: 0.842, 90: 1.282, 95: 1.645, 99: 2.326}
    return {float(p): center + spread * value for p, value in z.items()}


def median_of(cdf: list[float], range_min: float, range_max: float, zero_point=None) -> float:
    grid = scale_locations(len(cdf), range_min, range_max, zero_point)
    return float(np.interp(0.5, cdf, grid))


class CdfTests(unittest.TestCase):
    def test_closed_bounds(self):
        cdf = build_cdf(normal_percentiles(50, 10), 0, 100, None, False, False)
        self.assertEqual(len(cdf), 201)
        self.assertAlmostEqual(cdf[0], 0.0)
        self.assertAlmostEqual(cdf[-1], 1.0)
        self.assertAlmostEqual(median_of(cdf, 0, 100), 50, delta=0.5)

    def test_open_bounds_keep_minimum_mass(self):
        cdf = build_cdf(normal_percentiles(50, 5), 0, 100, None, True, True)
        self.assertGreaterEqual(cdf[0], 0.001)
        self.assertLessEqual(cdf[-1], 0.999)

    def test_mass_below_open_lower_bound(self):
        # Модель считает, что исход скорее всего ниже нижней границы шкалы (как в вопросе про голосование ООН).
        cdf = build_cdf(normal_percentiles(123, 3), 130, 160, None, True, False)
        self.assertGreater(cdf[0], 0.9)
        self.assertAlmostEqual(cdf[-1], 1.0)

    def test_percentiles_outside_closed_bound_are_squeezed_in(self):
        cdf = build_cdf(normal_percentiles(2, 5), 0, 100, None, False, True)
        self.assertAlmostEqual(cdf[0], 0.0)
        check_cdf(cdf, False, True)

    def test_very_narrow_distribution_is_capped(self):
        cdf = build_cdf(normal_percentiles(50, 0.01), 0, 100, None, False, False)
        self.assertLessEqual(max(np.diff(cdf)), 0.2 + 1e-9)
        self.assertAlmostEqual(median_of(cdf, 0, 100), 50, delta=1.0)

    def test_equal_percentile_values(self):
        percentiles = {float(p): 7.0 for p in PERCENTILES}
        cdf = build_cdf(percentiles, 0, 10, None, False, False)
        check_cdf(cdf, False, False)

    def test_log_scale(self):
        percentiles = normal_percentiles(1000, 200)
        cdf = build_cdf(percentiles, 10, 100000, 0, False, True)
        self.assertAlmostEqual(median_of(cdf, 10, 100000, 0), 1000, delta=60)

    def test_discrete(self):
        # 11 исходов от 0 до 10: границы шкалы -0.5 и 10.5, 12 точек CDF.
        cdf = build_cdf(normal_percentiles(3, 1.5), -0.5, 10.5, None, False, False, n=12)
        self.assertEqual(len(cdf), 12)
        check_cdf(cdf, False, False)
        self.assertGreater(cdf[4] - cdf[3], cdf[10] - cdf[9])

    def test_steps_within_limits_for_many_shapes(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            low, high = sorted(rng.uniform(-1000, 1000, size=2))
            if high - low < 1:
                high = low + 1
            center = rng.uniform(low - 0.3 * (high - low), high + 0.3 * (high - low))
            spread = rng.uniform(0.0005, 0.6) * (high - low)
            open_lower, open_upper = bool(rng.integers(2)), bool(rng.integers(2))
            try:
                cdf = build_cdf(normal_percentiles(center, spread), low, high, None, open_lower, open_upper)
            except ValueError as error:
                # Допустим только отказ, когда всё распределение лежит за закрытой границей.
                self.assertIn("за границами", str(error))
                continue
            check_cdf(cdf, open_lower, open_upper)


class AggregationTests(unittest.TestCase):
    def test_binary_median_and_clamp(self):
        self.assertEqual(aggregate_binary([0.2, 0.9, 0.3], False), 0.3)
        self.assertEqual(aggregate_binary([0.999, 0.9999, 0.998], False), 0.99)
        self.assertEqual(aggregate_binary([0.999, 0.9999, 0.9995], True), 0.997)
        self.assertEqual(aggregate_binary([0.0, 0.0001, 0.001], False), 0.01)

    def test_binary_two_runs_use_logit_mean(self):
        self.assertAlmostEqual(aggregate_binary([0.2, 0.8], False), 0.5, places=3)

    def test_multiple_choice(self):
        options = ["Alpha", "Beta", "Gamma"]
        runs = [
            {"Alpha": 0.7, "Beta": 0.3, "Gamma": 0.0},
            {"alpha ": 0.5, "BETA": 0.5},
        ]
        result = aggregate_multiple_choice(runs, options)
        self.assertEqual(list(result), options)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
        self.assertGreaterEqual(result["Gamma"], 0.004)
        self.assertGreater(result["Alpha"], result["Beta"])

    def test_percentiles_median_and_order(self):
        base = {p: float(p) for p in PERCENTILES}
        shifted = {p: float(p) + 10 for p in PERCENTILES}
        wild = {p: float(p) + 1000 for p in PERCENTILES}
        result = aggregate_percentiles([base, shifted, wild])
        self.assertEqual(result[50], 60.0)
        self.assertEqual(list(result.values()), sorted(result.values()))


class ParsingTests(unittest.TestCase):
    NUMERIC = {
        "id": 111,
        "type": "numeric",
        "title": "How many?",
        "description": "Background",
        "resolution_criteria": "Criteria",
        "fine_print": "",
        "unit": "states",
        "scaling": {"range_min": 130.0, "range_max": 160.0, "zero_point": None},
        "open_lower_bound": True,
        "open_upper_bound": False,
        "scheduled_close_time": "2026-10-20T20:00:00Z",
        "my_forecasts": {"latest": None},
    }

    def test_numeric_question(self):
        question = question_from_json(self.NUMERIC, 222, "minibench")
        self.assertEqual((question.question_id, question.post_id), (111, 222))
        self.assertEqual(question.cdf_size, 201)
        self.assertTrue(question.open_lower_bound)
        self.assertFalse(question.already_forecast)
        text = answer_instructions(question)
        self.assertIn("Outcomes below it are possible", text)
        self.assertIn("cannot be higher than 160", text)

    def test_discrete_question_size(self):
        data = {**self.NUMERIC, "type": "discrete",
                "scaling": {"range_min": -0.5, "range_max": 10.5, "zero_point": None, "inbound_outcome_count": 11}}
        question = question_from_json(data, 1)
        self.assertEqual(question.cdf_size, 12)
        self.assertIn("spaced 1 apart", answer_instructions(question))

    def test_already_forecast(self):
        data = {**self.NUMERIC, "my_forecasts": {"latest": {"forecast_values": [0.1, 0.9]}}}
        self.assertTrue(question_from_json(data, 1).already_forecast)

    def test_group_post(self):
        post = {"id": 5, "group_of_questions": {"questions": [
            {"id": 1, "type": "binary", "title": "A"}, {"id": 2, "type": "binary", "title": "B"}]}}
        self.assertEqual([q.question_id for q in questions_from_post(post)], [1, 2])

    def test_prompt_and_schema(self):
        question = question_from_json({"id": 1, "type": "binary", "title": "Will X happen?",
                                       "resolution_criteria": "If X."}, 9)
        prompt = build_prompt(question)
        self.assertIn("Will X happen?", prompt)
        self.assertIn("If X.", prompt)
        self.assertIn("rationale in English", prompt)
        self.assertIn("rationale in Russian", build_prompt(question, language="Russian"))
        schema = schema_for(question)
        self.assertEqual(set(schema["required"]), {"probability", "already_settled", "rationale"})
        numeric_schema = schema_for(question_from_json(self.NUMERIC, 1))
        self.assertIn("p50", numeric_schema["required"])

    def test_payloads(self):
        self.assertEqual(forecast_payload("binary", 0.3)["probability_yes"], 0.3)
        self.assertEqual(forecast_payload("multiple_choice", {"A": 1.0})["probability_yes_per_category"], {"A": 1.0})
        self.assertEqual(forecast_payload("discrete", [0.0, 1.0])["continuous_cdf"], [0.0, 1.0])


class JudgeTests(unittest.TestCase):
    """Судья: при SYNTHESIZE=1 итог берётся у него, при его сбое - механическое сведение."""

    def run_with(self, question_json, member_data, judge_data, synthesize=True):
        from unittest import mock

        import forecaster
        from forecaster import RunResult

        import threading

        question = question_from_json(question_json, 1)
        calls = []
        lock = threading.Lock()
        member_index = [0]

        def fake_run(prompt, schema, system_prompt=forecaster.SYSTEM_PROMPT):
            # Прогоны идут в нескольких потоках: номер прогона выдаётся под замком.
            with lock:
                calls.append(system_prompt)
                if system_prompt != forecaster.JUDGE_PROMPT:
                    index = member_index[0]
                    member_index[0] += 1
            if system_prompt == forecaster.JUDGE_PROMPT:
                self.assertIn("Member 2", prompt)
                return RunResult(ok=judge_data is not None, data=judge_data or {}, cost_usd=0.5)
            return RunResult(ok=True, data=member_data[index % len(member_data)])

        with mock.patch.object(forecaster, "run_claude", side_effect=fake_run), \
                mock.patch.object(forecaster, "SYNTHESIZE", synthesize):
            result = forecaster.forecast_question(question, runs_count=len(member_data))
        return result, calls

    BINARY = {"id": 1, "type": "binary", "title": "Will X?"}
    MEMBERS = [
        {"probability": 0.12, "already_settled": False, "rationale": "a"},
        {"probability": 0.19, "already_settled": False, "rationale": "b"},
        {"probability": 0.60, "already_settled": False, "rationale": "c"},
    ]

    def test_judge_decides(self):
        result, calls = self.run_with(self.BINARY, self.MEMBERS,
                                      {"probability": 0.15, "already_settled": False, "rationale": "judge"})
        self.assertEqual(result.value, 0.15)
        self.assertIn("judge", result.comment)
        self.assertEqual(len(calls), 4)

    def test_judge_failure_falls_back_to_median(self):
        result, _ = self.run_with(self.BINARY, self.MEMBERS, None)
        self.assertEqual(result.value, 0.19)
        # Неудачный судья всё равно попадает в журнал вместе с расходом.
        self.assertEqual(len(result.runs), 4)
        self.assertEqual(result.runs[-1].cost_usd, 0.5)

    def test_judge_alone_cannot_unlock_extremes(self):
        members = [{"probability": 0.97, "already_settled": False, "rationale": "m"}] * 3
        judge = {"probability": 0.999, "already_settled": True, "rationale": "judge"}
        result, _ = self.run_with(self.BINARY, members, judge)
        self.assertEqual(result.value, 0.99)

    def test_all_settled_unlocks_extremes(self):
        members = [{"probability": 0.999, "already_settled": True, "rationale": "m"}] * 3
        judge = {"probability": 0.999, "already_settled": True, "rationale": "judge"}
        result, _ = self.run_with(self.BINARY, members, judge)
        self.assertEqual(result.value, 0.997)

    def test_judge_with_paraphrased_options_falls_back(self):
        question = {"id": 1, "type": "multiple_choice", "title": "Who?", "options": ["Alpha", "Beta"]}
        member = {"probabilities": [{"option": "Alpha", "probability": 0.7}, {"option": "Beta", "probability": 0.3}],
                  "already_settled": False, "rationale": "m"}
        judge = {"probabilities": [{"option": "Option A", "probability": 0.5}, {"option": "Beta", "probability": 0.5}],
                 "already_settled": False, "rationale": "judge"}
        result, _ = self.run_with(question, [member, member], judge)
        self.assertAlmostEqual(result.value["Alpha"], 0.7, places=3)
        self.assertNotIn("итог судьи", result.summary)

    def test_bad_member_is_skipped(self):
        members = [
            {"probability": 66, "already_settled": False, "rationale": "percent"},
            {"probability": float("nan"), "already_settled": False, "rationale": "nan"},
            {"probability": 0.3, "already_settled": False, "rationale": "ok"},
        ]
        result, _ = self.run_with(self.BINARY, members, None, synthesize=False)
        self.assertEqual(result.value, 0.3)

    def test_judge_off_by_default_settings(self):
        result, calls = self.run_with(self.BINARY, self.MEMBERS, {"probability": 0.5}, synthesize=False)
        self.assertEqual(result.value, 0.19)
        self.assertEqual(len(calls), 3)

    def test_judge_numeric(self):
        numeric = {**ParsingTests.NUMERIC}
        member = {f"p{p}": 120.0 + i for i, p in enumerate(PERCENTILES)}
        member.update(already_settled=False, rationale="m")
        judge = {f"p{p}": 125.0 + i for i, p in enumerate(PERCENTILES)}
        judge.update(already_settled=False, rationale="judge")
        result, calls = self.run_with(numeric, [member, member], judge)
        self.assertEqual(len(calls), 3)
        self.assertIn("p50=131", result.summary)
        check_cdf(result.value, True, False)


class ParseTests(unittest.TestCase):
    def test_binary(self):
        from forecaster import parse_binary

        question = question_from_json({"id": 1, "type": "binary", "title": "X"}, 1)
        self.assertEqual(parse_binary({"probability": 0.4}, question), 0.4)
        for bad in (66, -0.1, float("nan"), float("inf"), True, "0.4", None):
            self.assertIsNone(parse_binary({"probability": bad}, question), bad)

    def test_numeric(self):
        from forecaster import parse_numeric

        good = {f"p{p}": float(p) for p in PERCENTILES}
        self.assertEqual(parse_numeric(good, None)[50], 50.0)
        self.assertIsNone(parse_numeric({**good, "p50": None}, None))
        self.assertIsNone(parse_numeric({**good, "p50": True}, None))

    def test_text_fallback_turns_nan_into_none(self):
        from forecaster import _json_from_text

        self.assertEqual(_json_from_text('answer {"probability": NaN}'), {"probability": None})


class RobustnessTests(unittest.TestCase):
    def test_run_claude_never_raises(self):
        from unittest import mock

        import forecaster

        with mock.patch.object(forecaster, "find_claude", side_effect=RuntimeError("нет claude")):
            result = forecaster.run_claude("q", {})
        self.assertFalse(result.ok)
        self.assertIn("нет claude", result.error)

    def test_all_tournaments_failing_raises(self):
        from unittest import mock

        import bot

        client = mock.Mock()
        client.open_questions.side_effect = RuntimeError("HTTP 401")
        with mock.patch.dict("os.environ", {"TOURNAMENTS": "1,minibench"}):
            with self.assertRaises(RuntimeError):
                bot.pending_questions(client)

    def test_one_tournament_failing_is_tolerated(self):
        from unittest import mock

        import bot

        client = mock.Mock()
        client.open_questions.side_effect = [RuntimeError("HTTP 500"), []]
        with mock.patch.dict("os.environ", {"TOURNAMENTS": "1,minibench"}):
            self.assertEqual(bot.pending_questions(client), [])

    def test_failed_comment_is_saved_and_retried(self):
        import tempfile
        from unittest import mock

        import bot
        from forecaster import Forecast

        question = question_from_json({"id": 7, "type": "binary", "title": "X"}, 70)
        forecast = Forecast(value=0.3, summary="30%", comment="why", runs=[])
        client = mock.Mock()
        client.question_details.return_value = question
        client.post_comment.side_effect = RuntimeError("HTTP 500")
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(bot, "PENDING_COMMENTS", Path(folder) / "pending.jsonl"), \
                mock.patch.object(bot, "LOG_FILE", Path(folder) / "log.jsonl"), \
                mock.patch.object(bot, "forecast_question", return_value=forecast):
            self.assertFalse(bot.handle_question(client, question, dry_run=False, runs=1))
            client.submit_forecast.assert_called_once()
            self.assertEqual(bot.retry_pending_comments(client), 1)  # всё ещё не уходит
            client.post_comment.side_effect = None
            self.assertEqual(bot.retry_pending_comments(client), 0)
            self.assertFalse(bot.PENDING_COMMENTS.exists())
            sent_question, sent_text = client.post_comment.call_args.args
            self.assertEqual((sent_question.post_id, sent_text), (70, "why"))


class OpenRouterTests(unittest.TestCase):
    def call(self, status, body, web="native"):
        from unittest import mock

        import forecaster

        response = mock.Mock(status_code=status)
        response.json.return_value = body
        response.text = json.dumps(body)
        with mock.patch.object(forecaster, "PROVIDER", "openrouter"), \
                mock.patch.object(forecaster, "OPENROUTER_WEB", web), \
                mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), \
                mock.patch.object(forecaster.requests, "post", return_value=response) as post:
            result = forecaster.run_claude("question", {"type": "object"}, "system")
        return result, post.call_args

    def test_success(self):
        body = {"choices": [{"message": {"content": '{"probability": 0.42, "already_settled": false, "rationale": "r"}'}}],
                "usage": {"cost": 0.12, "total_tokens": 3000, "server_tool_use": {"web_search_requests": 4}}}
        result, call = self.call(200, body)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["probability"], 0.42)
        self.assertEqual((result.cost_usd, result.tokens, result.web_searches), (0.12, 3000, 4))
        payload = call.kwargs["json"]
        self.assertEqual(payload["plugins"][0]["engine"], "native")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_web_off_has_no_plugin_and_says_so(self):
        body = {"choices": [{"message": {"content": 'Here: {"probability": 0.3}'}}], "usage": {}}
        result, call = self.call(200, body, web="off")
        self.assertTrue(result.ok)  # JSON внутри текста тоже разбирается
        payload = call.kwargs["json"]
        self.assertNotIn("plugins", payload)
        self.assertIn("Web search is not available", payload["messages"][0]["content"])

    def test_http_error(self):
        result, _ = self.call(402, {"error": {"message": "Insufficient credits"}})
        self.assertFalse(result.ok)
        self.assertIn("Insufficient credits", result.error)

    def test_missing_key(self):
        from unittest import mock

        import forecaster

        with mock.patch.object(forecaster, "PROVIDER", "openrouter"), mock.patch.dict("os.environ", {}, clear=True):
            result = forecaster.run_claude("q", {}, "s")
        self.assertFalse(result.ok)
        self.assertIn("OPENROUTER_API_KEY", result.error)


class LocalAskTests(unittest.TestCase):
    def test_own_question_prints_answer_and_logs_locally(self):
        import contextlib
        import io
        import json
        import tempfile
        from unittest import mock

        import bot
        from forecaster import Forecast

        fake = Forecast(value=0.4, summary="40.0%", comment="**Forecast: 40.0%**\n\nreason", runs=[])
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "log.jsonl"
            output = io.StringIO()
            with mock.patch.object(bot, "forecast_question", return_value=fake) as forecast, \
                    mock.patch.object(bot, "LOG_FILE", log), \
                    mock.patch.object(sys, "argv", ["bot.py", "--ask", "Will X happen?", "--criteria", "By 2027."]), \
                    contextlib.redirect_stdout(output):
                code = bot.main()
            question = forecast.call_args.args[0]
            entry = json.loads(log.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual((question.type, question.title, question.resolution_criteria),
                         ("binary", "Will X happen?", "By 2027."))
        self.assertIn("40.0%", output.getvalue())
        self.assertEqual(entry["status"], "local")


if __name__ == "__main__":
    unittest.main()
