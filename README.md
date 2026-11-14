# Forecasting bot for the Metaculus FutureEval tournaments

An autonomous bot that forecasts every open question in the Metaculus FutureEval bot tournament and in MiniBench. There is no human in the loop: the bot reads the question, researches it on the web, submits the forecast and posts its reasoning as a private comment through the Metaculus API.

Copyright (c) 2026 stoplus. All rights reserved: the code is published for review only, see [LICENSE](LICENSE).

Bot account on Metaculus: `ContinuousCustodian-bot`. Status: tested end to end in the Metaculus bot testing area; the GitHub schedule is off until the bot has its own model credits.

## How it works

1. A GitHub Actions schedule starts the bot every 20 minutes. Questions stay open for three hours, so each one is seen several times before it closes.
2. `bot.py` lists the open questions of the configured tournaments and keeps those the bot has not forecast yet. It makes one forecast per question and never updates it.
3. For each question `forecaster.py` starts Claude Code in non-interactive mode (`claude -p`) with two tools only, web search and web fetch. The model gets the question text, resolution criteria, fine print, bounds and timing, and a fixed forecasting method: read the criteria literally, establish the current state of the world from primary sources, check whether the outcome is already determined, take the base rate, adjust for current evidence, and calibrate.
4. The answer comes back as structured JSON:
   - binary: a probability;
   - multiple choice: a probability per option;
   - numeric and discrete: thirteen percentiles from the 1st to the 99th.
5. When several runs are configured, they are combined: the median for binary questions (the mean in log-odds for two runs), the mean per option for multiple choice, and the median of each percentile for numeric questions.
6. `cdf.py` turns the percentiles into the cumulative distribution Metaculus expects (201 points, or one per outcome boundary for discrete questions) and enforces the API limits on minimum and maximum step and on probability mass outside the bounds.
7. Binary forecasts are kept between 1% and 99%. The range widens to 0.3%–99.7% only when every run reports that an authoritative source already shows the outcome.

The model process has no shell or file tools and does not receive the Metaculus token, so text on a web page cannot make it act on the account.

## Files

| File | Purpose |
|---|---|
| `bot.py` | Entry point: finds questions, runs the forecaster, submits forecasts and comments, writes `forecast-log.jsonl` |
| `forecaster.py` | Prompt, Claude Code invocation, aggregation of runs |
| `cdf.py` | Percentiles to a Metaculus-compliant CDF |
| `metaculus.py` | Metaculus API client |
| `tests/test_bot.py` | Offline tests |
| `.github/workflows/forecast.yml` | Schedule |

## Running

```
pip install -r requirements.txt
python -m unittest discover -s tests            # offline tests
python bot.py --question-file tests/questions/binary_loudoun.json   # forecast a local question, nothing is submitted
python bot.py --ask "Will X happen?" --criteria "Yes if ... before 2027-01-01"   # your own yes/no question, nothing is submitted
python bot.py --check                           # count open questions without a forecast
python bot.py --dry-run                         # forecast tournament questions without submitting
python bot.py                                   # forecast and submit
```

The bot needs Claude Code on the `PATH` and two secrets: `METACULUS_TOKEN` (the bot account's API token) and, in CI, `CLAUDE_CODE_OAUTH_TOKEN` (created with `claude setup-token`).

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `TOURNAMENTS` | `33121,minibench` | Tournament IDs or slugs, comma-separated |
| `RUNS_PER_QUESTION` | `1` | Independent model runs per question |
| `TOOL_BUDGET` | `10` | Searches and page reads the model may use per run |
| `MAX_RUN_COST_USD` | `1.5` | Hard cost cap per run at API list prices |
| `FORECAST_MODEL` | `fable` | Claude Code model alias |
| `FORECAST_EFFORT` | `high` | Reasoning effort |
| `QUESTION_CONCURRENCY` | `2` | Questions forecast in parallel |
| `SYNTHESIZE` | `0` | `1`: with two or more runs, a lead-forecaster pass reads all answers, discounts errors and writes the final forecast instead of a plain median |
| `JUDGE_TOOL_BUDGET` | `4` | Searches and page reads allowed to the lead-forecaster pass |
