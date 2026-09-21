"""The report's arithmetic, which is the part that can be wrong quietly."""

from __future__ import annotations

from datetime import datetime, timedelta

import main


def attempt(product: str, user: str, skill: str, total: float, stamp: str) -> main.Attempt:
    return main.Attempt(
        product=product,
        user_id=user,
        name=f"name-{user}",
        experiment_number=f"CG{user[-2:]}",
        skill=skill,
        total=total,
        at=datetime.strptime(stamp, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=main.TAIPEI),
    )


def test_one_row_per_student_per_stroke_holding_their_best() -> None:
    attempts = [
        attempt("no-llm", "u01", "serve", 61.0, "2026-09-21-10-00-00"),
        attempt("no-llm", "u01", "serve", 74.5, "2026-09-21-10-05-00"),
        attempt("no-llm", "u01", "serve", 58.0, "2026-09-22-10-00-00"),
        attempt("no-llm", "u01", "smash", 80.0, "2026-09-22-11-00-00"),
        attempt("no-llm", "u02", "serve", 66.0, "2026-09-23-09-00-00"),
    ]

    bests = main.weekly_bests(attempts, "2026-W39")

    assert [(b.experiment_number, b.skill, b.total, b.attempts) for b in bests] == [
        ("CG01", "serve", 74.5, 3),
        ("CG02", "serve", 66.0, 1),
        ("CG01", "smash", 80.0, 1),
    ]


def test_attempts_from_other_weeks_are_left_out() -> None:
    attempts = [
        attempt("no-llm", "u01", "serve", 90.0, "2026-09-14-10-00-00"),
        attempt("no-llm", "u01", "serve", 50.0, "2026-09-21-10-00-00"),
    ]

    bests = main.weekly_bests(attempts, "2026-W39")

    assert [b.total for b in bests] == [50.0], "last week's 90 is not this week's best"


def test_both_products_appear_separately() -> None:
    attempts = [
        attempt("llm", "u01", "serve", 70.0, "2026-09-21-10-00-00"),
        attempt("no-llm", "u02", "serve", 60.0, "2026-09-21-10-00-00"),
    ]

    bests = main.weekly_bests(attempts, "2026-W39")

    assert {b.product for b in bests} == {"llm", "no-llm"}


def test_the_trend_averages_each_student_once() -> None:
    attempts = [
        # One student with three attempts must not outweigh the other.
        attempt("no-llm", "u01", "serve", 40.0, "2026-09-21-10-00-00"),
        attempt("no-llm", "u01", "serve", 60.0, "2026-09-21-10-01-00"),
        attempt("no-llm", "u01", "serve", 50.0, "2026-09-21-10-02-00"),
        attempt("no-llm", "u02", "serve", 80.0, "2026-09-21-10-00-00"),
    ]

    series = main.weekly_means(attempts, ["2026-W38", "2026-W39"])

    assert series[("no-llm", "serve")] == [None, 70.0], "mean of 60 and 80"


def test_a_week_with_no_attempts_is_a_gap_not_a_zero() -> None:
    attempts = [attempt("no-llm", "u01", "serve", 70.0, "2026-09-21-10-00-00")]

    series = main.weekly_means(attempts, ["2026-W37", "2026-W38", "2026-W39"])

    assert series[("no-llm", "serve")] == [None, None, 70.0]


def test_only_completed_analyses_count() -> None:
    user = {
        "id": "u01",
        "name": "n",
        "experiment_number": "CG01",
        "portfolio": {
            "serve": {
                "2026-09-21-10-00-00-aaaa": {
                    "analysis_status": "completed",
                    "grading_outcome": {"total_grade": 70.0},
                    "date": "2026-09-21-10-00-00-aaaa",
                },
                "2026-09-21-10-10-00-bbbb": {
                    "analysis_status": "failed",
                    "grading_outcome": {"score_status": "failed"},
                    "date": "2026-09-21-10-10-00-bbbb",
                },
                "2026-09-21-10-20-00-cccc": {
                    "analysis_status": "pending",
                    "grading_outcome": {"total_grade": 99.0},
                    "date": "2026-09-21-10-20-00-cccc",
                },
            }
        },
    }

    found = main.attempts_of("no-llm", user)

    assert [item.total for item in found] == [70.0]


def test_the_week_label_follows_the_iso_calendar() -> None:
    monday = datetime(2026, 9, 21, 9, 0, tzinfo=main.TAIPEI)
    sunday = monday + timedelta(days=6)

    assert main.iso_week(monday) == main.iso_week(sunday) == "2026-W39"
    assert main.recent_weeks(monday.date(), 3) == ["2026-W37", "2026-W38", "2026-W39"]


def test_the_csv_has_a_header_and_one_line_per_best() -> None:
    attempts = [
        attempt("no-llm", "u01", "serve", 74.5, "2026-09-21-10-00-00"),
        attempt("no-llm", "u02", "serve", 66.0, "2026-09-21-11-00-00"),
    ]

    lines = main.report_csv(main.weekly_bests(attempts, "2026-W39")).strip().splitlines()

    assert lines[0].startswith("product,experiment_number,name,skill")
    assert len(lines) == 3
    assert "74.50" in lines[1]


def test_the_channel_token_is_read_out_of_the_bot_environment() -> None:
    blob = "\n".join(
        ["GCP_PROJECT_ID=x", 'LINE_CHANNEL_TOKEN="abc123"', "OPENAI_API_KEY=should-not-matter"]
    )

    assert main.channel_token(blob) == "abc123"


def test_a_chart_renders_even_with_nothing_to_plot() -> None:
    png = main.trend_png({}, ["2026-W38", "2026-W39"])

    assert png.startswith(b"\x89PNG"), "an empty class still gets a readable chart"
