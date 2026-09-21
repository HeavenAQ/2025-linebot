"""Weekly class report, pushed to the instructors over LINE.

One score per student per stroke: the best completed attempt of the week. The
CSV carries that table, the chart carries the class mean of those bests week by
week, so the trend is a line per stroke rather than a cloud of attempts.

Sent over LINE because a service account cannot send mail as a consumer Gmail
account, and both instructors already have the bot in their chat list.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import functions_framework
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import requests  # noqa: E402
from google.auth import default as default_credentials  # noqa: E402
from google.auth.transport.requests import Request as AuthRequest  # noqa: E402
from google.cloud import firestore, storage  # noqa: E402

PUSH_ENDPOINT = "https://api.line.me/v2/bot/message/push"
SKILLS = ("serve", "smash")
TAIPEI = timezone(timedelta(hours=8))
LINK_TTL = timedelta(days=7)


@dataclass(frozen=True)
class Attempt:
    """One completed, graded upload."""

    product: str
    user_id: str
    name: str
    experiment_number: str
    skill: str
    total: float
    at: datetime


@dataclass(frozen=True)
class Best:
    """A student's best attempt at one stroke in one week."""

    product: str
    experiment_number: str
    name: str
    skill: str
    total: float
    attempts: int
    at: datetime


def iso_week(moment: datetime | date) -> str:
    year, week, _ = moment.isocalendar()
    return f"{year}-W{week:02d}"


def attempt_time(key: str) -> datetime | None:
    """Portfolio keys start with the Taipei wall clock of the upload."""
    try:
        return datetime.strptime(key[:19], "%Y-%m-%d-%H-%M-%S").replace(tzinfo=TAIPEI)
    except (ValueError, IndexError):
        return None


def attempts_of(product: str, user: dict) -> list[Attempt]:
    """Every completed attempt in one learner's portfolio."""
    portfolio = user.get("portfolio") or {}
    found: list[Attempt] = []
    for skill in SKILLS:
        works = portfolio.get(skill) or {}
        if not isinstance(works, dict):
            continue
        for key, work in works.items():
            if not isinstance(work, dict):
                continue
            status = work.get("analysis_status")
            if status not in ("", None, "completed"):
                continue
            grading = work.get("grading_outcome") or {}
            total = grading.get("total_grade")
            moment = attempt_time(str(work.get("date") or key))
            if total is None or moment is None:
                continue
            found.append(
                Attempt(
                    product=product,
                    user_id=str(user.get("id") or ""),
                    name=str(user.get("name") or ""),
                    experiment_number=str(user.get("experiment_number") or ""),
                    skill=skill,
                    total=float(total),
                    at=moment,
                )
            )
    return found


def weekly_bests(attempts: list[Attempt], week: str) -> list[Best]:
    """One row per student per stroke: their best attempt that week."""
    grouped: dict[tuple[str, str, str], list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        if iso_week(attempt.at) == week:
            grouped[(attempt.product, attempt.user_id, attempt.skill)].append(attempt)

    bests: list[Best] = []
    for (product, _, skill), items in grouped.items():
        top = max(items, key=lambda item: item.total)
        bests.append(
            Best(
                product=product,
                experiment_number=top.experiment_number,
                name=top.name,
                skill=skill,
                total=top.total,
                attempts=len(items),
                at=top.at,
            )
        )
    bests.sort(key=lambda best: (best.product, best.skill, -best.total))
    return bests


def weekly_means(attempts: list[Attempt], weeks: list[str]) -> dict[tuple[str, str], list[float | None]]:
    """Class mean of the per-student bests, per product and stroke, per week.

    A week nobody attempted is None rather than zero: an empty week is a gap in
    the line, not a class that scored nothing.
    """
    series: dict[tuple[str, str], list[float | None]] = {}
    for week in weeks:
        for best in weekly_bests(attempts, week):
            series.setdefault((best.product, best.skill), [None] * len(weeks))
    for index, week in enumerate(weeks):
        totals: dict[tuple[str, str], list[float]] = defaultdict(list)
        for best in weekly_bests(attempts, week):
            totals[(best.product, best.skill)].append(best.total)
        for key, values in totals.items():
            series[key][index] = sum(values) / len(values)
    return series


def recent_weeks(today: date, count: int) -> list[str]:
    return [iso_week(today - timedelta(weeks=offset)) for offset in range(count - 1, -1, -1)]


def report_csv(bests: list[Best]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["product", "experiment_number", "name", "skill", "best_total", "attempts", "best_at"]
    )
    for best in bests:
        writer.writerow(
            [
                best.product,
                best.experiment_number,
                best.name,
                best.skill,
                f"{best.total:.2f}",
                best.attempts,
                best.at.strftime("%Y-%m-%d %H:%M"),
            ]
        )
    return buffer.getvalue()


def trend_png(series: dict[tuple[str, str], list[float | None]], weeks: list[str]) -> bytes:
    """A line per product and stroke. Labels stay in English: the container has
    no CJK font, and a chart of empty boxes helps nobody."""
    figure, axes = plt.subplots(figsize=(9, 5), dpi=140)
    positions = range(len(weeks))
    # Plot against positions, not the labels: a week with no attempts must keep
    # its place on the axis instead of disappearing from it.
    if series:
        for (product, skill), values in sorted(series.items()):
            axes.plot(
                positions,
                [float("nan") if value is None else value for value in values],
                marker="o",
                label=f"{product} · {skill}",
            )
        axes.legend(frameon=False, fontsize=9)
    else:
        axes.text(0.5, 0.5, "no graded attempts yet", ha="center", va="center",
                  transform=axes.transAxes, color="#777")
    axes.set_xticks(list(positions))
    axes.set_xticklabels(weeks)
    axes.set_title("Class weekly mean of each student's best attempt")
    axes.set_ylabel("score")
    axes.set_ylim(0, 100)
    axes.grid(axis="y", alpha=0.25)
    axes.spines[["top", "right"]].set_visible(False)
    figure.autofmt_xdate(rotation=35)
    figure.tight_layout()

    out = io.BytesIO()
    figure.savefig(out, format="png")
    plt.close(figure)
    return out.getvalue()


def signed_url(bucket: storage.Bucket, path: str, data: bytes, content_type: str,
               signer_email: str, token: str) -> str:
    """Upload and mint a read URL. Cloud Functions has no private key, so the
    signature is made through IAM with the running service account."""
    blob = bucket.blob(path)
    blob.upload_from_string(data, content_type=content_type)
    return blob.generate_signed_url(
        version="v4",
        expiration=LINK_TTL,
        method="GET",
        service_account_email=signer_email,
        access_token=token,
    )


def summary_text(week: str, bests: list[Best], csv_url: str) -> str:
    lines = [f"📊 本週班級報告 {week}"]
    if not bests:
        lines.append("這週還沒有完成的分析紀錄。")
    for product in sorted({best.product for best in bests}):
        for skill in SKILLS:
            rows = [b for b in bests if b.product == product and b.skill == skill]
            if not rows:
                continue
            mean = sum(row.total for row in rows) / len(rows)
            lines.append(
                f"{product}｜{skill}：{len(rows)} 人，平均 {mean:.1f}，最高 {rows[0].total:.1f}"
            )
    lines.append("")
    lines.append(f"CSV（7 天內有效）：\n{csv_url}")
    return "\n".join(lines)


def push(token: str, recipients: list[str], text: str, image_url: str) -> None:
    for recipient in recipients:
        response = requests.post(
            PUSH_ENDPOINT,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "to": recipient,
                "messages": [
                    {
                        "type": "image",
                        "originalContentUrl": image_url,
                        "previewImageUrl": image_url,
                    },
                    {"type": "text", "text": text},
                ],
            },
            timeout=20,
        )
        response.raise_for_status()


def channel_token(blob: str) -> str:
    """The bots keep their whole .env in one secret; this needs one line of it."""
    for line in blob.splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "LINE_CHANNEL_TOKEN":
            return value.strip().strip('"')
    raise RuntimeError("LINE_CHANNEL_TOKEN is not in the bot environment secret")


def products() -> list[dict]:
    return json.loads(os.environ["CLASS_REPORT_PRODUCTS"])


def excluded() -> set[str]:
    return {
        value.strip()
        for value in os.environ.get("CLASS_STATS_EXCLUDED_USER_IDS", "").split(",")
        if value.strip()
    }


def collect(project: str) -> list[Attempt]:
    """Every completed attempt from every product, minus the test accounts."""
    skip = excluded()
    attempts: list[Attempt] = []
    for product in products():
        client = firestore.Client(project=project, database=product["database"])
        for document in client.collection(product.get("collection", "users")).stream():
            user = document.to_dict() or {}
            if str(user.get("id") or document.id) in skip:
                continue
            attempts.extend(attempts_of(product["label"], user))
    return attempts


@functions_framework.http
def class_report(request):  # noqa: ARG001 - the scheduler sends an empty POST
    project = os.environ["GCP_PROJECT_ID"]
    week_count = int(os.environ.get("CLASS_REPORT_WEEKS", "8"))
    recipients = [
        value.strip()
        for value in os.environ.get("CLASS_REPORT_RECIPIENTS", "").split(",")
        if value.strip()
    ]
    if not recipients:
        return ("CLASS_REPORT_RECIPIENTS is empty", 400)

    today = datetime.now(TAIPEI).date()
    weeks = recent_weeks(today, week_count)
    attempts = collect(project)
    bests = weekly_bests(attempts, weeks[-1])

    credentials, _ = default_credentials()
    credentials.refresh(AuthRequest())
    bucket = storage.Client(project=project).bucket(os.environ["CLASS_REPORT_BUCKET"])
    prefix = f"class-reports/{weeks[-1]}"
    csv_url = signed_url(
        bucket, f"{prefix}/class-week.csv", report_csv(bests).encode("utf-8-sig"),
        "text/csv", credentials.service_account_email, credentials.token,
    )
    image_url = signed_url(
        bucket, f"{prefix}/weekly-trend.png", trend_png(weekly_means(attempts, weeks), weeks),
        "image/png", credentials.service_account_email, credentials.token,
    )

    push(channel_token(os.environ["BOT_ENV"]), recipients,
         summary_text(weeks[-1], bests, csv_url), image_url)
    return (f"sent {weeks[-1]}: {len(bests)} rows to {len(recipients)} recipients", 200)
