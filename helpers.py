import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np

STOP = set("""a an the and or but if so to of in on at for with is are was were be been
it this that these those i you he she we they me my your our their his her its
as by from up out just like really very got get going gonna yeah okay oh um uh
then than too not no yes do does did have has had can could would should will
what when where who why how there here about into over after before all some
any more most one two know think see say said make made right now well""".split())

VIRAL_TAGS = ["#fyp", "#foryou", "#viral", "#trending"]
GAMING_TAGS = ["#gaming", "#twitch", "#streamer", "#clips", "#gamingclips"]

# General TikTok peak windows (UTC). Best days: Tue, Thu, Fri.
# Peak hours: late morning, early evening, and late night.
PEAK_HOURS_UTC = [(15, "evening in the US"), (19, "prime time in the US"),
                  (11, "lunch in Europe")]
BEST_DAYS = {1, 3, 4}  # Mon=0 ... Sun=6 -> Tue, Thu, Fri


def keywords(text, n=5):
    words = re.findall(r"[a-zA-Z']{4,}", text.lower())
    words = [w for w in words if w not in STOP]
    return [w for w, _ in Counter(words).most_common(n)]


def suggest_hashtags(text, n_keywords=5):
    tags = ["#" + w.replace("'", "") for w in keywords(text, n_keywords)]
    tags += GAMING_TAGS[:2] + VIRAL_TAGS[:3]
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:10]


def best_post_time(now=None):
    """Next good posting slot, returned as a UTC datetime and a label."""
    now = now or datetime.now(timezone.utc)
    slots = []
    for d in range(0, 8):
        day = now + timedelta(days=d)
        if day.weekday() not in BEST_DAYS:
            continue
        for hour, label in PEAK_HOURS_UTC:
            t = day.replace(hour=hour, minute=0, second=0, microsecond=0)
            if t > now + timedelta(minutes=30):
                slots.append((t, label))
    slots.sort()
    return slots[0] if slots else (now + timedelta(hours=2), "soon")


def viral_score(loud, speech, start, end, text, duration_total):
    """0-100 score from energy, talk density, a strong hook and ideal length."""
    a, b = int(start), int(end)
    energy = float(np.mean(loud[a:b])) if b > a else 0.0
    talk = float(np.mean(speech[a:b])) if b > a else 0.0
    peak = float(np.max(loud[a:b])) if b > a else 0.0
    first = " ".join(text.split()[:20]).lower()
    hook = 1.0 if re.search(r"\?|\bwow\b|\bno way\b|\bwait\b|\bwhat\b|\bomg\b|\bcrazy\b|!", first) else 0.4
    length = end - start
    length_fit = 1.0 if 30 <= length <= 60 else 0.7
    score = (0.35 * energy + 0.25 * peak + 0.2 * talk + 0.12 * hook
             + 0.08 * length_fit)
    return int(round(min(1.0, score * 1.25) * 100))


def clip_text(segments, start, end):
    return " ".join(s.text.strip() for s in segments
                    if s.end >= start and s.start <= end)