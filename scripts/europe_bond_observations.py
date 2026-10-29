"""Strict parsing of Borsa/Radiocor's public daily euro sovereign brief.

No network, files, orders, or strategy imports. A caller may fetch the English
archive and follow its actual article/Next links with its own bounded public
transport. HTTP denial is not permission to guess URLs or switch transports.
The benchmark is a provider-labelled generic 10Y yield, not a specific ISIN,
auction yield, zero-coupon curve, executable quote, or intraday risk input.
"""
from __future__ import annotations

import copy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from html.parser import HTMLParser
import re
from urllib.parse import parse_qsl, urljoin, urlsplit

INDEX_URL = "https://www.borsaitaliana.it/borsa/notizie/archivi/radiocor.html?lang=en"
PROVIDER = "Il Sole 24 Ore Radiocor / Borsa Italiana"
TITLE = "Euro zone benchmark 10-year sovereign bond yields"
MAX_BODY_BYTES = 2 * 1024 * 1024
_HOST = "www.borsaitaliana.it"
_INDEX_PATH = "/borsa/notizie/archivi/radiocor.html"
_ARTICLE = re.compile(
    r"/borsa/notizie/radiocor/finance/dettaglio/"
    r"euro-zone-benchmark-10year-sovereign-bond-yields-nRC_"
    r"(?P<day>\d{2})(?P<month>\d{2})(?P<year>20\d{2})_"
    r"(?P<hour>\d{2})(?P<minute>\d{2})_(?P<id>\d{1,20})\.html\Z")
_COUNTRIES = {"Germany": "de", "France": "fr", "Italy": "it",
              "Spain": "es", "Belgium": "be"}
_MONTHS = {name: i for i, name in enumerate(
    ("gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"), 1)}


class EuropeBondError(ValueError):
    """The source cannot support the claimed identity, date, or unit."""


def _ms(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EuropeBondError("positive_unix_milliseconds_required")
    try:
        datetime.fromtimestamp(value / 1000, timezone.utc)
    except (ValueError, OverflowError, OSError):
        raise EuropeBondError("timestamp_out_of_range") from None
    return value


def _text(payload: bytes | str) -> tuple[str, str]:
    if isinstance(payload, str):
        try:
            raw = payload.encode("utf-8")
        except UnicodeEncodeError:
            raise EuropeBondError("invalid_utf8") from None
    elif isinstance(payload, bytes):
        raw = payload
    else:
        raise EuropeBondError("html_bytes_or_text_required")
    if not raw or len(raw) > MAX_BODY_BYTES:
        raise EuropeBondError("empty_or_oversized_body")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise EuropeBondError("invalid_utf8") from None
    return text, hashlib.sha256(raw).hexdigest()


def _space(text: str) -> str:
    return " ".join(text.split())


def _url(url: str, *, article: bool) -> tuple[str, re.Match | None]:
    if (not isinstance(url, str) or len(url) > 2048 or "\\" in url
            or not re.fullmatch(r"[\x21-\x7e]+", url)):
        raise EuropeBondError("invalid_source_url")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise EuropeBondError("invalid_source_url") from None
    if (parts.scheme != "https" or parts.hostname != _HOST or port is not None
            or parts.username or parts.password or parts.fragment):
        raise EuropeBondError("borsa_https_source_required")
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    if len({key for key, _ in pairs}) != len(pairs):
        raise EuropeBondError("duplicate_url_query")
    query = dict(pairs)
    match = _ARTICLE.fullmatch(parts.path) if article else None
    if article:
        if not match or query != {"lang": "en"}:
            raise EuropeBondError("unexpected_article_identity")
        try:
            date(int(match["year"]), int(match["month"]), int(match["day"]))
            datetime(2000, 1, 1, int(match["hour"]), int(match["minute"]))
        except ValueError:
            raise EuropeBondError("invalid_article_date") from None
    elif (parts.path != _INDEX_PATH or query.get("lang") != "en"
          or set(query) - {"lang", "page"}
          or ("page" in query and (not re.fullmatch(r"[1-9]\d{0,2}", query["page"])
                                    or int(query["page"]) > 500))):
        raise EuropeBondError("unexpected_index_identity")
    return url, match


class _BriefHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.titles: list[str] = []
        self.paragraphs: list[str] = []
        self._h1: list[str] | None = None
        self._paragraph: list[str] | None = None
        self._span_depth = 0
        self.containers = 0
        self._ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._ignored += 1
        if self._ignored:
            return
        if tag == "h1":
            self._h1 = []
        if tag == "span":
            classes = set(dict(attrs).get("class", "").split())
            if self._span_depth:
                self._span_depth += 1
            elif classes == {"t-text", "-block", "-line-1-3em"}:
                self.containers += 1
                self._span_depth = 1
        if tag == "p" and self._span_depth:
            if self._paragraph is not None:
                raise EuropeBondError("nested_report_paragraph")
            self._paragraph = []
        if tag == "br" and self._paragraph is not None:
            self._paragraph.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self._ignored:
            self._ignored -= 1
            return
        if self._ignored:
            return
        if tag == "h1" and self._h1 is not None:
            self.titles.append(_space("".join(self._h1)))
            self._h1 = None
        if tag == "p" and self._paragraph is not None:
            self.paragraphs.append(_space("".join(self._paragraph)))
            self._paragraph = None
        if tag == "span" and self._span_depth:
            self._span_depth -= 1

    def handle_data(self, data):
        if not self._ignored:
            if self._h1 is not None:
                self._h1.append(data)
            if self._paragraph is not None:
                self._paragraph.append(data)


class _IndexHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.archive_count = 0
        self.in_archive = 0
        self.links: list[tuple[str, str]] = []
        self.next_links: list[str] = []
        self._anchor = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "ul":
            if self.in_archive:
                self.in_archive += 1
            elif attrs.get("id") == "itemListArchive":
                self.archive_count += 1
                self.in_archive = 1
        if tag == "a":
            if self.in_archive:
                self._anchor = [attrs.get("href", ""), []]
            elif attrs.get("title") == "Next":
                self.next_links.append(attrs.get("href", ""))

    def handle_endtag(self, tag):
        if tag == "a" and self._anchor is not None:
            self.links.append((self._anchor[0], _space("".join(self._anchor[1]))))
            self._anchor = None
        if tag == "ul" and self.in_archive:
            self.in_archive -= 1

    def handle_data(self, data):
        if self._anchor is not None:
            self._anchor[1].append(data)


def discover_europe_bond_briefs(payload: bytes | str, *, index_url: str = INDEX_URL,
                                asof_ms: int) -> dict:
    """Return only real links on the English archive, newest date first.

    This does not claim the first candidate is today's print. The caller must
    parse it and inspect freshness, and may follow only next_page_url within its
    own finite page/request budget. No URL is invented from today's date.
    """
    _ms(asof_ms)
    _url(index_url, article=False)
    text, digest = _text(payload)
    parser = _IndexHTML()
    parser.feed(text)
    parser.close()
    if parser.archive_count != 1 or parser.in_archive or not parser.links:
        raise EuropeBondError("archive_structure_changed")
    candidates = {}
    today = datetime.fromtimestamp(asof_ms / 1000, timezone.utc).date()
    for href, label in parser.links:
        if label != TITLE:
            if "euro-zone-benchmark-10year-sovereign-bond-yields" in href:
                raise EuropeBondError("article_title_identity_changed")
            continue
        url, match = _url(urljoin(index_url, href), article=True)
        day = date(int(match["year"]), int(match["month"]), int(match["day"]))
        if day > today:
            raise EuropeBondError("future_article_date")
        candidates[url] = {"url": url, "article_date": day.isoformat(),
                           "publication_wall_minute": f'{match["hour"]}:{match["minute"]}',
                           "publication_timezone": "unspecified_by_source"}
    next_urls = set()
    page = int(dict(parse_qsl(urlsplit(index_url).query)).get("page", "1"))
    for href in parser.next_links:
        url, _ = _url(urljoin(index_url, href), article=False)
        next_page = int(dict(parse_qsl(urlsplit(url).query)).get("page", "1"))
        if next_page != page + 1:
            raise EuropeBondError("invalid_next_page_sequence")
        next_urls.add(url)
    if len(next_urls) > 1:
        raise EuropeBondError("ambiguous_next_page")
    ordered = sorted(candidates.values(), key=lambda row: (
        row["article_date"], row["publication_wall_minute"], row["url"]), reverse=True)
    return {"schema_version": 1, "source_status": "discovered" if ordered else "missing",
            "index_url": index_url, "index_body_sha256": digest,
            "candidates": ordered, "next_page_url": next(iter(next_urls), None),
            "automatic_url_guessing": False}


def parse_europe_bond_brief(payload: bytes | str, *, source_url: str,
                            received_at_ms: int, asof_ms: int,
                            max_age_days: int = 4, previous: dict | None = None) -> dict:
    """Parse a dated same-source daily snapshot; reject future or changed shape.

    Explicit GMT observation time is preserved, but frequency is daily and
    eligible_intraday is ALWAYS false. Re-fetches of identical report content
    retain first availability. Stale data remains inspectable and ineligible.
    """
    _ms(received_at_ms)
    _ms(asof_ms)
    if previous is not None and not isinstance(previous, dict):
        raise EuropeBondError("previous_snapshot_object_required")
    if received_at_ms > asof_ms:
        raise EuropeBondError("receipt_after_asof")
    if isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or not 0 <= max_age_days <= 7:
        raise EuropeBondError("max_age_days_must_be_zero_to_seven")
    _, match = _url(source_url, article=True)
    text, body_digest = _text(payload)
    parser = _BriefHTML()
    parser.feed(text)
    parser.close()
    if (parser.containers != 1 or parser._span_depth or parser._paragraph is not None
            or [title for title in parser.titles if title] != [TITLE]
            or len(parser.paragraphs) not in {6, 7, 8}):
        raise EuropeBondError("brief_structure_changed")
    # The verified October 27/28 brief splits one country row after "from the."
    # into an isolated "previous close." paragraph. Repair only this exact
    # source formatting artifact inside the already identified report. The
    # complete row is still validated below; arbitrary extra text is rejected.
    lines = []
    joined_previous_close = False
    for line in parser.paragraphs:
        if line == "previous close.":
            if (joined_previous_close or not lines or not re.fullmatch(
                    r"(?:Germany|France|Italy|Spain|Belgium) - yield -?\d+(?:\.\d{1,3})?%; "
                    r"(?:(?:up|down) \d+(?:\.\d{1,3})? percentage points|unchanged) from the\.",
                    lines[-1])):
                raise EuropeBondError("brief_structure_changed")
            lines[-1] = lines[-1][:-1] + " previous close."
            joined_previous_close = True
        else:
            lines.append(line)
    if len(lines) not in {6, 7}:
        raise EuropeBondError("brief_structure_changed")
    header = re.fullmatch(
        r"\(Il Sole 24 Ore Radiocor\) - Milano, (\d{1,2}) ([a-z]{3}) - "
        r"Euro zone benchmark 10-year sovereign bond yields at (\d{2})(\d{2}) GMT\.", lines[0])
    footer = re.fullmatch(
        r"\(RADIOCOR\) (\d{2})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) \(\d+\) \d+ NNNN",
        lines[-1])
    if not header or not footer:
        raise EuropeBondError("report_date_or_gmt_time_missing")
    try:
        day = date(int(match["year"]), int(match["month"]), int(match["day"]))
        footer_day = date(2000 + int(footer[3]), int(footer[2]), int(footer[1]))
        wall_time = datetime.combine(day, datetime.min.time()).replace(
            hour=int(footer[4]), minute=int(footer[5]), second=int(footer[6]))
        observed = datetime(day.year, day.month, day.day, int(header[3]), int(header[4]), tzinfo=timezone.utc)
    except (ValueError, KeyError):
        raise EuropeBondError("invalid_report_date_or_time") from None
    if (day != footer_day or int(header[1]) != day.day or _MONTHS.get(header[2]) != day.month
            or footer[4] != match["hour"] or footer[5] != match["minute"]):
        raise EuropeBondError("url_and_report_date_disagree")
    observed_ms = int(observed.timestamp() * 1000)
    if observed_ms > received_at_ms or day > datetime.fromtimestamp(asof_ms / 1000, timezone.utc).date():
        raise EuropeBondError("future_observation")
    values = {}
    for line in lines[1:-1]:
        row = re.fullmatch(
            r"(Germany|France|Italy|Spain|Belgium) - yield (-?\d+(?:\.\d{1,3})?)%; "
            r"(?:(up|down) (\d+(?:\.\d{1,3})?) percentage points from the previous close|"
            r"(unchanged) from the previous close)\.?", line)
        if not row or row[1] in values:
            raise EuropeBondError("yield_row_missing_duplicate_or_changed")
        try:
            value = Decimal(row[2])
            change = Decimal(row[4] or "0") * (-1 if row[3] == "down" else 1)
        except InvalidOperation:
            raise EuropeBondError("invalid_percent_value") from None
        if not -5 <= value <= 30 or not -10 <= change <= 10:
            raise EuropeBondError("implausible_percent_value")
        values[row[1]] = (value, change)
    if not {"Germany", "France", "Italy", "Spain"} <= values.keys():
        raise EuropeBondError("required_same_snapshot_countries_missing")
    canonical = "\n".join([source_url, *lines])
    snapshot_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    first = received_at_ms
    if previous is not None and previous.get("snapshot_id") == snapshot_id:
        prior_first = previous.get("first_seen_at_ms")
        _ms(prior_first)
        if not observed_ms <= prior_first <= received_at_ms:
            raise EuropeBondError("invalid_previous_availability")
        first = prior_first
    age_days = (datetime.fromtimestamp(asof_ms / 1000, timezone.utc).date() - day).days
    current = age_days <= max_age_days
    valid_until_ms = int(datetime.combine(day + timedelta(days=max_age_days + 1),
                                         datetime.min.time(), timezone.utc).timestamp() * 1000)
    freshness = {"status": "daily_current" if current else "stale", "age_days": age_days,
                 "age_ms": asof_ms - observed_ms, "max_age_days": max_age_days,
                 "valid_until_ms": valid_until_ms, "eligible_daily": current,
                 "eligible_intraday": False}
    provenance = {"provider": PROVIDER, "source_url": source_url,
                  "source_body_sha256": body_digest, "canonical_report_sha256": snapshot_id,
                  "format_normalizations": (["joined_previous_close_paragraph"]
                                            if joined_previous_close else []),
                  "publication_wall_time": wall_time.isoformat(),
                  "publication_timezone": "unspecified_by_source", "published_at_ms": None,
                  "publication_time_basis": "RADIOCOR_footer_wall_time_not_assumed_UTC",
                  "retrieved_at_ms": received_at_ms}
    shared = {"provider": PROVIDER, "source_url": source_url, "snapshot_id": snapshot_id,
              "observed_at_ms": observed_ms, "observed_date": day.isoformat(),
              "precision": "timestamp", "observation_timezone": "UTC", "frequency": "daily",
              "time_basis": "daily_brief_explicit_GMT_minute_not_executable_quote",
              "first_seen_at_ms": first, "available_at_ms": first, "received_at_ms": first,
              "eligible_daily": current, "eligible_intraday": False,
              "freshness": freshness, "provenance": provenance}
    observations = {}
    for country, (value, change) in values.items():
        code = _COUNTRIES[country]
        key = f"{code}_10y_pct"
        observations[key] = {**copy.deepcopy(shared), "key": key, "value": float(value), "unit": "percent",
            "change_from_previous_close_pp": float(change),
            "instrument": {"country": code.upper(), "tenor": "10Y", "currency": "EUR",
                           "kind": "provider_generic_sovereign_benchmark_yield",
                           "name": f"{country} benchmark 10-year sovereign bond yield",
                           "isin": None, "tradable_quote": False}}
    spreads = {}
    for country, (value, _) in values.items():
        if country == "Germany":
            continue
        code = _COUNTRIES[country]
        key = f"{code}_de_10y_spread_bp"
        spreads[key] = {**copy.deepcopy(shared), "key": key, "value": float((value - values["Germany"][0]) * 100),
                        "unit": "basis_points", "components": [f"{code}_10y_pct", "de_10y_pct"],
                        "instrument": {"kind": "same_report_10Y_sovereign_benchmark_spread",
                                       "formula": "100 * (left_percent - Germany_percent)",
                                       "same_snapshot_required": True, "tradable_quote": False}}
    return {"schema_version": 1, "mode": "OBSERVATION_ONLY", "places_orders": False,
            "status": "daily_current" if current else "stale", "source_status": "parsed",
            "snapshot_id": snapshot_id, "first_seen_at_ms": first, "frequency": "daily",
            "observed_at_ms": observed_ms, "observed_date": day.isoformat(),
            "eligible_intraday": False, "freshness": freshness, "provenance": provenance,
            "observations": observations, "derived_spreads": spreads}
