"""In-memory query layer over the mod-log TSV snapshot.

The TSV is loaded into DuckDB once at process start. Every tool call is then a
parameterised SQL query against memory, which takes single-digit milliseconds
for 15k rows, so the tools never become the latency bottleneck.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import duckdb
from pydantic import BaseModel, Field

Action = Literal["none", "warn", "mute", "timeout", "kick", "delete", "ban"]
MessageType = Literal["message", "reply", "thread_created", "pin", "member_join", "member_leave"]
GroupBy = Literal["mod", "author", "action", "channel", "message_type", "reason", "day", "week", "month"]

GROUP_EXPR: dict[str, str] = {
    "mod": "mod_username",
    "author": "author_username",
    "action": "action_taken",
    "channel": "channel_name",
    "message_type": "message_type",
    "reason": "reason",
    "day": "strftime(ts, '%Y-%m-%d')",
    "week": "strftime(date_trunc('week', ts), '%Y-%m-%d')",
    "month": "strftime(ts, '%Y-%m')",
}


class EventFilter(BaseModel):
    """Filters shared by every tool. All fields are optional and combine with AND."""

    actions: list[Action] | None = Field(None, description="Mod actions to include. Use ['warn','mute','timeout','kick','delete','ban'] for 'any mod action'.")
    message_types: list[MessageType] | None = None
    channel: str | None = Field(None, description="Channel name including '#', e.g. '#general'.")
    mod: str | None = Field(None, description="Moderator username (case-insensitive, exact).")
    author: str | None = Field(None, description="Author username. Exact match first; falls back to a partial match.")
    since: date | None = Field(None, description="Inclusive start date (UTC).")
    until: date | None = Field(None, description="Inclusive end date (UTC).")
    text: str | None = Field(None, description="Case-insensitive substring of the message content or the mod's reason.")


class EventRow(BaseModel):
    time: str
    channel: str
    author: str
    type: str
    action: str
    mod: str | None
    reason: str | None
    content: str


class EventsResult(BaseModel):
    total_matches: int
    returned: int
    rows: list[EventRow]
    authors_matched: list[str] | None = Field(None, description="Set when 'author' was a partial match, so you can tell the admin which users were included.")


class GroupCount(BaseModel):
    key: str | None
    count: int


class AggregateResult(BaseModel):
    total_matches: int
    group_by: str
    groups: list[GroupCount]
    authors_matched: list[str] | None = None


@dataclass
class ModLogStore:
    con: duckdb.DuckDBPyConnection
    path: Path
    _lock: threading.Lock

    @classmethod
    def load(cls, path: str | Path) -> "ModLogStore":
        path = Path(path)
        con = duckdb.connect(":memory:")
        con.execute(
            """
            CREATE TABLE events AS
            SELECT
                message_id, channel_id, channel_name, author_id, author_username,
                CAST(timestamp_utc AS TIMESTAMP) AS ts,
                message_type, action_taken,
                NULLIF(reason, '') AS reason,
                NULLIF(mod_id, '') AS mod_id,
                NULLIF(mod_username, '') AS mod_username,
                NULLIF(target_user_id, '') AS target_user_id,
                replace(replace(message_content, '\\n', chr(10)), '\\t', chr(9)) AS content,
                CAST(attachment_count AS INTEGER) AS attachment_count,
                TRY_CAST(NULLIF(edited_at_utc, '') AS TIMESTAMP) AS edited_at
            FROM read_csv(?, delim='\t', header=true, all_varchar=true, quote='', escape='')
            """,
            [str(path)],
        )
        con.execute("CREATE INDEX idx_author ON events(author_username)")
        # Lock the database down now that the data is in memory. Model-written SQL
        # (sql / discover modes) can then never read files, reach the network, or
        # re-enable either.
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        return cls(con=con, path=path, _lock=threading.Lock())

    # ---- helpers -----------------------------------------------------------

    def _sql(self, query: str, params: list) -> list[tuple]:
        # One shared in-memory connection; DuckDB cursors are cheap and thread safe.
        with self._lock:
            cur = self.con.cursor()
        return cur.execute(query, params).fetchall()

    def _resolve_author(self, author: str) -> tuple[list[str], bool]:
        exact = self._sql("SELECT DISTINCT author_username FROM events WHERE lower(author_username) = lower(?)", [author])
        if exact:
            return [exact[0][0]], False
        partial = self._sql(
            "SELECT DISTINCT author_username FROM events WHERE author_username ILIKE ? ORDER BY 1 LIMIT 25",
            [f"%{author}%"],
        )
        return [r[0] for r in partial], True

    def _where(self, f: EventFilter) -> tuple[str, list, list[str] | None]:
        clauses, params = ["TRUE"], []
        authors_matched = None
        if f.actions:
            clauses.append(f"action_taken IN ({','.join('?' * len(f.actions))})")
            params += f.actions
        if f.message_types:
            clauses.append(f"message_type IN ({','.join('?' * len(f.message_types))})")
            params += f.message_types
        if f.channel:
            ch = f.channel if f.channel.startswith("#") else f"#{f.channel}"
            clauses.append("lower(channel_name) = lower(?)")
            params.append(ch)
        if f.mod:
            clauses.append("lower(mod_username) = lower(?)")
            params.append(f.mod)
        if f.author:
            names, partial = self._resolve_author(f.author)
            if not names:
                clauses.append("FALSE")
            else:
                clauses.append(f"author_username IN ({','.join('?' * len(names))})")
                params += names
                if partial:
                    authors_matched = names
        if f.since:
            clauses.append("ts >= ?")
            params.append(f.since)
        if f.until:
            clauses.append("ts < ? + INTERVAL 1 DAY")
            params.append(f.until)
        if f.text:
            clauses.append("(content ILIKE ? OR reason ILIKE ?)")
            params += [f"%{f.text}%", f"%{f.text}%"]
        return " AND ".join(clauses), params, authors_matched

    # ---- tool backends -----------------------------------------------------

    def find_events(self, f: EventFilter, limit: int = 20, newest_first: bool = True) -> EventsResult:
        where, params, matched = self._where(f)
        total = self._sql(f"SELECT count(*) FROM events WHERE {where}", params)[0][0]
        order = "DESC" if newest_first else "ASC"
        rows = self._sql(
            f"""SELECT strftime(ts, '%Y-%m-%d %H:%M'), channel_name, author_username, message_type,
                       action_taken, mod_username, reason, left(content, 240)
                FROM events WHERE {where} ORDER BY ts {order} LIMIT ?""",
            params + [limit],
        )
        return EventsResult(
            total_matches=total,
            returned=len(rows),
            rows=[EventRow(time=r[0], channel=r[1], author=r[2], type=r[3], action=r[4],
                           mod=r[5], reason=r[6], content=r[7]) for r in rows],
            authors_matched=matched,
        )

    def aggregate(self, f: EventFilter, group_by: GroupBy, top_n: int = 15) -> AggregateResult:
        where, params, matched = self._where(f)
        expr = GROUP_EXPR[group_by]
        total = self._sql(f"SELECT count(*) FROM events WHERE {where}", params)[0][0]
        order = "key ASC" if group_by in ("day", "week", "month") else "count DESC, key ASC"
        rows = self._sql(
            f"SELECT {expr} AS key, count(*) AS count FROM events WHERE {where} GROUP BY 1 ORDER BY {order} LIMIT ?",
            params + [top_n],
        )
        return AggregateResult(total_matches=total, group_by=group_by,
                               groups=[GroupCount(key=r[0], count=r[1]) for r in rows],
                               authors_matched=matched)

    def snapshot_summary(self) -> str:
        """Compact overview baked into the agent's instructions (and prompt-cached)."""
        (n, lo, hi) = self._sql("SELECT count(*), min(ts), max(ts) FROM events", [])[0]

        def counts(expr: str, where: str = "TRUE") -> str:
            rows = self._sql(f"SELECT {expr}, count(*) c FROM events WHERE {where} GROUP BY 1 ORDER BY c DESC", [])
            return ", ".join(f"{k}={c}" for k, c in rows)

        acted = "action_taken <> 'none'"
        return (
            f"Rows: {n}. Covers {lo:%Y-%m-%d %H:%M} to {hi:%Y-%m-%d %H:%M} UTC.\n"
            f"Actions: {counts('action_taken')}\n"
            f"Message types: {counts('message_type')}\n"
            f"Channels: {counts('channel_name')}\n"
            f"Mod actions by moderator: {counts('mod_username', acted)}\n"
            f"Moderators: {', '.join(r[0] for r in self._sql('SELECT DISTINCT mod_username FROM events WHERE mod_username IS NOT NULL ORDER BY 1', []))}"
        )

    # ---- raw SQL backends (sql / discover modes) -----------------------------

    SQL_TIMEOUT_S = 5.0
    SQL_MAX_ROWS = 50
    SQL_MAX_CHARS = 6000

    def run_sql(self, sql: str) -> str:
        """Run one read-only SELECT and return a compact pipe-separated table.

        Raises ValueError with a model-readable message on rejected or failing SQL.
        """
        try:
            stmts = duckdb.extract_statements(sql)
        except duckdb.Error as e:
            raise ValueError(f"SQL parse error: {e}") from None
        if len(stmts) != 1:
            raise ValueError("Send exactly one statement per call.")
        if stmts[0].type != duckdb.StatementType.SELECT:
            raise ValueError(f"Only read-only SELECT queries are allowed (got {stmts[0].type.name}).")

        with self._lock:
            cur = self.con.cursor()
        timer = threading.Timer(self.SQL_TIMEOUT_S, cur.interrupt)
        timer.start()
        try:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(self.SQL_MAX_ROWS + 1)
        except duckdb.InterruptException:
            raise ValueError(f"Query exceeded {self.SQL_TIMEOUT_S:.0f}s; aggregate or filter more.") from None
        except duckdb.Error as e:
            raise ValueError(f"{type(e).__name__}: {e}") from None
        finally:
            timer.cancel()

        truncated = len(rows) > self.SQL_MAX_ROWS
        rows = rows[: self.SQL_MAX_ROWS]
        cell = lambda v: "NULL" if v is None else str(v).replace("\n", "\\n").replace("|", "\\|")  # noqa: E731
        lines = [" | ".join(cols)] + [" | ".join(cell(v) for v in r) for r in rows]
        out = "\n".join(lines)
        if len(out) > self.SQL_MAX_CHARS:
            out = out[: self.SQL_MAX_CHARS] + "\n…[output truncated: select fewer columns or aggregate]"
        footer = f"\n({len(rows)} row{'s' * (len(rows) != 1)}"
        footer += f"; more rows exist, only first {self.SQL_MAX_ROWS} shown — use LIMIT/aggregation)" if truncated else ")"
        return out + footer

    def list_tables(self) -> str:
        rows = self._sql("SELECT table_name, estimated_size FROM duckdb_tables() ORDER BY 1", [])
        return "\n".join(f"{t} (~{n} rows)" for t, n in rows)

    def describe_table(self, table: str) -> str:
        known = {r[0] for r in self._sql("SELECT table_name FROM duckdb_tables()", [])}
        if table not in known:
            raise ValueError(f"Unknown table {table!r}. Known tables: {', '.join(sorted(known))}")
        cols = self._sql(f'DESCRIBE "{table}"', [])
        lines = ["column | type | distinct | nulls | examples"]
        for name, typ, *_ in cols:
            distinct, nulls = self._sql(
                f'SELECT count(DISTINCT "{name}"), count(*) - count("{name}") FROM "{table}"', [])[0]
            # Low-cardinality columns list every value (that's what a model needs to filter
            # correctly); high-cardinality ones show a few examples.
            n = 20 if distinct <= 20 else 4
            examples = self._sql(
                f'SELECT DISTINCT left(CAST("{name}" AS VARCHAR), 40) AS v FROM "{table}" '
                f'WHERE "{name}" IS NOT NULL ORDER BY v LIMIT {n}', [])
            ex = ", ".join(repr(e[0]) for e in examples) + ("" if distinct <= 20 else ", …")
            lines.append(f"{name} | {typ} | {distinct} | {nulls} | {ex}")
        return "\n".join(lines)

    def schema_ddl(self) -> str:
        return self._sql("SELECT sql FROM duckdb_tables() WHERE table_name = 'events'", [])[0][0]

    @property
    def as_of(self) -> date:
        return self._sql("SELECT max(ts)::DATE FROM events", [])[0][0]
