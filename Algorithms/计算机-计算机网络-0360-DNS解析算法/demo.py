"""DNS iterative resolution MVP.

This demo models root -> TLD -> authoritative delegation using in-memory zones,
then runs an iterative resolver with TTL cache and CNAME following.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ResourceRecord:
    name: str
    rtype: str
    value: str
    ttl: int


@dataclass(frozen=True)
class Delegation:
    child_zone: str
    ns_host: str
    glue_ip: str
    ttl: int


@dataclass(frozen=True)
class Response:
    kind: str  # answer | cname | referral | nxdomain
    records: Tuple[ResourceRecord, ...] = ()
    next_zone: Optional[str] = None


@dataclass(frozen=True)
class CacheEntry:
    records: Tuple[ResourceRecord, ...]
    expire_at: int


@dataclass(frozen=True)
class ResolutionResult:
    query_name: str
    query_type: str
    final_records: Tuple[ResourceRecord, ...]
    trace: Tuple[str, ...]
    status: str


class AuthorityServer:
    """A tiny authoritative server model for one zone."""

    def __init__(
        self,
        zone: str,
        answers: Dict[Tuple[str, str], List[ResourceRecord]],
        cname_map: Dict[str, ResourceRecord],
        delegations: List[Delegation],
    ) -> None:
        self.zone = normalize_name(zone)
        self.answers = {
            (normalize_name(k[0]), k[1].upper()): tuple(v) for k, v in answers.items()
        }
        self.cname_map = {normalize_name(k): v for k, v in cname_map.items()}
        self.delegations = tuple(sorted(delegations, key=lambda d: len(d.child_zone), reverse=True))

    def query(self, qname: str, qtype: str) -> Response:
        qname = normalize_name(qname)
        qtype = qtype.upper()
        key = (qname, qtype)

        if key in self.answers:
            return Response(kind="answer", records=self.answers[key])

        if qtype != "CNAME" and qname in self.cname_map:
            return Response(kind="cname", records=(self.cname_map[qname],))

        for delegation in self.delegations:
            child = normalize_name(delegation.child_zone)
            if in_zone(qname, child):
                ns_rr = ResourceRecord(
                    name=child,
                    rtype="NS",
                    value=delegation.ns_host,
                    ttl=delegation.ttl,
                )
                glue_rr = ResourceRecord(
                    name=delegation.ns_host,
                    rtype="A",
                    value=delegation.glue_ip,
                    ttl=delegation.ttl,
                )
                return Response(kind="referral", records=(ns_rr, glue_rr), next_zone=child)

        return Response(kind="nxdomain")


class IterativeDNSResolver:
    def __init__(self, authorities: Dict[str, AuthorityServer]) -> None:
        self.authorities = {normalize_name(k): v for k, v in authorities.items()}
        self.cache: Dict[Tuple[str, str], CacheEntry] = {}
        self.clock = 0

    def resolve(self, qname: str, qtype: str = "A") -> ResolutionResult:
        self.clock += 1
        qname = normalize_name(qname)
        qtype = qtype.upper()

        trace: List[str] = [f"t={self.clock}: resolve {qname} {qtype}"]

        cached = self._cache_lookup(qname, qtype)
        if cached is not None:
            trace.append(f"cache-hit: {qname} {qtype}")
            return ResolutionResult(
                query_name=qname,
                query_type=qtype,
                final_records=cached,
                trace=tuple(trace),
                status="NOERROR",
            )

        current_name = qname
        current_zone = "."
        referral_hops = 0
        cname_hops = 0

        while referral_hops < 12:
            authority = self.authorities.get(current_zone)
            if authority is None:
                trace.append(f"error: missing authority for zone {current_zone}")
                return ResolutionResult(
                    query_name=qname,
                    query_type=qtype,
                    final_records=(),
                    trace=tuple(trace),
                    status="SERVFAIL",
                )

            response = authority.query(current_name, qtype)
            trace.append(
                f"ask zone={current_zone} name={current_name} type={qtype} -> {response.kind}"
            )

            if response.kind == "answer":
                self._cache_store(current_name, qtype, response.records)
                return ResolutionResult(
                    query_name=qname,
                    query_type=qtype,
                    final_records=response.records,
                    trace=tuple(trace),
                    status="NOERROR",
                )

            if response.kind == "cname":
                cname_rr = response.records[0]
                self._cache_store(cname_rr.name, "CNAME", (cname_rr,))
                current_name = normalize_name(cname_rr.value)
                current_zone = "."
                cname_hops += 1
                trace.append(f"follow-cname -> {current_name}")
                if cname_hops > 6:
                    trace.append("error: cname chain too deep")
                    return ResolutionResult(
                        query_name=qname,
                        query_type=qtype,
                        final_records=(),
                        trace=tuple(trace),
                        status="SERVFAIL",
                    )
                continue

            if response.kind == "referral" and response.next_zone is not None:
                self._cache_store(current_name, "NS", response.records)
                current_zone = normalize_name(response.next_zone)
                referral_hops += 1
                continue

            return ResolutionResult(
                query_name=qname,
                query_type=qtype,
                final_records=(),
                trace=tuple(trace),
                status="NXDOMAIN",
            )

        trace.append("error: too many referrals")
        return ResolutionResult(
            query_name=qname,
            query_type=qtype,
            final_records=(),
            trace=tuple(trace),
            status="SERVFAIL",
        )

    def _cache_lookup(self, name: str, rtype: str) -> Optional[Tuple[ResourceRecord, ...]]:
        entry = self.cache.get((name, rtype))
        if entry is None:
            return None
        if self.clock >= entry.expire_at:
            del self.cache[(name, rtype)]
            return None
        return entry.records

    def _cache_store(self, name: str, rtype: str, records: Tuple[ResourceRecord, ...]) -> None:
        if not records:
            return
        ttl = min(rr.ttl for rr in records)
        self.cache[(name, rtype)] = CacheEntry(records=records, expire_at=self.clock + ttl)


def normalize_name(name: str) -> str:
    lowered = name.strip().lower().rstrip(".")
    return "." if lowered == "" else lowered


def in_zone(qname: str, zone: str) -> bool:
    if zone == ".":
        return True
    return qname == zone or qname.endswith("." + zone)


def build_authorities() -> Dict[str, AuthorityServer]:
    root = AuthorityServer(
        zone=".",
        answers={},
        cname_map={},
        delegations=[Delegation("com", "a.gtld-servers.test", "192.0.2.53", 60)],
    )

    com = AuthorityServer(
        zone="com",
        answers={},
        cname_map={},
        delegations=[Delegation("example.com", "ns1.example.com", "203.0.113.53", 60)],
    )

    example_com_answers = {
        ("www.example.com", "A"): [
            ResourceRecord("www.example.com", "A", "203.0.113.80", 5),
        ],
        ("edge.example.com", "A"): [
            ResourceRecord("edge.example.com", "A", "203.0.113.42", 5),
        ],
        ("ns1.example.com", "A"): [
            ResourceRecord("ns1.example.com", "A", "203.0.113.53", 60),
        ],
    }
    example_com_cname = {
        "api.example.com": ResourceRecord(
            "api.example.com", "CNAME", "edge.example.com", 5
        )
    }
    example_com = AuthorityServer(
        zone="example.com",
        answers=example_com_answers,
        cname_map=example_com_cname,
        delegations=[],
    )

    return {".": root, "com": com, "example.com": example_com}


def format_records(records: Tuple[ResourceRecord, ...]) -> str:
    if not records:
        return "<empty>"
    return "; ".join(f"{rr.name} {rr.rtype} {rr.value} ttl={rr.ttl}" for rr in records)


def run_demo() -> List[ResolutionResult]:
    resolver = IterativeDNSResolver(build_authorities())
    queries = [
        ("www.example.com", "A"),
        ("api.example.com", "A"),
        ("www.example.com", "A"),
        ("missing.example.com", "A"),
    ]

    results: List[ResolutionResult] = []
    for name, rtype in queries:
        results.append(resolver.resolve(name, rtype))
    return results


def validate_results(results: List[ResolutionResult]) -> None:
    by_query = {(r.query_name, r.query_type): r for r in results}

    first_www = results[0]
    assert first_www.status == "NOERROR"
    assert first_www.final_records[0].value == "203.0.113.80"

    api = by_query[("api.example.com", "A")]
    assert api.status == "NOERROR"
    assert api.final_records[0].name == "edge.example.com"
    assert api.final_records[0].value == "203.0.113.42"

    second_www = results[2]
    assert any("cache-hit" in line for line in second_www.trace), "expected cache hit"

    missing = by_query[("missing.example.com", "A")]
    assert missing.status == "NXDOMAIN"


def main() -> None:
    results = run_demo()
    for result in results:
        print(f"\\nQUERY: {result.query_name} {result.query_type}")
        print(f"STATUS: {result.status}")
        print(f"ANSWER: {format_records(result.final_records)}")
        print("TRACE:")
        for line in result.trace:
            print(f"  - {line}")

    validate_results(results)
    print("\\nAll assertions passed.")


if __name__ == "__main__":
    main()
