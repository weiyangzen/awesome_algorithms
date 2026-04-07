"""Minimal runnable MVP for X.509-style digital certificates.

This is a teaching implementation that mirrors the high-level X.509 workflow:
- build certificates as signed TBSCertificate payloads,
- construct a leaf -> intermediate -> root chain,
- verify validity window, issuer relation, basic constraints, signature, and trust anchor.

Security note:
- RSA signing here uses textbook `pow(hash, d, n)` without PKCS#1 padding,
  so this code is NOT production cryptography.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from math import gcd
import hashlib
import json
import secrets
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RSAKeyPair:
    n: int
    e: int
    d: int


@dataclass(frozen=True)
class Certificate:
    serial_number: int
    subject: str
    issuer: str
    not_before: datetime
    not_after: datetime
    is_ca: bool
    path_len_constraint: Optional[int]
    public_key_n: int
    public_key_e: int
    signature_algorithm: str
    signature: int


@dataclass(frozen=True)
class ChainValidationResult:
    passed: bool
    errors: list[str]


def to_utc_z(dt: datetime) -> str:
    dt_utc = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt_utc.isoformat().replace("+00:00", "Z")


def tbs_payload(cert: Certificate) -> dict[str, object]:
    return {
        "serial_number": cert.serial_number,
        "subject": cert.subject,
        "issuer": cert.issuer,
        "not_before": to_utc_z(cert.not_before),
        "not_after": to_utc_z(cert.not_after),
        "is_ca": cert.is_ca,
        "path_len_constraint": cert.path_len_constraint,
        "public_key_n": cert.public_key_n,
        "public_key_e": cert.public_key_e,
        "signature_algorithm": cert.signature_algorithm,
    }


def full_payload(cert: Certificate) -> dict[str, object]:
    payload = tbs_payload(cert)
    payload["signature"] = cert.signature
    return payload


def canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_int(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest(), "big")


def rsa_sign_tbs(tbs_bytes: bytes, issuer_key: RSAKeyPair) -> int:
    hashed = sha256_int(tbs_bytes) % issuer_key.n
    return pow(hashed, issuer_key.d, issuer_key.n)


def rsa_verify_tbs_signature(tbs_bytes: bytes, signature: int, issuer_n: int, issuer_e: int) -> bool:
    if not (0 <= signature < issuer_n):
        return False
    expected = sha256_int(tbs_bytes) % issuer_n
    got = pow(signature, issuer_e, issuer_n)
    return got == expected


def is_probable_prime(n: int, rounds: int = 16) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_prime(bits: int) -> int:
    if bits < 32:
        raise ValueError("bits must be >= 32")
    while True:
        candidate = secrets.randbits(bits)
        candidate |= 1
        candidate |= 1 << (bits - 1)
        if is_probable_prime(candidate):
            return candidate


def generate_rsa_keypair(bits: int = 128, e: int = 65537) -> RSAKeyPair:
    if bits < 32:
        raise ValueError("bits must be >= 32")

    while True:
        p = generate_prime(bits)
        q = generate_prime(bits)
        if p == q:
            continue
        phi = (p - 1) * (q - 1)
        if gcd(e, phi) != 1:
            continue
        n = p * q
        d = pow(e, -1, phi)
        return RSAKeyPair(n=n, e=e, d=d)


def issue_certificate(
    *,
    serial_number: int,
    subject: str,
    issuer: str,
    subject_key: RSAKeyPair,
    issuer_signing_key: RSAKeyPair,
    not_before: datetime,
    not_after: datetime,
    is_ca: bool,
    path_len_constraint: Optional[int],
) -> Certificate:
    unsigned_cert = Certificate(
        serial_number=serial_number,
        subject=subject,
        issuer=issuer,
        not_before=not_before,
        not_after=not_after,
        is_ca=is_ca,
        path_len_constraint=path_len_constraint,
        public_key_n=subject_key.n,
        public_key_e=subject_key.e,
        signature_algorithm="toy-rsa-sha256",
        signature=0,
    )
    signature = rsa_sign_tbs(canonical_json_bytes(tbs_payload(unsigned_cert)), issuer_signing_key)
    return replace(unsigned_cert, signature=signature)


def certificate_fingerprint(cert: Certificate) -> str:
    return hashlib.sha256(canonical_json_bytes(full_payload(cert))).hexdigest()


def verify_chain(
    chain_leaf_to_root: list[Certificate],
    trust_anchor_fingerprints: set[str],
    now: datetime,
) -> ChainValidationResult:
    errors: list[str] = []

    if not chain_leaf_to_root:
        return ChainValidationResult(False, ["empty certificate chain"])

    for idx, cert in enumerate(chain_leaf_to_root):
        is_root = idx == len(chain_leaf_to_root) - 1
        issuer_cert = cert if is_root else chain_leaf_to_root[idx + 1]

        if cert.issuer != issuer_cert.subject:
            errors.append(f"{cert.subject}: issuer mismatch")

        if not (cert.not_before <= now <= cert.not_after):
            errors.append(f"{cert.subject}: certificate not valid at evaluation time")

        tbs_bytes = canonical_json_bytes(tbs_payload(cert))
        sig_ok = rsa_verify_tbs_signature(
            tbs_bytes=tbs_bytes,
            signature=cert.signature,
            issuer_n=issuer_cert.public_key_n,
            issuer_e=issuer_cert.public_key_e,
        )
        if not sig_ok:
            errors.append(f"{cert.subject}: signature verification failed")

        if not is_root and not issuer_cert.is_ca:
            errors.append(f"{issuer_cert.subject}: issuer is not a CA")

        if cert.is_ca and cert.path_len_constraint is not None:
            ca_below = sum(1 for lower_cert in chain_leaf_to_root[:idx] if lower_cert.is_ca)
            if ca_below > cert.path_len_constraint:
                errors.append(
                    f"{cert.subject}: pathLen constraint violated "
                    f"(ca_below={ca_below}, limit={cert.path_len_constraint})"
                )

    root_fingerprint = certificate_fingerprint(chain_leaf_to_root[-1])
    if root_fingerprint not in trust_anchor_fingerprints:
        errors.append("root certificate is not in trust anchor set")

    return ChainValidationResult(passed=len(errors) == 0, errors=errors)


def build_demo_pki(now: datetime) -> tuple[list[Certificate], set[str]]:
    root_key = generate_rsa_keypair(bits=128)
    inter_key = generate_rsa_keypair(bits=128)
    leaf_key = generate_rsa_keypair(bits=128)

    root_subject = "CN=Demo Root CA"
    inter_subject = "CN=Demo Intermediate CA"
    leaf_subject = "CN=api.demo.local"

    root_cert = issue_certificate(
        serial_number=1001,
        subject=root_subject,
        issuer=root_subject,
        subject_key=root_key,
        issuer_signing_key=root_key,
        not_before=now - timedelta(days=1),
        not_after=now + timedelta(days=3650),
        is_ca=True,
        path_len_constraint=1,
    )

    inter_cert = issue_certificate(
        serial_number=1002,
        subject=inter_subject,
        issuer=root_subject,
        subject_key=inter_key,
        issuer_signing_key=root_key,
        not_before=now - timedelta(days=1),
        not_after=now + timedelta(days=1825),
        is_ca=True,
        path_len_constraint=0,
    )

    leaf_cert = issue_certificate(
        serial_number=1003,
        subject=leaf_subject,
        issuer=inter_subject,
        subject_key=leaf_key,
        issuer_signing_key=inter_key,
        not_before=now - timedelta(days=1),
        not_after=now + timedelta(days=365),
        is_ca=False,
        path_len_constraint=None,
    )

    chain = [leaf_cert, inter_cert, root_cert]
    trust_anchors = {certificate_fingerprint(root_cert)}
    return chain, trust_anchors


def chain_summary(chain: list[Certificate]) -> pd.DataFrame:
    rows = []
    for cert in chain:
        rows.append(
            {
                "subject": cert.subject,
                "issuer": cert.issuer,
                "is_ca": cert.is_ca,
                "path_len": cert.path_len_constraint,
                "not_before": to_utc_z(cert.not_before),
                "not_after": to_utc_z(cert.not_after),
                "sig_alg": cert.signature_algorithm,
                "sig_bits": cert.signature.bit_length(),
            }
        )
    return pd.DataFrame(rows)


def run_validation_cases(chain: list[Certificate], trust_anchors: set[str], now: datetime) -> pd.DataFrame:
    leaf, inter, root = chain

    tampered_leaf = replace(leaf, subject="CN=mallory.demo.local")
    expired_eval_time = now + timedelta(days=500)
    untrusted_anchors: set[str] = set()

    cases = [
        ("valid_chain", chain, trust_anchors, now),
        ("tampered_leaf_subject", [tampered_leaf, inter, root], trust_anchors, now),
        ("expired_leaf", chain, trust_anchors, expired_eval_time),
        ("untrusted_root", chain, untrusted_anchors, now),
    ]

    rows = []
    for name, case_chain, anchors, eval_time in cases:
        result = verify_chain(case_chain, anchors, eval_time)
        rows.append(
            {
                "case": name,
                "passed": result.passed,
                "error_count": len(result.errors),
                "errors": " | ".join(result.errors) if result.errors else "OK",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    now = datetime.now(timezone.utc)
    chain, trust_anchors = build_demo_pki(now)

    cert_df = chain_summary(chain)
    result_df = run_validation_cases(chain, trust_anchors, now)

    pass_ratio = float(np.mean(result_df["passed"].to_numpy(dtype=np.int32)))

    valid_row = result_df.loc[result_df["case"] == "valid_chain", "passed"].iloc[0]
    tampered_row = result_df.loc[result_df["case"] == "tampered_leaf_subject", "passed"].iloc[0]
    expired_row = result_df.loc[result_df["case"] == "expired_leaf", "passed"].iloc[0]
    untrusted_row = result_df.loc[result_df["case"] == "untrusted_root", "passed"].iloc[0]

    assert bool(valid_row) is True
    assert bool(tampered_row) is False
    assert bool(expired_row) is False
    assert bool(untrusted_row) is False

    print("=== X.509 Certificate MVP (Teaching Version) ===")
    print("\n[Certificate Chain]")
    print(cert_df.to_string(index=False))

    print("\n[Validation Cases]")
    print(result_df.to_string(index=False))

    print("\n[Metrics]")
    print(f"pass_ratio={pass_ratio:.2f} (expected 0.25 for 1/4 passing cases)")
    print(f"trust_anchor_count={len(trust_anchors)}")
    print("Status: PASS")


if __name__ == "__main__":
    main()
