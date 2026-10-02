"""Regression checks for the Model Registry publisher/verifier workflow.

These checks intentionally inspect the workflow file's structure as text only
-- no YAML parsing dependency, no network, no Docker -- so they stay safe and
fast in ordinary unit-test jobs. They exist to prevent recurrence of the
v0.1.8 verifier defect: a grep-based metadata check that queried the
human-oriented summary of an attestation-bearing multi-manifest OCI index,
where the platform image's own labels never appear.

Developer:
    Manish Kumar <manish@omnibioai.org>
"""

from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "publish-model-registry.yml"
)


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_metadata_is_resolved_from_the_platform_manifest_by_digest() -> None:
    """The verifier must select the runtime platform manifest explicitly by
    digest/platform rather than grepping the multi-manifest index summary
    (the v0.1.8 defect: labels live on the platform image's own config, never
    on the index-level text rendering)."""
    text = _workflow()
    assert 'select(.platform.os == "linux" and .platform.architecture == $arch)' in text
    assert "--format '{{json .Image.Config.Labels}}'" in text
    # The old defective pattern must not return.
    assert 'grep -q \'org.opencontainers.image.source\' "/tmp/model-registry-$arch.txt"' not in text


def test_attestation_descriptors_are_not_mistaken_for_runtime_platforms() -> None:
    """Attestation-manifest descriptors (platform unknown/unknown) must be
    selected separately from, and never confused with, the real linux/amd64
    or linux/arm64 runtime manifest descriptor."""
    text = _workflow()
    assert 'select(.annotations["vnd.docker.reference.type"] == "attestation-manifest")' in text
    # The runtime-manifest selector constrains platform.os to "linux", which
    # excludes the attestation manifest's "unknown/unknown" platform.
    assert 'select(.platform.os == "linux"' in text


def test_oci_revision_requires_exact_40_character_sha_not_substring() -> None:
    """Revision verification must be exact-length and exact-equality, never a
    substring/grep match that could accidentally hit unrelated text."""
    text = _workflow()
    assert 'test "${#actual_revision}" -eq 40' in text
    assert 'test "$actual_revision" = "$SOURCE_SHA"' in text


def test_both_architectures_are_checked_independently() -> None:
    text = _workflow()
    assert "for arch in amd64 arm64; do" in text
    assert "smoke_amd64:" in text
    assert "smoke_arm64:" in text


def test_supply_chain_attestations_are_subject_bound_not_merely_present() -> None:
    """An attestation descriptor existing is not sufficient; its `subject`
    must be verified to point back at the exact runtime manifest digest it
    claims to cover."""
    text = _workflow()
    assert 'subject_digest=$(printf \'%s\' "$attest_json" | jq -r \'.subject.digest' in text
    assert 'test "$subject_digest" = "$runtime_digest"' in text
    assert "https://spdx.dev/Document" in text
    assert "https://slsa.dev/provenance/v1" in text


def test_github_native_attestation_api_is_not_relied_upon() -> None:
    """Policy decision: OCI-native, subject-bound in-toto attestations
    (already present and valid on the published artifacts) satisfy the
    supply-chain requirement. `gh attestation verify` checks a separate store
    (GitHub's native Attestations API) that this pipeline never populates, so
    it must not be used as a pass/fail gate."""
    assert "gh attestation verify" not in _workflow()


def test_runtime_smoke_pulls_immutable_digest_not_a_mutable_tag() -> None:
    text = _workflow()
    assert "needs.build_amd64.outputs.digest" in text
    assert "needs.build_arm64.outputs.digest" in text
    assert 'docker pull --platform linux/amd64 "$IMAGE@$DIGEST"' in text
    assert 'docker pull --platform linux/arm64 "$IMAGE@$DIGEST"' in text


def test_runtime_smoke_runs_after_release_verification() -> None:
    text = _workflow()
    assert "needs: [build_amd64, build_arm64, assemble_and_verify]" in text


def test_runtime_smoke_asserts_native_runner_with_no_qemu_fallback() -> None:
    text = _workflow()
    assert 'test "$(uname -m)" = x86_64' in text
    assert 'test "$(uname -m)" = aarch64' in text
    assert "runs-on: ubuntu-24.04-arm" in text
    for token in ("setup-qemu-action", "tonistiigi/binfmt"):
        assert token not in text


def test_runtime_smoke_never_references_latest_tag() -> None:
    """Smoke jobs must validate the exact immutable digest under test, never
    the floating `latest` tag (publish_latest defaults to false and must stay
    uninvolved in smoke assumptions)."""
    text = _workflow()
    smoke_start = text.index("smoke_amd64:")
    smoke_text = text[smoke_start:]
    assert ":latest" not in smoke_text


def test_runtime_smoke_jobs_are_fail_closed_and_bounded() -> None:
    text = _workflow()
    assert "timeout-minutes: 5" in text
    assert "curl -sf --max-time 3" in text
    assert "trap cleanup EXIT" in text
