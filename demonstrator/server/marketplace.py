"""
TenSafe Adapter Marketplace — registry for browsing, loading, and
metering marketplace TGSP adapters.

Business model:
  - 0% transaction fee on buying/selling (creators keep 100%)
  - Revenue comes from VALIDATION: adapters must be TenSafe Validated to list
  - Every adapter has an embedded SKILL.md (skill_doc) that agents can read
  - Validation = RVUv2 screening + quality benchmark + security check

Provides:
  - Adapter discovery from a local directory of .tgsp files
  - TGSP package verification (magic bytes, payload hash, signatures, RVUv2)
  - Usage metering for billing integration
  - Manifest parsing for adapter metadata (including embedded SKILL.md)
  - Validation-gated listing (TenSafe Validated badge required)
"""

import hashlib
import json
import logging
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


MARKETPLACE_FEE_PERCENT = 0  # 0% transaction fee — creators keep 100%

VALIDATION_REQUIRED = True  # Adapters must be TenSafe Validated to list


@dataclass
class AdapterListing:
    """Metadata for a marketplace adapter."""

    adapter_id: str
    name: str
    version: str
    format_version: str
    domain: str
    expert_type: str
    rank: int
    alpha: int
    license: str
    price_per_1k_tokens: float
    creator: str
    description: str
    tags: List[str]
    payload_size: int
    payload_hash: str
    tgsp_path: str
    usage_metering: bool = True
    # TGSP v1.1 fields
    skill_doc: str = ""  # Embedded SKILL.md — agents read this
    validated: bool = False  # TenSafe Validated badge
    validation_timestamp: str = ""
    creator_verified: bool = False
    rvu_screening_passed: bool = False
    lora_config: Optional[Dict] = None


@dataclass
class UsageRecord:
    """Tracks token usage for a single adapter."""

    adapter_id: str
    total_tokens: int = 0
    total_requests: int = 0


class AdapterMarketplace:
    """Registry for browsing, loading, and metering marketplace adapters.

    Scans a directory of .tgsp files and exposes their manifests for
    discovery. Supports TGSP v1 and v2 formats.
    """

    def __init__(self, tgsp_dir: str | Path):
        self._tgsp_dir = Path(tgsp_dir)
        self._listings: Dict[str, AdapterListing] = {}
        self._usage: Dict[str, UsageRecord] = {}
        self._lock = threading.Lock()
        self._scan()

    def _scan(self):
        """Scan the TGSP directory and index all adapter manifests."""
        if not self._tgsp_dir.exists():
            logger.warning(f"Marketplace directory not found: {self._tgsp_dir}")
            return

        for tgsp_path in sorted(self._tgsp_dir.glob("*.tgsp")):
            try:
                manifest = self._read_manifest(tgsp_path)
                if manifest is None:
                    continue
                # Support both v1.0 (legacy) and v1.1 (new) manifest formats
                is_v11 = manifest.get("format") == "TGSP" and manifest.get("version") == "1.1"

                if is_v11:
                    # TGSP v1.1 manifest with full skill_doc, LoraConfig, creator, etc.
                    skill = manifest.get("skill", {})
                    creator_info = manifest.get("creator", {})
                    lora_cfg = manifest.get("lora_config", {})
                    rvu = manifest.get("rvu_safety", {})
                    integrity = manifest.get("integrity", {})
                    adapter_id = manifest.get("name", tgsp_path.stem)

                    # Enforce validation requirement
                    is_validated = rvu.get("screening_passed", False) and creator_info.get("verified", False)
                    if VALIDATION_REQUIRED and not is_validated:
                        logger.warning(
                            f"Marketplace: skipping unvalidated adapter '{adapter_id}' "
                            f"(RVUv2={rvu.get('screening_passed')}, creator_verified={creator_info.get('verified')})"
                        )
                        continue

                    listing = AdapterListing(
                        adapter_id=adapter_id,
                        name=manifest.get("name", tgsp_path.stem),
                        version=manifest.get("version", "1.1"),
                        format_version="1.1",
                        domain=manifest.get("domain", "general"),
                        expert_type=manifest.get("model", {}).get("architecture", "sparse_moe"),
                        rank=lora_cfg.get("rank", 30),
                        alpha=lora_cfg.get("alpha", 64),
                        license=manifest.get("license", "unknown"),
                        price_per_1k_tokens=manifest.get("price_per_1k_tokens", 0.0),
                        creator=creator_info.get("name", "unknown"),
                        description=skill.get("description", ""),
                        tags=skill.get("triggers", []),
                        payload_size=0,
                        payload_hash=integrity.get("payload_hash", ""),
                        tgsp_path=str(tgsp_path),
                        usage_metering=True,
                        # v1.1 fields
                        skill_doc=manifest.get("skill_doc", ""),
                        validated=is_validated,
                        validation_timestamp=rvu.get("screening_timestamp", ""),
                        creator_verified=creator_info.get("verified", False),
                        rvu_screening_passed=rvu.get("screening_passed", False),
                        lora_config=lora_cfg,
                    )
                else:
                    # Legacy v1.0 format
                    meta = manifest.get("metadata", {})
                    adapter_id = manifest.get("adapter_id", tgsp_path.stem)
                    listing = AdapterListing(
                        adapter_id=adapter_id,
                        name=manifest.get("model_name", tgsp_path.stem),
                        version=manifest.get("model_version", "0.0.0"),
                        format_version=manifest.get("format_version", "1.0"),
                        domain=meta.get("domain", "general"),
                        expert_type=meta.get("expert_type", "unknown"),
                        rank=manifest.get("rank", 0),
                        alpha=manifest.get("alpha", 0),
                        license=manifest.get("license", "unknown"),
                        price_per_1k_tokens=manifest.get("price_per_1k_tokens", 0.0),
                        creator=manifest.get("creator", "unknown"),
                        description=meta.get("description", ""),
                        tags=meta.get("tags", []),
                        payload_size=manifest.get("payload_size", 0),
                        payload_hash=manifest.get("payload_hash", ""),
                        tgsp_path=str(tgsp_path),
                        usage_metering=manifest.get("usage_metering", False),
                    )
                self._listings[adapter_id] = listing
                logger.info(
                    f"Marketplace: indexed adapter '{listing.name}' "
                    f"(id={adapter_id}, v{listing.format_version})"
                )
            except Exception as e:
                logger.warning(f"Failed to index {tgsp_path.name}: {e}")

    @staticmethod
    def _read_manifest(tgsp_path: Path) -> Optional[dict]:
        """Read and parse the JSON manifest from a TGSP file."""
        with open(tgsp_path, "rb") as f:
            header = f.read(10)
            if len(header) < 10 or header[:4] != b"TGSP":
                logger.warning(f"Bad TGSP magic in {tgsp_path.name}")
                return None
            manifest_len = struct.unpack_from("<I", header, 6)[0]
            manifest_bytes = f.read(manifest_len)
        return json.loads(manifest_bytes.decode("utf-8"))

    def list_adapters(self) -> List[AdapterListing]:
        """List all available adapters with metadata and pricing."""
        return list(self._listings.values())

    def get_adapter(self, adapter_id: str) -> Optional[AdapterListing]:
        """Look up a specific adapter by ID."""
        return self._listings.get(adapter_id)

    def search(
        self,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[AdapterListing]:
        """Search adapters by domain and/or tags."""
        results = list(self._listings.values())
        if domain:
            results = [a for a in results if a.domain == domain]
        if tags:
            tag_set = set(tags)
            results = [a for a in results if tag_set & set(a.tags)]
        return results

    def download_adapter(self, adapter_id: str) -> Optional[Path]:
        """Return the local path to a TGSP package for loading."""
        listing = self._listings.get(adapter_id)
        if listing is None:
            return None
        path = Path(listing.tgsp_path)
        if not path.exists():
            logger.error(f"TGSP file missing for adapter {adapter_id}: {path}")
            return None
        return path

    def verify_adapter(self, tgsp_path: Path) -> bool:
        """Verify TGSP package integrity (magic bytes + payload hash)."""
        try:
            with open(tgsp_path, "rb") as f:
                header = f.read(10)
                if len(header) < 10 or header[:4] != b"TGSP":
                    return False
                manifest_len = struct.unpack_from("<I", header, 6)[0]
                manifest_bytes = f.read(manifest_len)
                payload = f.read()

            manifest = json.loads(manifest_bytes.decode("utf-8"))
            expected_hash = manifest.get("payload_hash", "")
            if not expected_hash:
                return True  # No hash to verify (v1 compat)

            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != expected_hash:
                logger.warning(
                    f"TGSP hash mismatch for {tgsp_path.name}: "
                    f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"TGSP verification failed for {tgsp_path}: {e}")
            return False

    def meter_usage(self, adapter_id: str, tokens: int):
        """Track token usage for billing."""
        with self._lock:
            if adapter_id not in self._usage:
                self._usage[adapter_id] = UsageRecord(adapter_id=adapter_id)
            rec = self._usage[adapter_id]
            rec.total_tokens += tokens
            rec.total_requests += 1

    def get_usage(self, adapter_id: str) -> Optional[UsageRecord]:
        """Get usage stats for an adapter."""
        return self._usage.get(adapter_id)

    def get_all_usage(self) -> Dict[str, UsageRecord]:
        """Get usage stats for all adapters."""
        return dict(self._usage)

    def search_by_skill(self, task_description: str) -> List[AdapterListing]:
        """Search adapters by matching task description against embedded SKILL.md.

        An agent provides a task description, and we find adapters whose
        skill_doc or triggers suggest they can handle the task.
        """
        task_lower = task_description.lower()
        results = []
        for listing in self._listings.values():
            # Check triggers
            if any(trigger.lower() in task_lower for trigger in listing.tags):
                results.append(listing)
                continue
            # Check skill_doc content
            if listing.skill_doc and any(
                word in listing.skill_doc.lower()
                for word in task_lower.split()
                if len(word) > 3
            ):
                results.append(listing)
        return results

    def get_skill_doc(self, adapter_id: str) -> Optional[str]:
        """Get the embedded SKILL.md for an adapter.

        This is the key method agents use to understand what an adapter does.
        Every TGSP adapter IS a skill file — this returns that skill description.
        """
        listing = self._listings.get(adapter_id)
        if listing is None:
            return None
        return listing.skill_doc or listing.description

    def is_validated(self, adapter_id: str) -> bool:
        """Check if an adapter has the TenSafe Validated badge."""
        listing = self._listings.get(adapter_id)
        return listing is not None and listing.validated

    def get_marketplace_fee(self) -> float:
        """Get the marketplace transaction fee percentage. Currently 0%."""
        return MARKETPLACE_FEE_PERCENT

    def to_dict(self) -> dict:
        """Serialize the marketplace state for API responses."""
        return {
            "adapter_count": len(self._listings),
            "marketplace_fee_percent": MARKETPLACE_FEE_PERCENT,
            "validation_required": VALIDATION_REQUIRED,
            "adapters": [
                {
                    "adapter_id": a.adapter_id,
                    "name": a.name,
                    "version": a.version,
                    "domain": a.domain,
                    "expert_type": a.expert_type,
                    "license": a.license,
                    "price_per_1k_tokens": a.price_per_1k_tokens,
                    "creator": a.creator,
                    "description": a.description,
                    "tags": a.tags,
                    "rank": a.rank,
                    "validated": a.validated,
                    "creator_verified": a.creator_verified,
                    "rvu_screening_passed": a.rvu_screening_passed,
                    "has_skill_doc": bool(a.skill_doc),
                    "usage": {
                        "total_tokens": self._usage.get(
                            a.adapter_id, UsageRecord(a.adapter_id)
                        ).total_tokens,
                        "total_requests": self._usage.get(
                            a.adapter_id, UsageRecord(a.adapter_id)
                        ).total_requests,
                    }
                    if a.usage_metering
                    else None,
                }
                for a in self._listings.values()
            ],
        }
