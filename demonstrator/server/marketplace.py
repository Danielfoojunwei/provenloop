"""
TenSafe Adapter Marketplace — registry for browsing, loading, and
metering marketplace TGSP adapters.

Provides:
  - Adapter discovery from a local directory of .tgsp files
  - TGSP package verification (magic bytes, payload hash)
  - Usage metering for billing integration
  - Manifest parsing for adapter metadata
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

    def to_dict(self) -> dict:
        """Serialize the marketplace state for API responses."""
        return {
            "adapter_count": len(self._listings),
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
