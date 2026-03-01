"""TG Tinker -- Agentic LoRA Creator.

Creates production-quality LoRA adapters from any dataset.
Default output is TGSP (safe, signed, sellable).
Optional export to safetensors, PyTorch, or GGUF.

Usage::

    from tensafe.skills.tg_tinker import TGTinker

    tinker = TGTinker()
    result = tinker.create_adapter("./my_data", domain="finance")
"""

from tensafe.skills.tg_tinker.tinker import (
    TGTinker,
    AdapterManifest,
    ContinuousLoopConfig,
    CuratedDataset,
    EvaluationResult,
    ImprovementPlan,
    ScreeningResult,
    TrainingResult,
    curate_dataset,
    package_tgsp,
)
from tensafe.skills.tg_tinker.data_analyzer import (
    DataAnalysis,
    FileInfo,
    LoraConfig,
    analyze_data,
    classify_domain,
    suggest_full_config,
)
from tensafe.skills.tg_tinker.export import (
    export_tgsp_to_gguf,
    export_tgsp_to_pytorch,
    export_tgsp_to_safetensors,
    import_lora_to_tgsp,
    read_tgsp,
    verify_tgsp,
    write_tgsp,
)

__all__ = [
    # Core class
    "TGTinker",
    # Data analysis
    "DataAnalysis",
    "FileInfo",
    "LoraConfig",
    "analyze_data",
    "classify_domain",
    "suggest_full_config",
    # Pipeline types
    "AdapterManifest",
    "ContinuousLoopConfig",
    "CuratedDataset",
    "EvaluationResult",
    "ImprovementPlan",
    "ScreeningResult",
    "TrainingResult",
    "curate_dataset",
    "package_tgsp",
    # Export/import
    "export_tgsp_to_gguf",
    "export_tgsp_to_pytorch",
    "export_tgsp_to_safetensors",
    "import_lora_to_tgsp",
    "read_tgsp",
    "verify_tgsp",
    "write_tgsp",
]
