from dataclasses import dataclass, field


@dataclass
class Hit:
    id: str
    text: str
    payload: dict
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class Timings:
    ms: dict = field(default_factory=dict)