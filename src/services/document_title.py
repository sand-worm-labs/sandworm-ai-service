import random
from src.web.routes.document.models import DocumentTitleRequest, DocumentTitleResponse

_ADJECTIVES = [
    "Blazing", "Silent", "Golden", "Cosmic", "Frozen",
    "Hidden", "Ancient", "Crimson", "Velvet", "Neon",
    "Hollow", "Twisted", "Shattered", "Endless", "Vivid",
    "Midnight", "Electric", "Wandering", "Forgotten", "Radiant",
]

_NOUNS = [
    "Protocol", "Horizon", "Circuit", "Archive", "Frontier",
    "Ledger", "Signal", "Matrix", "Chronicle", "Nexus",
    "Theorem", "Codex", "Bastion", "Vertex", "Catalyst",
    "Blueprint", "Spectrum", "Cipher", "Epoch", "Paradox",
]

_VERBS = [
    "Analysis", "Overview", "Breakdown", "Deep Dive", "Summary",
    "Exploration", "Audit", "Review", "Report", "Playbook",
]


def random_document_title(seed: str | None = None) -> str:
    """Generate a random document title: ``<Adjective> <Noun> — <Verb>``.

    Example: ``"Blazing Nexus — Deep Dive"``

    Pass a *seed* (``workspace_id:document_id``) for a stable, deterministic
    title per document; omit for a fresh random title each call.
    """
    rng = random.Random(seed)
    adj = rng.choice(_ADJECTIVES)
    noun = rng.choice(_NOUNS)
    verb = rng.choice(_VERBS)
    return f"{adj} {noun} — {verb}"


async def generate_document_title(req: DocumentTitleRequest) -> DocumentTitleResponse:
    """Return a deterministic title seeded by ``workspace_id:document_id``."""
    seed = f"{req.context.workspace_id}:{req.context.document_id} ededed"
    return DocumentTitleResponse(
        title=random_document_title(seed),
        context=req.context,
    )
