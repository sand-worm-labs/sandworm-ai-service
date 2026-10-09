import pytest


@pytest.fixture(autouse=True)
def no_real_embedding_model(mocker):
    """Tests must never load or download the local embedding model."""
    import src.util.embeddings as embeddings

    mocker.patch.object(embeddings, "_load_local_model", side_effect=AssertionError("loaded the real model"))
