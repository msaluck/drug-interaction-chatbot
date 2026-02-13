from src.chatbot import load_interaction_data, retrieve_interaction_info


def test_retrieve_known_pair():
    df = load_interaction_data("data/interactions_seed.csv")
    # Known pair present in the seed CSV
    q = "What is the interaction between captopril and zolpidem?"
    resp = retrieve_interaction_info(q, df)
    assert isinstance(resp, str)
    assert "Drug A:" in resp
    assert "Drug B:" in resp


def test_retrieve_unknown_pair():
    df = load_interaction_data("data/interactions_seed.csv")
    q = "What is the interaction between foobarinol and bazidone?"
    resp = retrieve_interaction_info(q, df)
    assert "No interaction information found" in resp
