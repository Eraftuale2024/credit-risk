from src.demo import EXAMPLES, to_record

def test_examples_land_in_intended_bands(scorer):
    out = {k: scorer.predict(to_record(v)) for k, v in EXAMPLES.items()}
    assert not any(o["out_of_domain"] for o in out.values())
    assert out["Low risk"]["decision"] == "approve"
    assert out["Borderline"]["decision"] == "review"
    assert out["High risk"]["decision"] == "decline"