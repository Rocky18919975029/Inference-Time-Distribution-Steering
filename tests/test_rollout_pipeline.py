from offline_subtb.limit_of_rlvr_io import build_qwen_boxed_prompt
from scripts.rollout_pipeline import normalize_prompt


def test_normalize_prompt_falls_back_to_qwen_template():
    question = "What is 2+2?"
    assert normalize_prompt(None, question) == build_qwen_boxed_prompt(question)


def test_normalize_prompt_reads_simplelr_prompt_list():
    prompt = [{"content": "hello", "role": "user"}]
    assert normalize_prompt(prompt, "ignored") == "hello"
