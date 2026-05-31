from offline_subtb.limit_of_rlvr_io import build_qwen_boxed_prompt, flatten_limit_of_rlvr_rows


def test_flatten_limit_of_rlvr_rows_reconstructs_prompt():
    rows = [
        {
            "idx": 7,
            "question": "What is 1+1?",
            "gt": "2",
            "answer": "2",
            "pred": ["2", "bad"],
            "code": ["reason\n\\boxed{2}", "wrong reasoning\n\\boxed{3}"],
            "finish_reason": ["stop", "length"],
            "score": [True, False],
        }
    ]

    flattened = flatten_limit_of_rlvr_rows(
        rows,
        source_path="math/examples/math_eval/EVAL/checkpoints/run/eval_results/global_step_0/math500/test_qwen-boxed_-1_seed1_t0.6_s0_e-1.jsonl",
        model_name_or_path="Qwen/Qwen2.5-7B",
        top_p=0.95,
        max_tokens=4096,
    )

    assert len(flattened) == 2
    assert flattened[0]["problem_id"] == 7
    assert flattened[0]["sample_id"] == 0
    assert flattened[0]["reward"] == 1.0
    assert flattened[1]["reward"] == 0.0
    assert flattened[0]["response"] == "reason\n\\boxed{2}"
    assert flattened[0]["extracted_answer"] == "2"
    assert flattened[1]["response"] == "wrong reasoning\n\\boxed{3}"
    assert flattened[1]["extracted_answer"] == "bad"
    assert flattened[0]["prompt"] == build_qwen_boxed_prompt("What is 1+1?")
    assert flattened[0]["benchmark"] == "math500"
    assert flattened[0]["template"] == "qwen-boxed"
    assert flattened[0]["temperature"] == 0.6
    assert flattened[0]["sampling_seed"] == 1
