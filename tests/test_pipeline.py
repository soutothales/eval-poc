"""End-to-end pipeline with the offline model and a stubbed LLM judge (no API calls)."""

import asyncio

import yaml

from eval import constants as c
from eval import score as score_mod
from eval.adapters.agent_adapter import AgentAdapter
from eval.compare import compare_runs
from eval.gate import evaluate_gate, load_thresholds
from eval.generate import generate
from eval.io import read_json, read_jsonl
from eval.report import write_report

OFFLINE = c.CONFIGS_DIR / "offline.yaml"
OFFLINE_LOW = c.CONFIGS_DIR / "offline_low_topk.yaml"


class StubMetric:
    """Mimics a DeepEval metric: 0.9 with several passages, 0.4 otherwise."""

    def __init__(self, name, judge, threshold):
        self.name, self.judge, self.threshold = name, judge, threshold
        self.score = self.success = self.reason = None

    async def a_measure(self, test_case, _show_indicator=False):
        self.judge.input_tokens += 1000
        self.judge.output_tokens += 100
        ctx = test_case.retrieval_context or []
        self.score = 0.9 if len(ctx) > 1 else 0.4
        self.success = self.score >= self.threshold
        self.reason = f"stub {self.name}: {len(ctx)} passages"


def _judge_cfg(tmp_path):
    cfg = yaml.safe_load(c.JUDGE_CONFIG_PATH.read_text())
    cfg[c.JCFG_PROVIDER] = "fake"
    cfg[c.JCFG_MODEL_ID] = "claude-opus-5-5"  # priced, so judge cost is computed
    path = tmp_path / "judge.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_generate_resume_and_error_rows(tmp_runs, monkeypatch):
    run_id = asyncio.run(generate(OFFLINE, repeats=2, limit=6))
    rows = read_jsonl(tmp_runs / run_id / c.OUTPUTS_FILE)
    core_in_first6 = {"QA-0001", "QA-0004"}
    assert len(rows) == 6 + len(core_in_first6)
    assert all(r[c.OUT_USAGE] for r in rows)

    again = asyncio.run(generate(OFFLINE, repeats=2, limit=6, resume_run_id=run_id))
    assert again == run_id and len(read_jsonl(tmp_runs / run_id / c.OUTPUTS_FILE)) == len(rows)

    original = AgentAdapter.ainvoke

    async def flaky(self, question):
        if "Orion" in question:
            raise RuntimeError("boom")
        return await original(self, question)

    monkeypatch.setattr(AgentAdapter, "ainvoke", flaky)
    run2 = asyncio.run(generate(OFFLINE, limit=8))
    errs = [r for r in read_jsonl(tmp_runs / run2 / c.OUTPUTS_FILE) if r[c.OUT_ERROR]]
    assert [r[c.OUT_CASE_ID] for r in errs] == ["QA-0005"]  # recorded, not dropped


def test_score_compare_gate_report(tmp_runs, tmp_path, monkeypatch):
    monkeypatch.setattr(
        score_mod, "build_metric",
        lambda name, judge, cfg: StubMetric(name, judge, cfg[c.JCFG_THRESHOLDS][name]),
    )
    judge_path = _judge_cfg(tmp_path)
    run_a = asyncio.run(generate(OFFLINE))
    run_b = asyncio.run(generate(OFFLINE_LOW))
    for rid in (run_a, run_b):
        summary = asyncio.run(score_mod.score_run(rid, judge_config_path=judge_path, progress=lambda m: None))
        assert summary[c.SUM_N_CASES] == 30
        assert c.M_CORRECTNESS in summary[c.SUM_QUALITY]
        assert summary[c.SUM_OPERATIONAL][c.SUM_JUDGE_COST_TOTAL] > 0

    manifest = read_json(tmp_runs / run_a / c.RUN_MANIFEST_FILE)
    assert manifest[c.RM_JUDGE][c.RM_CRITERIA_HASH].startswith("sha256:")
    assert manifest[c.RM_PRICE_TABLE_VERSION]

    # Not-found/decline cases only get correctness; empty context -> contextual metrics = 0.
    scores = read_jsonl(tmp_runs / run_a / c.SCORES_FILE)
    nf = {s[c.SC_METRIC] for s in scores if s[c.SC_CASE_ID] == "QA-0022"}
    assert c.M_FAITHFULNESS not in nf and c.M_CORRECTNESS in nf

    cmp = compare_runs(run_a, run_b)
    assert cmp[c.CMP_DATASET_MATCH] and cmp[c.CMP_SHARED_CASES] == 30
    res = evaluate_gate(cmp, load_thresholds())
    assert not res[c.GATE_PASSED]
    assert any(c.M_HIT_RATE in b[c.GATE_RULE] for b in res[c.GATE_BREACHES])

    same = evaluate_gate(compare_runs(run_a, run_a), load_thresholds())
    assert same[c.GATE_PASSED]

    path = write_report(run_a, run_b)
    html = path.read_text()
    assert "Gate failed" in html and "QA-" in html
