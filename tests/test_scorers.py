from pathlib import Path

from agent_bench.config import DatabaseSpec
from agent_bench.dataset import Sample
from agent_bench.db import SqliteDatabase
from agent_bench.scorers import ScorerContext
from agent_bench.scorers.deterministic import AnswerValues, ResultMatch, Tables, _numbers
from agent_bench.scorers.judge import CriterionVerdict, JudgeOutput, RubricJudge
from agent_bench.trace import ToolCall, Trace

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"
DB = SqliteDatabase(DatabaseSpec(path="db/schema.sql"), DEMO)
CTX = ScorerContext(db=DB, base_dir=DEMO)


def trace(*sql: str, answer: str = "", **kw) -> Trace:
    calls = [ToolCall(name="run_sql", input={"sql": q}, output="") for q in sql]
    return Trace(run_key="k", variant="v", sample_id="s", epoch=0, variant_fingerprint={},
                 tool_calls=calls, final_answer=answer, **kw)


SAMPLE = Sample(id="s", question="top carrier?", reference_sql=(
    "SELECT carrier, COUNT(*) FROM shipments WHERE delivered_at IS NOT NULL GROUP BY carrier ORDER BY 2 DESC LIMIT 1"
), expected_values=["DHL", "4"], expected_tables=["shipments"])


def test_result_match_ignores_column_order_and_extra_columns():
    scorer = ResultMatch(CTX)
    same = "SELECT COUNT(*) AS n, carrier, 'x' AS extra FROM shipments WHERE delivered_at IS NOT NULL " \
           "GROUP BY carrier ORDER BY n DESC LIMIT 1"
    assert scorer.score(SAMPLE, trace(same))[0].value == 1.0
    wrong = "SELECT carrier, COUNT(*) FROM shipments GROUP BY carrier ORDER BY 2 ASC LIMIT 1"
    assert scorer.score(SAMPLE, trace(wrong))[0].value == 0.0
    # Only the last successful query counts.
    assert scorer.score(SAMPLE, trace(wrong, same))[0].value == 1.0
    assert ResultMatch(CTX, allow_extra_columns=False).score(SAMPLE, trace(same))[0].value == 0.0
    assert scorer.score(SAMPLE, trace())[0].value == 0.0


def test_answer_values_and_number_formats():
    assert _numbers("3 684,50 € et 1,234.5 puis 716") == [3684.5, 1234.5, 716.0]
    scorer = AnswerValues(CTX)
    assert scorer.score(SAMPLE, trace(answer="C'est DHL, avec 4 livraisons."))[0].value == 1.0
    assert scorer.score(SAMPLE, trace(answer="Colissimo, 4."))[0].value == 0.5


def test_tables_recall_precision():
    t = trace("SELECT * FROM shipments s JOIN orders o ON o.order_id = s.order_id")
    recall, precision = Tables(CTX).score(SAMPLE, t)
    assert recall.value == 1.0 and precision.value == 0.5


class FakeJudge:
    def __init__(self, passed: dict[str, bool]):
        self.passed = passed
        self.prompts = []

    def judge(self, system, prompt):
        self.prompts.append(prompt)
        verdicts = [CriterionVerdict(criterion_id=c, reasoning="r", passed=p) for c, p in self.passed.items()]
        return JudgeOutput(verdicts=verdicts), {"judge_model_served": "fake"}


def test_rubric_judge_scopes_criteria_and_votes():
    fake = FakeJudge({"correct": True, "grounded": True, "business_rules": False, "concise": True, "abstains": True})
    ctx = ScorerContext(db=DB, base_dir=DEMO, clients={"judge": fake})
    judge = RubricJudge(ctx, rubric="rubrics/answer_v1.yaml", votes=3)

    scores = {s.name: s.value for s in judge.score(SAMPLE, trace("SELECT 1", answer="DHL"))}
    assert "judge.abstains" not in scores  # only applies to unanswerable questions
    assert scores["judge.business_rules"] == 0.0 and scores["judge"] == 0.75
    assert len(fake.prompts) == 3 and "<agent_sql_queries>" in fake.prompts[0]

    unanswerable = Sample(id="u", question="marge ?", answerable=False)
    scores = {s.name for s in judge.score(unanswerable, trace(answer="impossible"))}
    assert scores == {"judge.abstains", "judge.concise", "judge"}


def test_judge_fingerprint_tracks_rubric_content(tmp_path):
    rubric = tmp_path / "r.yaml"
    rubric.write_text("name: r\ncriteria: [{id: a, question: q}]\n")
    ctx = ScorerContext(db=DB, base_dir=tmp_path, clients={"judge": FakeJudge({"a": True})})
    before = RubricJudge(ctx, rubric="r.yaml").fingerprint()
    rubric.write_text("name: r\ncriteria: [{id: a, question: q2}]\n")
    assert RubricJudge(ctx, rubric="r.yaml").fingerprint() != before
