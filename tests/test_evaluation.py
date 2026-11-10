from usv_sim.dataset import list_cases
from usv_sim.ppo_agent import PPOPolicy
from evaluate_ppo import evaluate_case


def test_evaluation_exposes_safety_and_learning_contribution_metrics():
    row=evaluate_case(list_cases()[0],PPOPolicy(),1,200,100)
    for key in ('安全域侵入时长_s','安全干预次数','无解决策次数','单动作决策比例',
                '规划延迟_p95_s','规则违反步数','策略模式'):
        assert key in row
    assert row['策略模式']=='deterministic'
