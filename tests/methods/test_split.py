import torch

from tscp.datasets import gaussian_control, variance_shift
from tscp.methods import ConformalPredictor, NexCP, SplitCP
from tscp.scores import AbsoluteScore
from tscp.sources import ExpandingSource, HoldoutSource


def test_split_cp_coverage_on_exchangeable_data():
    """Mean coverage lies in [1 - alpha, 1 - alpha + 1/(n + 1)] for exchangeable residuals."""
    num_calibration, num_test, repeats, alpha = 99, 200, 300, 0.1
    coverages = []
    for seed in range(repeats):
        archive = gaussian_control(
            horizon=1,
            num_origins=num_calibration + num_test,
            sizes={"calibration": num_calibration, "test": None},
            seed=seed,
        )
        region = SplitCP(alpha=alpha).calibrate(archive).predict(archive)
        coverages.append(region.contains(archive.split("test").residuals).double().mean())
    mean = torch.stack(coverages).mean().item()
    # The Monte Carlo standard error of the mean is about 0.002; allow about 4.5 of them.
    tolerance = 0.01
    assert 1 - alpha - tolerance <= mean <= 1 - alpha + 1 / (num_calibration + 1) + tolerance


def test_split_threshold_is_the_conformal_order_statistic(make_archive):
    calibration = torch.randperm(99).float() + 1  # scores 1..99
    archive = make_archive(torch.cat([calibration, torch.zeros(5)]), {"calibration": 99, "test": 5})
    region = SplitCP(alpha=0.1).calibrate(archive).predict(archive)
    assert torch.equal(region.upper.flatten(), torch.full((5,), 90.0))  # ceil(0.9 * 100) = 90
    assert torch.equal(region.lower, -region.upper)


def test_nexcp_with_decay_one_is_split_cp():
    archive = variance_shift(horizon=2, num_origins=500)
    split = SplitCP().calibrate(archive).predict(archive)
    nexcp = NexCP(decay=1.0, source=HoldoutSource()).calibrate(archive).predict(archive)
    assert torch.equal(split.upper, nexcp.upper)

    expanding = ConformalPredictor(AbsoluteScore(), source=ExpandingSource())
    uniform = expanding.calibrate(archive).predict(archive)
    nexcp = NexCP(decay=1.0).calibrate(archive).predict(archive)
    assert torch.equal(uniform.upper, nexcp.upper)


def test_adaptive_methods_recover_after_a_variance_shift():
    archive = variance_shift(horizon=1, num_origins=3000)
    test = archive.split("test")
    after = test.origins >= int(0.75 * archive.num_origins) + 200  # well after the shift
    coverage = {}
    for name, method in (("split", SplitCP()), ("nexcp", NexCP(decay=0.99))):
        region = method.calibrate(archive).predict(archive)
        coverage[name] = region.contains(test.residuals)[after].double().mean().item()
    assert coverage["split"] < 0.6
    assert coverage["nexcp"] > 0.8
