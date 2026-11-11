import json
import math

import pytest
import torch

from tscp.data import ResidualArchive
from tscp.metrics import coverage, evaluate
from tscp.regions import Box, Interval

# Four origins, two steps, one series: both steps inside, step 1 out, step 2 out, both out.
RESIDUALS = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]).unsqueeze(-1)


def archive_of(residuals, observed=None):
    num_origins, horizon, _ = residuals.shape
    return ResidualArchive(
        residuals=residuals,
        origins=torch.arange(num_origins),
        steps=torch.arange(1, horizon + 1),
        observed=observed,
    )


def unit_intervals():
    return Interval(torch.full((4, 2, 1), -1.0), torch.full((4, 2, 1), 1.0))


def unit_boxes():
    return Box(torch.full((4, 1, 2), -1.0), torch.full((4, 1, 2), 1.0), event_dims=("step",))


def test_per_step_regions_report_each_step_and_the_whole_horizon():
    report = evaluate(unit_intervals(), archive_of(RESIDUALS), alpha=0.1)
    assert report.num_regions == 8 and report.coverage == 0.5
    assert report.coverage_by_step == (0.5, 0.5)
    assert report.joint_coverage == 0.25  # only the first origin is covered at both steps
    assert report.log_width_by_step == pytest.approx((math.log(2), math.log(2)))


def test_joint_regions_cover_only_when_every_target_is_inside():
    report = evaluate(unit_boxes(), archive_of(RESIDUALS), alpha=0.1)
    assert report.num_regions == 4
    assert report.coverage == 0.25 and report.joint_coverage == 0.25
    assert report.coverage_by_step is None and report.winkler is None
    assert report.log_volume_per_dim == pytest.approx(math.log(2))  # 2 log 2 over 2 dims


def test_regions_with_a_missing_target_are_not_evaluated():
    observed = torch.ones(4, 2, 1, dtype=torch.bool)
    observed[0, 1, 0] = False  # the second step of the first origin never arrived
    residuals = RESIDUALS.clone()
    residuals[0, 1, 0] = 0.0
    archive = archive_of(residuals, observed)
    joint = evaluate(unit_boxes(), archive, alpha=0.1)
    assert joint.num_regions == 3 and joint.coverage == 0.0  # origins 2, 3 and 4 miss
    per_step = evaluate(unit_intervals(), archive, alpha=0.1)
    assert per_step.num_regions == 7 and per_step.coverage == pytest.approx(3 / 7)
    assert per_step.joint_coverage == 0.0  # the first origin's horizon is not all observed
    assert coverage(unit_boxes(), residuals, observed).item() == 0.0


def test_reports_serialize_to_json():
    report = evaluate(unit_intervals(), archive_of(RESIDUALS), 0.1, window=2)
    text = json.dumps(report.to_dict())
    assert json.loads(text)["coverage_by_step"] == [0.5, 0.5]
