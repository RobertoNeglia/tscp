import numpy as np
import pytest
import torch

from tscp.quantile.width import WidthOptimal
from tscp.utils import from_config


@pytest.mark.parametrize("alpha", [0.1, 0.13, 0.07, 0.5, 0.9])
def test_journal_quantiles_match_independently_evaluated_percentiles(alpha):
    residuals = np.array([-4.0, -1.2, -0.3, 0.2, 0.5, 0.8, 1.0, 1.4, 2.0, 5.0])
    beta = np.linspace(0, alpha, 5)
    lower = np.percentile(residuals, np.ceil(100 * beta), method="linear")
    upper = np.percentile(residuals, np.ceil(100 * (1 - alpha + beta)), method="linear")
    i = np.argmin(upper - lower)
    actual = WidthOptimal(round_percentiles=True)(torch.from_numpy(residuals), alpha)
    torch.testing.assert_close(actual, torch.tensor([lower[i], upper[i]]), atol=1e-14, rtol=1e-14)


def test_width_selector_uses_first_tie_and_has_no_conformal_rank_correction():
    selector = WidthOptimal()
    candidates = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
    assert torch.equal(selector.select(candidates), torch.tensor([0.0, 2.0]))
    assert torch.equal(selector(torch.tensor([2.0]), 0.01), torch.tensor([2.0, 2.0]))
    assert torch.equal(selector(torch.empty(0), 0.1), torch.tensor([-torch.inf, torch.inf]))
    assert from_config(selector.to_config()).to_config() == selector.to_config()


def test_width_selector_checks_its_inputs():
    with pytest.raises(ValueError, match="bins"):
        WidthOptimal(bins=1)
    with pytest.raises(ValueError, match="alpha"):
        WidthOptimal().probabilities(0)
    with pytest.raises(ValueError, match="finite"):
        WidthOptimal()(torch.tensor([torch.nan]), 0.1)
