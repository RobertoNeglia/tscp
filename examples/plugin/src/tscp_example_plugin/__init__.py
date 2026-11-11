"""An example tscp plugin: a weight scheme, and a method built on it.

Installing this package (``pip install -e examples/plugin``) makes both buildable by
name, because its ``pyproject.toml`` declares this module as a ``tscp.plugins`` entry
point::

    from tscp.methods import compose, get_method

    method = get_method("linear_decay", window=200)
    method = compose({"score": "signed", "weights": {"name": "linear_decay"}})
"""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.methods import ConformalPredictor
from tscp.registry import register
from tscp.scores import AbsoluteScore, Score
from tscp.sources import ExpandingSource, Source
from tscp.weights import Weights


@register("weights", "linear_decay")
class LinearDecayWeights(Weights):
    """Weights that fall linearly with age, to 0 at ``window`` base steps.

    Args:
        window: Age, in base time steps, at which a residual stops counting.
    """

    def __init__(self, window: int = 100):
        super().__init__()
        if window < 1:
            raise ValueError("window must be positive")
        self.window = int(window)

    def forward(self, pool: ResidualArchive, query: ResidualArchive) -> Tensor:
        age = (query.origins[:, None] - pool.origins[None, :]).clamp(min=0)
        return (1 - age.to(torch.float64) / self.window).clamp(min=0)


@register("method", "linear_decay")
class LinearDecayCP(ConformalPredictor):
    """Conformal prediction over an expanding pool, with linearly decaying weights.

    Args:
        window: Age, in base time steps, at which a residual stops counting.
        score: Defaults to the absolute residual.
        source: Defaults to the calibration split plus test residuals once known.
        alpha: Target miscoverage level.
    """

    def __init__(
        self,
        window: int = 100,
        score: Score | None = None,
        source: Source | None = None,
        alpha: float = 0.1,
    ):
        super().__init__(
            score=score if score is not None else AbsoluteScore(),
            source=source if source is not None else ExpandingSource(),
            weights=LinearDecayWeights(window),
            alpha=alpha,
        )
