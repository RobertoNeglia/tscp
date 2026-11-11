"""The base class shared by every axis component."""

from typing import TYPE_CHECKING, Self

import torch

from tscp.utils.config import ConfigMixin

if TYPE_CHECKING:
    from tscp.data import ResidualArchive


class Component(ConfigMixin, torch.nn.Module):
    """A torch module with a recorded config and an optional fit step.

    Fitted tensors are registered as parameters or buffers, so they live in
    ``state_dict()`` and move with ``.to(device)``. Settings live in the config
    (:meth:`~tscp.utils.config.ConfigMixin.to_config`).
    """

    def fit(self, archive: "ResidualArchive") -> Self:
        """Fit trainable state.

        The composer calls this only with the component-fitting split
        (``archive.split("fit")``), and only for components that override it.
        """
        return self

    @property
    def is_trainable(self) -> bool:
        """Whether :meth:`fit` does anything for this component."""
        return type(self).fit is not Component.fit

    def extra_repr(self) -> str:
        config = self.__dict__.get("_config_arguments", {})
        # Submodules print themselves; None means "use the default", shown as a submodule.
        return ", ".join(
            f"{name}={value!r}"
            for name, value in config.items()
            if value is not None and not isinstance(value, torch.nn.Module)
        )


__all__ = ["Component"]
