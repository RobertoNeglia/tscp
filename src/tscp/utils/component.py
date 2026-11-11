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

    #: Whether loading a saved method fits this component again, on the fit split of the
    #: archive it was fitted on, because its fitted state lives outside torch (a forest,
    #: say) and is not in ``state_dict()``.
    refit_on_load: bool = False

    #: Whether prediction must construct this component's inputs after each release.
    #: Online state is prediction-scoped and must stay out of persistent buffers.
    adapts_online: bool = False

    def reset_online(self) -> None:
        """Clear prediction-scoped state before replaying an archive."""

    def fit(self, train: "ResidualArchive", validation: "ResidualArchive | None" = None) -> Self:
        """Fit trainable state.

        The composer calls this only for components that override it. ``train`` is the
        component-fitting split, ``archive.split("fit")``; ``validation`` is the
        ``validation`` split when the plan has one, for early stopping and model choice.
        Both are row subsets whose timeline ends where the split ends: the history before
        and within the split is there, and nothing after it.
        """
        return self

    @property
    def is_trainable(self) -> bool:
        """Whether :meth:`fit` does anything for this component."""
        return type(self).fit is not Component.fit

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, errors
    ):
        # Fitted buffers start as None, and their shape depends on the data, so a fresh
        # component cannot hold the saved values yet: make room for them, then load.
        for name, buffer in list(self._buffers.items()):
            key = prefix + name
            if name in self._non_persistent_buffers_set or key not in state_dict:
                continue
            incoming = state_dict[key]
            if buffer is None or buffer.shape != incoming.shape:
                device = buffer.device if buffer is not None else _device_of(self, incoming)
                self._buffers[name] = torch.empty_like(incoming, device=device)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, errors
        )

    def extra_repr(self) -> str:
        config = self.__dict__.get("_config_arguments", {})
        # Submodules print themselves; None means "use the default", shown as a submodule.
        return ", ".join(
            f"{name}={value!r}"
            for name, value in config.items()
            if value is not None and not isinstance(value, torch.nn.Module)
        )


def _device_of(module: torch.nn.Module, fallback: torch.Tensor) -> torch.device:
    """The device of a module's first tensor, or of ``fallback`` if it holds none."""
    for tensor in (*module.parameters(recurse=False), *module.buffers(recurse=False)):
        return tensor.device
    return fallback.device


__all__ = ["Component"]
