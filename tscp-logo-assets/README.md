# tscp logo assets

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="tscp-horizontal-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="tscp-horizontal.svg">
  <img src="tscp-horizontal.svg" width="600" alt="tscp">
</picture>

**Conformal prediction for time series, in PyTorch.**

Write the project name as lowercase **tscp**. The symbol combines a time-series trace,
nested prediction bands and a caliper measuring their width.

## Choose an asset

| Use | Light background | Dark background |
|---|---|---|
| Symbol and wordmark | [tscp-horizontal.svg](tscp-horizontal.svg) | [tscp-horizontal-dark.svg](tscp-horizontal-dark.svg) |
| Symbol only | [tscp-symbol.svg](tscp-symbol.svg) | [tscp-symbol-dark.svg](tscp-symbol-dark.svg) |

Use the SVGs for READMEs, websites and documents. They have transparent backgrounds
and outlined lettering, so they require no installed fonts or external assets.
Preserve their proportions and leave clear space around the artwork.

The supplied PNGs are transparent previews of the light-background artwork, including
32- and 64-pixel symbol exports. Use the SVGs when a different size or background is needed.

## Colors

| Color | Hex | Use |
|---|---|---|
| Orange | `#FF7900` | Inner prediction band and accent |
| Light orange | `#FFAD45` | Outer prediction band |
| Charcoal | `#393B36` | Signal; caliper and wordmark on light backgrounds |
| Off-white | `#F0F2EE` | Caliper and wordmark on dark backgrounds |

The signal stays charcoal in both variants so it remains distinct against the orange.

## GitHub README example

For a README at the repository root:

```html
<picture>
  <source media="(prefers-color-scheme: dark)"
    srcset="tscp-logo-assets/tscp-horizontal-dark.svg">
  <source media="(prefers-color-scheme: light)"
    srcset="tscp-logo-assets/tscp-horizontal.svg">
  <img src="tscp-logo-assets/tscp-horizontal.svg" width="600" alt="tscp">
</picture>
```

The default image is the light-background variant. Keep the alternate source when
embedding the logo on a page that supports both light and dark themes.
