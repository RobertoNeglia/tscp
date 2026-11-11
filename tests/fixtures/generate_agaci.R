# Generate the AgACI endpoint-aggregation oracle with unmodified OPERA 1.2.0 R code.
# Usage: Rscript tests/fixtures/generate_agaci.R tests/fixtures/agaci_reference.json
# OPERA is used only as an external reference; it is not a tscp dependency.
# Set TSCP_REFERENCE_RUNTIME to identify a nonnative R runtime in the metadata.

revision <- "7057c8c76235a39db2234599dce27e67c873c3a7"
base_url <- paste0("https://raw.githubusercontent.com/cran/opera/", revision, "/R/")
for (file in c("loss.R", "BOA.R")) {
    local_file <- tempfile(fileext = ".R")
    download.file(paste0(base_url, file), local_file, quiet = TRUE)
    source(local_file)
    unlink(local_file)
}

calibration <- c(-2, -1, -0.4, 0, 0.6, 1, 1.8, 2.5)
residuals <- c(0, 2.7, -3.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 8, -6, 0.5)
forecasts <- c(10, 12, 9, 11, 10, 8, 9, 13, 12, 10, 10, 10, 5, 6, 7)
gammas <- c(0, 0.05, 0.6)
alpha <- 0.2
endpoint_bounds <- c(-10, 20)
level <- rep(alpha, length(gammas))
levels <- experts_lower <- experts_upper <- matrix(0, length(residuals), length(gammas))
pool <- abs(calibration)
for (t in seq_along(residuals)) {
    levels[t, ] <- level
    radius <- vapply(level, function(a) {
        if (a <= 0) return(Inf)
        if (a >= 1) return(0)
        rank <- ceiling((1 - a - 1e-9) * (length(pool) + 1))
        if (rank > length(pool)) return(Inf)
        sort(pool)[rank]
    }, numeric(1))
    low <- forecasts[t] - radius
    high <- forecasts[t] + radius
    low[!is.finite(low)] <- endpoint_bounds[1]
    high[!is.finite(high)] <- endpoint_bounds[2]
    experts_lower[t, ] <- low
    experts_upper[t, ] <- high
    level <- level + gammas * (alpha - (abs(residuals[t]) > radius))
    pool <- c(pool, abs(residuals[t]))
}

reference <- function(experts, tau) {
    BOA(y = forecasts + residuals, experts = experts,
        awake = matrix(1, nrow(experts), ncol(experts)),
        loss.type = list(name = "pinball", tau = tau), loss.gradient = TRUE,
        w0 = rep(1 / ncol(experts), ncol(experts)), use_cpp = FALSE, quiet = TRUE)
}
lower <- reference(experts_lower, alpha / 2)
upper <- reference(experts_upper, 1 - alpha / 2)

vector_json <- function(x) paste0("[", paste(sprintf("%.17g", x), collapse = ","), "]")
matrix_json <- function(x) paste0("[", paste(apply(x, 1, vector_json), collapse = ","), "]")
string_json <- function(x) encodeString(x, quote = '"')
strings_json <- function(x) paste0("[", paste(vapply(x, string_json, character(1)), collapse = ","), "]")
entry <- function(name, value) paste0('"', name, '":', value)
fixture <- paste0("{\n", paste(c(
    entry("reference", '"OPERA 1.2.0, BOA with pinball gradient, uniform prior"'),
    entry("reference_revision", paste0('"', revision, '"')),
    entry("reference_url", paste0('"', base_url, 'BOA.R"')),
    entry("generator", '"tests/fixtures/generate_agaci.R"'),
    entry("versions", paste0('{"R":', string_json(as.character(getRversion())), ',"OPERA":"1.2.0"}')),
    entry("runtime", string_json(Sys.getenv("TSCP_REFERENCE_RUNTIME", unset = "native R"))),
    entry("platform", string_json(R.version$platform)),
    entry("tolerance", '{"atol":1e-12,"rtol":1e-12}'),
    entry("conventions", '"Float64; uniform prior; OPERA adaptive rates and pinball gradient; separate endpoint aggregators"'),
    entry("deviations", strings_json(c(
        "fixed forecasts replace forecaster refitting",
        "experts use an expanding absolute-residual pool and the conformal order statistic",
        "finite endpoint replacements are explicit synthetic-fixture settings"
    ))),
    entry("paper_url", '"https://proceedings.mlr.press/v162/zaffran22a.html"'),
    entry("expert_convention", '"Fixed forecasts, expanding absolute residuals, conformal order statistic; alpha >= 1 gives radius zero"'),
    entry("calibration", vector_json(calibration)),
    entry("residuals", vector_json(residuals)),
    entry("forecasts", vector_json(forecasts)),
    entry("gammas", vector_json(gammas)),
    entry("alpha", sprintf("%.17g", alpha)),
    entry("endpoint_bounds", vector_json(endpoint_bounds)),
    entry("levels", matrix_json(levels)),
    entry("experts_lower", matrix_json(experts_lower)),
    entry("experts_upper", matrix_json(experts_upper)),
    entry("weights_lower", matrix_json(lower$weights)),
    entry("weights_upper", matrix_json(upper$weights)),
    entry("lower", vector_json(lower$prediction)),
    entry("upper", vector_json(upper$prediction))
), collapse = ",\n"), "\n}")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Pass the output JSON file as the only argument")
writeLines(fixture, args[1])
