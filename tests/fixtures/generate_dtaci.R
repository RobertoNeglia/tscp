# Generate the DtACI oracle by running the authors' unmodified conformalAdaptStable.
# Usage: Rscript tests/fixtures/generate_dtaci.R tests/fixtures/dtaci_reference.json
# The authors' code is used only as an external reference; it is not a tscp dependency.
# Set TSCP_REFERENCE_SOURCE to a local copy of the pinned DtACI.R to skip the download;
# the recorded MD5 identifies the file that ran. Set TSCP_REFERENCE_RUNTIME to identify
# a nonnative R runtime in the metadata.

revision <- "09ab1674bc579b46d0e54b3525059c0591be342f"
url <- paste0("https://raw.githubusercontent.com/isgibbs/DtACI/", revision, "/DtACI.R")
source_file <- Sys.getenv("TSCP_REFERENCE_SOURCE", unset = "")
if (!nzchar(source_file)) {
    source_file <- tempfile(fileext = ".R")
    download.file(url, source_file, quiet = TRUE)
}
source_md5 <- unname(tools::md5sum(source_file))
source(source_file)

# The deterministic outputs below do not depend on the sampled expert, but the routine
# draws one at every step; a fixed seed keeps the run reproducible.
set.seed(1)
cases <- list(
    list(
        name = "three_experts",
        betas = c(0.7, 0.03, 0.5, 0.01, 0.4, 0.08, 0.9, 0.02, 0.6, 0.04, 0.2, 0.001),
        alpha = 0.1, gammas = c(0.01, 0.05, 0.2), sigma = 0.001, eta = 2.72
    ),
    list(
        # A deterministic low-discrepancy sequence: no random draws in the inputs.
        name = "paper_grid",
        betas = ((1:80) * 0.6180339887498949) %% 1,
        alpha = 0.2, gammas = c(0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128),
        sigma = 0.01, eta = 1.3
    )
)

vector_json <- function(x) paste0("[", paste(sprintf("%.17g", x), collapse = ","), "]")
string_json <- function(x) encodeString(x, quote = '"')
entry <- function(name, value) paste0('"', name, '":', value)
case_json <- function(case) {
    result <- conformalAdaptStable(
        case$betas, case$alpha, case$gammas, sigma = case$sigma, eta = case$eta
    )
    paste0("{", paste(c(
        entry("name", string_json(case$name)),
        entry("alpha", sprintf("%.17g", case$alpha)),
        entry("gammas", vector_json(case$gammas)),
        entry("sigma", sprintf("%.17g", case$sigma)),
        entry("eta", sprintf("%.17g", case$eta)),
        entry("betas", vector_json(case$betas)),
        # meanAlphaSeq and meanGammas: the deterministic Algorithm 2 outputs.
        entry("combined_levels", vector_json(result[[5]])),
        entry("mean_gammas", vector_json(result[[7]]))
    ), collapse = ","), "}")
}

fixture <- paste0("{\n", paste(c(
    entry("reference", '"DtACI.R conformalAdaptStable, deterministic outputs"'),
    entry("reference_revision", string_json(revision)),
    entry("reference_url", string_json(url)),
    entry("reference_md5", string_json(source_md5)),
    entry("paper_url", '"https://jmlr.org/papers/v25/22-1218.html"'),
    entry("generator", '"tests/fixtures/generate_dtaci.R"'),
    entry("versions", paste0('{"R":', string_json(as.character(getRversion())), "}")),
    entry("runtime", string_json(Sys.getenv("TSCP_REFERENCE_RUNTIME", unset = "native R"))),
    entry("platform", string_json(R.version$platform)),
    entry("tolerance", '{"atol":1e-12,"rtol":0}'),
    entry("conventions", string_json(paste(
        "beta_t is the largest covering level; expert misses are alpha_i > beta_t;",
        "combined level is the probability-weighted mean of expert levels"
    ))),
    entry("deviations", '["a continuous score distribution, threshold(alpha) = 1 - alpha, has no quantile ties"]'),
    entry("cases", paste0("[\n", paste(vapply(cases, case_json, character(1)), collapse = ",\n"), "\n]"))
), collapse = ",\n"), "\n}")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Pass the output JSON file as the only argument")
writeLines(fixture, args[1])
