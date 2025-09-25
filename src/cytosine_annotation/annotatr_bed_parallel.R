suppressPackageStartupMessages({
  library(annotatr)
  library(BiocParallel)
  library(GenomicRanges)
  library(glue)
  library(optparse)
})

# option_list <- list(
#   make_option(
#     opt_str	= c("-i", "--input"),
#     dest    = "input",
#     type    = "character",
#     help    = "the path of the bed file to be annotated"),
#   make_option(
#     opt_str	= c("-o", "--output"),
#     dest    = "output",
#     type    = "character",
#     help    = "the path of the output bed file")
# )

# args <- parse_args(OptionParser(option_list=option_list))

option_list <- list(
  make_option(
    opt_str	= c("-i", "--input"),
    dest    = "input_list",
    type    = "character",
    help    = "the list of path of the bed files to be annotated"),
  make_option(
    opt_str	= c("-p", "--cores"),
    dest    = "cores",
    type    = "integer",
    default = 4,
    help    = "Number of cores to use"),
  make_option(
    opt_str	= c("-s", "--strand"),
    dest    = "strand",
    action  = "store_true",
    default = FALSE,
    help    = "Whether to consider strand information")
)


args <- parse_args(OptionParser(option_list=option_list))
input_beds <- unlist(strsplit(args$input_list, ","))
valid_beds <- input_beds[file.exists(input_beds)]
invalid_beds <- setdiff(input_beds, valid_beds)

if (length(valid_beds) == 0) {
  message("no vaild bed files found, exit")
  quit()
}

if (length(invalid_beds) > 0) {
  message(glue("WARNING: the following files not found:\n{paste(invalid_beds, collapse = '\n')}"))
}

if (args$cores > 0) {
  bioc_param <- MulticoreParam(workers = args$cores)
} else {
  bioc_param <- MulticoreParam(workers = 4)
}
register(bioc_param)


annotate_regions_from_file <- function(input,
                                       built_annotations) {
  
  if (args$strand) {
    tmp_gr <- read_regions(
      con       = input,
      genome    = "hg38",
      extraCols = c(strand_col = "character")
    )
    strand(tmp_gr) <- tmp_gr$strand_col
    tmp_gr$strand_col <- NULL

    annotated_gr <- annotate_regions(
      regions       = tmp_gr,
      annotations   = built_annotations,
      ignore.strand = FALSE)

    cols_to_write <- c("seqnames", "start", "end", "strand",
                       "annot.symbol", "annot.type")

  } else {
    annotated_gr <- annotate_regions(
      regions       = read_regions(
        con    = input,
        genome = "hg38"),
      annotations   = built_annotations,
      ignore.strand = TRUE)

    cols_to_write <- c("seqnames", "start", "end",
                       "annot.symbol", "annot.type")
  }

  write.table(
    x         = as.data.frame(annotated_gr)[, cols_to_write],
    file      = glue("{input}.annotatr"),
    sep       = "\t",
    quote     = FALSE,
    col.names = FALSE,
    row.names = FALSE)
  system(glue("chmod 777 {input}.annotatr"))
  message(glue("Done {basename(input)} -> {basename(input)}.annotatr"))
}

message(glue("Start annotating {length(valid_beds)} files, using {args$cores} cores..."))
hg38_annotations <- readRDS("/opt/hg38_annotations.rds")
bplapply(
  valid_beds,
  annotate_regions_from_file,
  BPPARAM = bioc_param,
  built_annotations = hg38_annotations
)
