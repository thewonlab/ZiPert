check_file <- function(file) {
  if (!file.exists(file)) {
    stop(paste("File", file, "does not exist."))
  }
}

read_data <- function(file) {
  check_file(file)
  data <- read.csv(file)
  return(data)
}