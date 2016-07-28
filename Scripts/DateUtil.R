getCurrentTime <- function() {
  return(as.numeric(proc.time()[3]))
}

getElapsedTime <- function(startTime) {
  return(round(as.numeric(proc.time()[3] - startTime), 2))
}

printElapsedTime <- function(startTime, msg="Elapsed Time: ") {
  print(paste0(msg, getElapsedTime(startTime), "s"), quote="F")
}