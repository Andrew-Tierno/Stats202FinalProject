getCurrentTime <- function() {
  return(as.numeric(proc.time()[3]))
}

printElapsedTime <- function(startTime, msg="Elapsed Time: ") {
  timeDiff <- 
  print(paste0(msg, round(as.numeric(proc.time()[3] - startTime), 2), "s"), quote="F")
}