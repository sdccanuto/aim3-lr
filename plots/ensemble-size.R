require(plotrix)
require(gplots)

X11.options(type="nbcairo")

createPlot <- function(filename) {
  # Define colors to be used for lines
  plot_colors <- c("blue","red")

  to_file <- 1

  if (to_file==1) {
    png(filename=paste("ensemble-size", filename, ".png"), height=600, width=600, 
      bg="white")
    # postscript("fig1.eps")
  }

  accuracy_global <- 0.9422005270206056

  # Read car and truck values from tab-delimited autos.dat
  data_majority <- read.table(filename, header=T, sep=" ")
  #data_merge <- read.table("merged.dat", header=T, sep=" ")

  x <- data_majority$size
  y1 <- data_majority$avg
  ci_lower <- data_majority$ci_low
  ci_upper <- data_majority$ci_hig
  g_range <- range(y1, y2)
  #y1 <- c(91,92,92,92.5,93,92,90)
  #y2 <- c(90,91,91.4,91.8,92,92,90)

  # calculate confidence intervals
  # See http://www.cyclismo.org/tutorial/R/confidence.html


  g_range <- range(y1)
  g_range[1] <- g_range[1] - 0.01
  g_range[2] <- g_range[2] + 0.01

  barplot2(
    y1, 
    main="Ensemble Size impact", 
    xlab="Ensemble size",  
    ylab="Accuracy", 
    ylim=g_range,
    xpd=FALSE,
    names.arg=data_majority$size, 
    border="blue", 
    axes=TRUE,
    plot.ci=TRUE,
    ci.l=ci_lower,
    ci.u=ci_upper)
    #density=c(10,10,10,10,10,10,20), 

  abline(accuracy_global, 0)
  #text(x[length(x)], accuracy_global, paste("global", accuracy_global))

  # Make y axis with horizontal labels that display ticks at 
  # every 4 marks. 4*0:g_range[2] is equivalent to c(0,4,8,12).
  # axis(2, las=1, at=1*0:max(y1,y2))

  # box()

  # lines(y2, type="o", pch=22, lty=2, col=plot_colors[2])

  # Label the x and y axes with dark green text
  #title(xlab="Ensemble Size", col.lab=rgb(0,0.5,0))
  #title(ylab="Accuracy", col.lab=rgb(0,0.5,0))

  # Create a title with a red, bold/italic font
  #title(main="Ensemble Size impact", col.main="red", font.main=4)

  # Create a legend in the top-left corner that is slightly  
  # smaller and has no border
  #legend("topleft", names(data[2:3]), cex=0.8, col=plot_colors, 
  #   lty=1:3, lwd=2, bty="n");

  # Turn off device driver (to flush output to png)
  if (to_file==1) {
    dev.off()
  }
}

createPlot("majority.dat")
createPlot("merged.dat")