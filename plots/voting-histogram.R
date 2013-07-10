X11.options(type="nbcairo")

# Define colors to be used for lines
plot_colors <- c("blue","red")

to_file <- 1

if (to_file==1) {
  png(filename="voting-histogram.png", height=600, width=800, 
   bg="white")
  # postscript("fig1.eps")
} else {
  dev.new(width=8, height=6)
}

data <- read.table("voting-histogram.dat", header=T, sep=" ")
#data <- read.table("voting-histogram.dat", header=T, sep=" ")

# Expand right side of clipping rect to make room for the legend
par(xpd=T, mar=par()$mar+c(0,1,0,6))

# Graph autos (transposing the matrix) using heat colors,  
# put 10% of the space between each bar, and make labels  
# smaller with horizontal y-axis labels
barplot(t(data), main="Voting Histogram", ylab="Outliers", 
   col=heat.colors(8), space=0.1, cex.axis=1.5, cex.names=1.5, las=1,
   names.arg=c(2,3,4,6,8,16), cex=1.5, cex.lab=1.7)
   
# Place the legend at (6,30) using heat colors
legend(6.9, 1600, names(data), cex=1.5, fill=heat.colors(8));
  
# Restore default clipping rect
par(mar=c(5, 4, 4, 2) + 0.1)

# Turn off device driver (to flush output to png)
if (to_file==1) {
  dev.off()
}