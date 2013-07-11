
#  - 47,236     highest term id
#  - 23,149     training vectors
#  - 781,265    test vectors
#  - 804,414    total (training + test)
#  - 810,935    highest document-id
#  - 381,327    points labeled with CCAT (RCV1-v2)
#  - 119,920    points labeled with ECAT (RCV1-v2)
#  - 239,267    points labeled with GCAT (RCV1-v2)
#  - 204,820    points labeled with MCAT (RCV1-v2)

X11.options(type="nbcairo")

# Define colors to be used for lines
plot_colors <- c("green","red")

to_file <- 1

if (to_file==1) {
  png(filename="rcv1.png", height=600, width=800, 
   bg="white")
  # postscript("fig1.eps")
} else {
  dev.new(width=8, height=6)
}

data <- read.table("rcv1.dat", header=T, sep=" ")

# Expand right side of clipping rect to make room for the legend
par(xpd=T, mar=par()$mar+c(0,2,0,4.5))

# Graph autos (transposing the matrix) using heat colors,  
# put 10% of the space between each bar, and make labels  
# smaller with horizontal y-axis labels
barplot(
  t(data),
  #main="RCV1 Class Distribution",
  ylab=" ", 
  col=plot_colors, space=0.1, cex.axis=1.5, las=1,
  names.arg=c("ccat", "ecat", "gcat", "mcat"), cex=1.5,
  axes=FALSE)

#maxValue = data$positive[1] + data$negative[1]
#print(maxValue)
y <- 10^5 * (1:8)
# format(y, scientific = FALSE)
# formatC(y, digits = 0, format = "f")
options(scipen = 50)
axis(2, las=1, at=y, cex.axis=1.5)

# Place the legend at (6,30) using heat colors
legend(4.4, 500000, names(data), cex=1.5, bty="n", fill=plot_colors);
  
# Restore default clipping rect
par(mar=c(5, 4, 4, 2) + 0.1)

# Turn off device driver (to flush output to png)
if (to_file==1) {
  dev.off()
}