X11.options(type="nbcairo")

# Define colors to be used for lines
plot_colors <- c("blue","red")

to_file <- 1

if (to_file==1) {
  png(filename="majority-vs-merge.png", height=600, width=800, 
   bg="white")
  # postscript("fig1.eps")
} else {
  dev.new(width=8, height=6)
}

# Read car and truck values from tab-delimited autos.dat
data_majority <- read.table("majority.dat", header=T, sep=" ")
data_merge <- read.table("merged.dat", header=T, sep=" ")

line_names = c("majority", "merged")
x <- data_majority$size
y1 <- data_majority$avg
y2 <- data_merge$avg

g_range <- range(y1, y2)
g_range[1] <- g_range[1] - 0.01
g_range[2] <- g_range[2] + 0.01

plot(x, y1, type="o", col=plot_colors[1], ylim=g_range, 
   axes=TRUE, ann=FALSE, cex.axis=1.5)

# Make y axis with horizontal labels that display ticks at 
# every 4 marks. 4*0:g_range[2] is equivalent to c(0,4,8,12).
#axis(2, las=1, at=0.01*0:max(y1,y2))

box()

lines(x, y2, type="o", pch=22, lty=2, col=plot_colors[2])

# Label the x and y axes with dark green text
title(xlab="Ensemble Size", cex.lab=1.5)
title(ylab="Accuracy", cex.lab=1.5)

# Create a title with a red, bold/italic font
#title(main="Majority vs Merge", col.main="red", font.main=4)

# Create a legend in the top-left corner that is slightly  
# smaller and has no border
legend("topleft", line_names, cex=1.5, col=plot_colors, 
   lty=1:3, lwd=2, bty="n");

# Turn off device driver (to flush output to png)
if (to_file==1) {
  dev.off()
}