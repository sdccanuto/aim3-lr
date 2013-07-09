X11.options(type="nbcairo")

# Define colors to be used for lines
plot_colors <- c("blue","red")

to_file <- 1

if (to_file==1) {
  png(filename="unanimity.png", height=600, width=600, 
   bg="white")
  # postscript("fig1.eps")
}

# Read car and truck values from tab-delimited autos.dat
data_majority <- read.table("einstimmigkeitsrate.dat", header=T, sep=" ")

line_name = "unanimity"
x <- data_majority$size
y <- data_majority$einstimmigkeit

g_range <- range(y)
g_range[1] <- g_range[1] - 2
g_range[2] <- g_range[2] + 2

plot(x, y, type="o", col=plot_colors[1], ylim=g_range, 
   axes=TRUE, ann=FALSE)

# Make y axis with horizontal labels that display ticks at 
# every 4 marks. 4*0:g_range[2] is equivalent to c(0,4,8,12).
#axis(2, las=1, at=0.01*0:max(y1,y2))

box()

# Label the x and y axes with dark green text
title(xlab="Ensemble Size", col.lab=rgb(0,0.5,0))
title(ylab="Unanimity", col.lab=rgb(0,0.5,0))

# Create a title with a red, bold/italic font
title(main="Unanimity of ensemble votes", col.main="red", font.main=4)

# Create a legend in the top-left corner that is slightly  
# smaller and has no border
#legend("topleft", line_name, cex=0.8, col=plot_colors, 
#   lty=1:3, lwd=2, bty="n");

# Turn off device driver (to flush output to png)
if (to_file==1) {
  dev.off()
}