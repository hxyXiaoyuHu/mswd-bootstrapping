library(ggplot2)

############# simulation ###########################

beta <- seq(0,1,0.2)
## provide the results 
data <- c(rate_mswd, rate_mmd, rate_edl1, rate_edl2, rate_bg, rate_pw)
data <- cbind(rep(beta,6), data)
colnames(data) <- c('beta', 'power')
data <- as.data.frame(data)
data$group <- rep(c('A', 'B','C', 'D', 'E', 'F'), each=6)

p <- ggplot(data,aes(x=beta)) + theme(panel.grid.major =element_blank(),
                                      panel.grid.minor = element_blank(),
                                      panel.background = element_rect(colour = 'black', fill='white'),
                                      axis.line = element_line(colour = "black"))+
  theme(plot.title = element_text(hjust=0.5))

p <- p + geom_line(aes(y=power,colour=group, linetype=group), lwd=1.5)
p <- p + scale_y_continuous(limits = c(0, 1))
p <- p + scale_colour_manual(name='group', values=c('red', 'black', 'blue', 'green', 'purple', 'orange'),
                             labels=c('Proposed', 'MMD', expression(ED[1]), expression(ED[2]), 'BG', 'PW'))
p <- p + scale_linetype_manual(name='group', values=c('solid', 'dashed', 'dotted', 'longdash', 'dotdash', 'twodash'),
                               labels=c('Proposed', 'MMD', expression(ED[1]), expression(ED[2]), 'BG', 'PW'))
p <- p + ggtitle('Model A, p=500') + theme(plot.title = element_text(size=20))
p <- p + labs(y = 'Power',x = expression(beta)) +
  # theme(legend.position='none')
  theme(axis.title = element_text(size=20)) + theme(axis.text = element_text(size=20))+ 
  theme(legend.title = element_blank(), legend.position = c(0, 1),
        legend.justification = c(0,1),
        legend.background = element_rect(color ='black', fill='white'),
        legend.key = element_rect(fill='white'),
        legend.key.height = unit(5, 'pt'),
        legend.key.width = unit(60, 'pt'), legend.text.align = 0,
        legend.text = element_text(size=20),
        legend.direction = 'vertical')
p

################# END ###############################


############# real data #######################
############ GBM grade low or high #############

data1 <- read.table('GBM/GBM_prognostic_low.txt')
data2 <- read.table('GBM/GBM_prognostic_high.txt')
n1 <- dim(data1)[1]; n2 <- dim(data2)[1]; pp <- dim(data1)[2]
mean1 <- colMeans(data1)
mean2 <- colMeans(data2)

# mean difference
data <- data.frame(ind=1:pp, mean1=mean1, mean2=mean2, meandiff = mean2-mean1)
summary(data$meandiff)
pvals = read.table('GBM/GBM_grade_marginal_pvals.txt')
pvals = pvals$V1
idx <- which(pvals<0.05)
df <- data.frame(x=idx, y=data$meandiff[idx])
# w <- 10
# h <- 9
# ggsave(file='GBM/GBM_grade_meandiff.pdf', w=w, h=h)
p <- ggplot(data, fill=group) + theme(panel.grid.major =element_blank(),
                          panel.grid.minor = element_blank(),
                          panel.background = element_rect(colour = 'black', fill='white'),
                          axis.line = element_line(colour = "black"))
p <- p + geom_point(aes(x=ind, y=meandiff), size=3) + ylim(c(-1, 1))
p <- p + geom_point(data=df, aes(x=x,y=y), colour='red')
p <- p + labs(x='Dimension', y='Mean difference') +
  theme(axis.title = element_text(size=20)) + theme(axis.text = element_text(size=20)) 
p
# graphics.off()

##################### end ######################


################ significant subsets of genes ###################
data1 <- as.matrix(data1)
data2 <- as.matrix(data2)
scale <- sqrt((n1*n2)/(n1+n2))

gene_sets = c('mitotic_cell_cycle', 'mitotic_cell_cycle_process', 'cell_cycle_process',
              'cell_cycle', 'DNA_metabolic_process', 'cell_division', 'mitotic_nuclear_division')
idx = read.table('GBM/idx_genes_mitotic_nuclear_division.txt')
idx = idx$V1
used_genes[idx]

############ Q-Q plot #######
v = read.table('GBM/direction_mitotic_nuclear_division.txt')
v = matrix(v$V1, ncol=1)
data_proj1 = data1[,idx] %*% matrix(v,ncol=1)
data_proj2 = data2[,idx] %*% matrix(v,ncol=1)
summary(data_proj1)
summary(data_proj2)
# w <- 10
# h <- 9
# pdf(file='GBM/qqplot_mitotic_nuclear_division.pdf',w=w,h=h)
df <- as.data.frame(qqplot(data_proj1[,1], data_proj2[,1], plot.it=F))
p <- ggplot(df, aes(x=x, y=y)) + theme(panel.grid.major =element_blank(),
                                         panel.grid.minor = element_blank(),
                                         panel.background = element_rect(colour = 'black', fill='white'),
                                         axis.line = element_line(colour = "black"))
p <- p + geom_point(size=6, shape=1)
p <- p + labs(x='Quantiles (LGG)', y='Quantiles (GBM)') + theme(axis.title = element_text(size=30)) +
  theme(axis.text = element_text(size=30))
# p <- p + scale_y_continuous(limits = c(0.2, 0.6)) + scale_x_continuous(limits = c(-0.7, 0.5))
p <- p + geom_abline(slope = 1, linetype='dotted', size=1)
p <- p + ggtitle('Mitotic nuclear division') + theme(plot.title = element_text(size=30)) +
  theme(plot.title = element_text(hjust=0.5))
p
# graphics.off()

