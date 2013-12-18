# ------------------------------------------------------------
# analyzeTagNetwork.R
# 
# John Cardente
#
# R script to select tags to build a reduced dataset.
#
# ------------------------------------------------------------

library(igraph)

# Arguments
# 1 - name of input edgelist file
# 2 - name of output tags file
# 3 - name of output graph file


# Read edge list and create initial graph
args     <- commandArgs(trailingOnly = TRUE)
edgeFile <- args[1] #"Data/edgelist200K.csv"
edgelist <- read.csv(edgeFile, header=FALSE)
colnames(edgelist) <- c("N1","N2","Count")

g     <- graph.data.frame(d=edgelist, directed=FALSE)
comps <- clusters(g, mode="strong")

# Get the largest component
gg <- induced.subgraph(g, vids=which(comps$membership == which.max(comps$csize)))
gg <- simplify(gg,
                remove.multiple=TRUE,
                remove.loops=TRUE,
                edge.attr.comb="sum")

# Get the top edges and create a subgraph with just those edges
# and then get the largest connected component
top.edges <- E(gg)[which(E(gg)$Count >= 150)]
ggg   <- subgraph.edges(gg, top.edges, delete.vertices=TRUE)
comps <- clusters(ggg)
ggg   <- induced.subgraph(ggg, vids=which(comps$membership == which.max(comps$csize)))

print(paste("Number of Nodes: ", length(V(ggg))))
print(paste("Number of Edges: ", length(E(ggg))))

#length(V(ggg))
#length(E(ggg))
#
#diameter(ggg)
#average.path.length(ggg)
#transitivity(ggg)

# Top nodes by degree
#top.degree <- head(order(degree(ggg), decreasing=TRUE),10)
#top.closeness <-  head(order(closeness(ggg), decreasing=TRUE),10)
#top.bet <- head(order(betweenness(ggg), decreasing=TRUE), 10)


# Save the tags we'd like to use
write(paste(V(ggg)$name, collapse =" "), file=args[2]) # "data/selectedtags.txt")

# Create the plot
V(ggg)$color <- "blue"
V(ggg)$frame.color <- "white"
V(ggg)$size <- 3
E(ggg)$color <- "#CCCCCC"

## V(ggg)$color[top.degree] <- "red"
## V(ggg)$color[top.bet[!top.bet %in% top.degree]] <- "blue"
## V(ggg)$color[top.closeness[! top.closeness %in% c(top.degree, top.bet)]] <- "green"
##
## V(ggg)$size[unique(c(top.degree, top.bet, top.closeness))] <- 3

pdf(file=args[3])

plot.igraph(ggg, layout=layout.fruchterman.reingold,
            edge.width=1,
            vertex.size=2,
            vertex.label=NA)

## points(-0.5,-1.2,pch=19,col="red")
## text(-0.45,-1.2,labels="Degree", cex=0.75)

## points(0,-1.2,pch=19,col="blue")
## text(0.05,-1.2,labels="Betweenness", cex=0.75)

## points(0.5,-1.2,pch=19,col="green")
## text(0.55,-1.2,labels="Closeness", cex=0.75)

dev.off()


