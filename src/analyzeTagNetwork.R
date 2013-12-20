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
edgeFile <- args[1] 
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


# Save the tags we'd like to use
write(paste(V(ggg)$name, collapse =" "), file=args[2])

# Create the plot
V(ggg)$color       <- "blue"
V(ggg)$frame.color <- "white"
V(ggg)$size        <- 3
E(ggg)$color       <- "#CCCCCC"

pdf(file=args[3])

plot.igraph(ggg, layout=layout.fruchterman.reingold,
            edge.width=1,
            vertex.size=2,
            vertex.label=NA)
dev.off()


