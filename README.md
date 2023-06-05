# MeshStuff

The command line application takes an OBJ file and converts to mesh clusters with several levels LODs. It built on C++ 17, uses CUDA, tinyobjload, tinyexr, METIS, and flip. 
It exports out triangle mesh vertex, indices data of the clusters, cluster info, cluster group info, and cluster tree. These are used for a propriatary cluster based renderer. The application also
exports out clusters in obj format, good for debugging. 
