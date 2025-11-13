Contiguous “waves” of writes across X: Each thread computes 8 X points separated by BLOCKSIZEX. For a fixed wave n, thread t writes to [yindex, xindex_base(t) + n*BLOCKSIZEX], while thread t+1 writes the immediate next X. Within a warp, this forms a contiguous, aligned segment, yielding coalesced global stores.

Row-major grid with X as fastest axis: Storing energygrid as [Y, X] ensures threads in the same warp touch adjacent X addresses on the same Y row, avoiding strided, scattered access and maximizing memory transaction width.

Cooperative, contiguous atom loads: Threads load atom records with consecutive indices into shared memory (atomtile). These sequential global reads are coalesced, and the tile is reused for all threads in the block, drastically reducing global memory traffic.

Reuse of Y/Z terms per atom: For each atom, dy and dyz2 are computed once and reused across 8 X positions, minimizing redundant arithmetic and keeping register usage friendly to occupancy, which indirectly supports sustained memory throughput.

    