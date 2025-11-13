import numpy as np
from numba import cuda, float32, int32

# Configuration
BLOCKSIZEX = 32
BLOCKSIZEY = 8
ATOM_TILE = 128  # cooperative shared-memory tile

@cuda.jit
def dcs_kernel(atominfo,  # shape: (numatoms, 4) => (x, y, z_or_z2, w)
               numatoms,
               gridspacing,
               gridDimX,
               gridDimY,
               energygrid):  # shape: (gridDimY, gridDimX)
    # Shared memory tile (float4 equivalent)
    atomtile = cuda.shared.array(shape=(ATOM_TILE, 4), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Each thread processes one Y and 8 X positions spaced by BLOCKSIZEX
    yindex = by * BLOCKSIZEY + ty
    if yindex >= gridDimY:
        return

    xindex_base = bx * (8 * BLOCKSIZEX) + tx
    coory = gridspacing * yindex
    coorx = gridspacing * xindex_base
    gridspacing_coalesce = gridspacing * BLOCKSIZEX

    # Accumulators for 8 adjacent X points
    e1 = float32(0.0); e2 = float32(0.0); e3 = float32(0.0); e4 = float32(0.0)
    e5 = float32(0.0); e6 = float32(0.0); e7 = float32(0.0); e8 = float32(0.0)

    # Tile over atoms
    for base in range(0, numatoms, ATOM_TILE):
        # Cooperative load from global memory -> shared memory (coalesced)
        # Flatten 2D block indices to linear loader
        loader_idx = ty * BLOCKSIZEX + tx
        aidx = base + loader_idx
        if aidx < numatoms:
            atomtile[loader_idx, 0] = atominfo[aidx, 0]  # x
            atomtile[loader_idx, 1] = atominfo[aidx, 1]  # y
            atomtile[loader_idx, 2] = atominfo[aidx, 2]  # z^2 or z
            atomtile[loader_idx, 3] = atominfo[aidx, 3]  # w
        cuda.syncthreads()

        tileCount = ATOM_TILE
        if base + tileCount > numatoms:
            tileCount = numatoms - base

        for k in range(tileCount):
            ax = atomtile[k, 0]
            ay = atomtile[k, 1]
            az = atomtile[k, 2]  # interpret as z^2 (or adapt dyz2 accordingly)
            aw = atomtile[k, 3]

            dy = coory - ay
            dy2 = dy * dy
            dyz2 = dy2 + az

            dx1 = coorx
            dx1 -= ax
            dx2 = dx1 + gridspacing_coalesce
            dx3 = dx2 + gridspacing_coalesce
            dx4 = dx3 + gridspacing_coalesce
            dx5 = dx4 + gridspacing_coalesce
            dx6 = dx5 + gridspacing_coalesce
            dx7 = dx6 + gridspacing_coalesce
            dx8 = dx7 + gridspacing_coalesce

            # rsqrtf equivalent via 1.0 / sqrtf
            e1 += aw * (1.0 / math.sqrt(dx1 * dx1 + dyz2))
            e2 += aw * (1.0 / math.sqrt(dx2 * dx2 + dyz2))
            e3 += aw * (1.0 / math.sqrt(dx3 * dx3 + dyz2))
            e4 += aw * (1.0 / math.sqrt(dx4 * dx4 + dyz2))
            e5 += aw * (1.0 / math.sqrt(dx5 * dx5 + dyz2))
            e6 += aw * (1.0 / math.sqrt(dx6 * dx6 + dyz2))
            e7 += aw * (1.0 / math.sqrt(dx7 * dx7 + dyz2))
            e8 += aw * (1.0 / math.sqrt(dx8 * dx8 + dyz2))
        cuda.syncthreads()

    # Write-back (row-major, X fastest)
    # energygrid is 2D: [Y, X]
    # For coalescing, each "wave" (n) writes contiguous X across threads
    if xindex_base < gridDimX:
        energygrid[yindex, xindex_base] += e1
    if xindex_base + 1 * BLOCKSIZEX < gridDimX:
        energygrid[yindex, xindex_base + 1 * BLOCKSIZEX] += e2
    if xindex_base + 2 * BLOCKSIZEX < gridDimX:
        energygrid[yindex, xindex_base + 2 * BLOCKSIZEX] += e3
    if xindex_base + 3 * BLOCKSIZEX < gridDimX:
        energygrid[yindex, xindex_base + 3 * BLOCKSIZEX] += e4
    if xindex_base + 4 * BLOCKSIZEX < gridDimX:
        energygrid[yindex, xindex_base + 4 * BLOCKSIZEX] += e5
    if xindex_base + 5 * BLOCKSIZEX < gridDimX:
        energygrid[yindex, xindex_base + 5 * BLOCKSIZEX] += e6
    if xindex_base + 6 * BLOCKSIZEX < gridDimX:
        energygrid[yindex, xindex_base + 6 * BLOCKSIZEX] += e7
    if xindex_base + 7 * BLOCKSIZEX < gridDimX:
        energygrid[yindex, xindex_base + 7 * BLOCKSIZEX] += e8

# Host-side setup example
import math
# from numba import cuda

gridDimX = 1024
gridDimY = 1024
gridspacing = 0.1

# Example atoms: float4 (x, y, z^2, w)
numatoms = 5000
atoms = np.zeros((numatoms, 4), dtype=np.float32)
atoms[:, 0] = np.random.uniform(-50, 50, size=numatoms)  # x
atoms[:, 1] = np.random.uniform(-50, 50, size=numatoms)  # y
z = np.random.uniform(-50, 50, size=numatoms)            # z
atoms[:, 2] = (z * z).astype(np.float32)                 # z^2 for dyz2 = dy^2 + z^2
atoms[:, 3] = np.random.uniform(0.1, 1.0, size=numatoms) # w (charge or weight)

energy = np.zeros((gridDimY, gridDimX), dtype=np.float32)

d_atoms = cuda.to_device(atoms)
d_energy = cuda.to_device(energy)

threadsperblock = (BLOCKSIZEX, BLOCKSIZEY)
blockspergrid = ( (gridDimX + (8 * BLOCKSIZEX) - 1) // (8 * BLOCKSIZEX),
                  (gridDimY + BLOCKSIZEY - 1) // BLOCKSIZEY )

dcs_kernel[blockspergrid, threadsperblock](d_atoms,
                                           np.int32(numatoms),
                                           np.float32(gridspacing),
                                           np.int32(gridDimX),
                                           np.int32(gridDimY),
                                           d_energy)

energy = d_energy.copy_to_host()

