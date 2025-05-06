from gpu import thread_idx, block_idx, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from memory import stack_allocation
from layout import Layout, LayoutTensor
from math import iota
from sys import sizeof

def main():
    fn printing_kernel():
        print("GPU thread: [", thread_idx.x, thread_idx.y, thread_idx.z, "]")

    var ctx = DeviceContext()

    ctx.enqueue_function[printing_kernel](grid_dim=1, block_dim=4)
    ctx.synchronize()