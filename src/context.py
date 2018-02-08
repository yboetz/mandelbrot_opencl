# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:45:39 2015

@author: yboetz
"""

# Create context and define kernel

import numpy as np
import pyopencl as cl

platforms = cl.get_platforms()
devs = platforms[0].get_devices()
print(f'Using device {devs[0].name}')
ctx = cl.Context(devices = devs)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

code = """
    __kernel
    __attribute__(( vec_type_hint(double8))) 
    void mandelbrot(const double xMin, const double yMax, const double dx, const double dy, 
                    const ushort maxIter, const ushort color, __global ushort *output)
    {
    const ushort Idx = get_global_id(0) * 8;
    const ushort Idy = get_global_id(1);
    const int Index = Idx + 8 * get_global_size(0) * Idy;
    
    const double8 vdx = (double8)(dx);
    const double8 vdy = (double8)(dy);
    const double8 vxMin = (double8)(xMin);
    const double8 vyMax = (double8)(yMax);
    const double8 fours = (double8)(4);
    const double8 vIdx = (double8)(Idx) + (double8)(0,1,2,3,4,5,6,7);
    const double8 vIdy = (double8)(Idy);
    
    const double8 c_re = vxMin + vIdx * vdx;
    const double8 c_im = vyMax + vIdy * vdy;
    
    double8 re = (double8)(0);
    double8 im = (double8)(0);
    double8 re_2 = (double8)(0);
    double8 im_2 = (double8)(0);
    
    int8 counter = (int8)(0);
    int8 which = (int8)(-1);
    ushort i;
     
    for(i = 0; i < maxIter && any(which); i++) 
        {
        im = re*im;
        im = 2*im + c_im;
        re = re_2 - im_2 + c_re;
        re_2 = re*re;
        im_2 = im*im;
        
        which = convert_int8(re_2 + im_2 < fours);
        counter -= which;      
        }
    vstore8(convert_ushort8(counter % color), 0, output + Index);
    }
    """

prg = cl.Program(ctx,code).build()

def mandelbrot(xMin, yMax, xScale, yScale, xSize, ySize, maxIter, color, data):
    dx = xScale / xSize
    dy = yScale / ySize
    
    output_cl = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)
            
    prg.mandelbrot(queue, (xSize//8, ySize), None, np.float64(xMin), np.float64(yMax), 
                   np.float64(dx), np.float64(dy), np.uint16(maxIter), np.uint16(color), output_cl)

    cl.enqueue_copy(queue, data, output_cl).wait()

def move_right(colums, xMin, yMax, xScale, yScale, xSize, ySize, maxIter, color, data):
    if colums == 0: 
        return
    data = data.reshape((ySize,xSize))
    data[:,:] = np.roll(data, -colums, axis=1)
    tmp = np.zeros(ySize*colums, dtype=np.uint16)
    mandelbrot(xMin + xScale, yMax, xScale / xSize * colums, yScale, colums, ySize, maxIter, color, tmp)
    data[:,xSize-colums:xSize] = tmp.reshape((ySize, colums))
    data = data.reshape(xSize*ySize)

def move_left(colums, xMin, yMax, xScale, yScale, xSize, ySize, maxIter, color, data):
    if colums == 0: 
        return
    data = data.reshape((ySize,xSize))
    data[:,:] = np.roll(data, colums, axis=1)
    tmp = np.zeros(ySize*colums, dtype=np.uint16)
    mandelbrot(xMin - colums / xSize * xScale, yMax, xScale / xSize * colums, yScale, colums, ySize, maxIter, color, tmp)
    data[:,0:colums] = tmp.reshape((ySize, colums))
    data = data.reshape(xSize*ySize)

def move_up(rows, xMin, yMax, xScale, yScale, xSize, ySize, maxIter, color, data):
    if rows == 0: 
        return
    data = data.reshape((ySize,xSize))
    data[:,:] = np.roll(data, rows, axis=0)
    tmp = np.zeros(xSize*rows, dtype=np.uint16)
    mandelbrot(xMin, yMax - rows / ySize * yScale, xScale, yScale / ySize * rows, xSize, rows, maxIter, color, tmp)
    data[0:rows,:] = tmp.reshape((rows, xSize))
    data = data.reshape(xSize*ySize)

def move_down(rows, xMin, yMax, xScale, yScale, xSize, ySize, maxIter, color, data):
    if rows == 0: 
        return
    data = data.reshape((ySize,xSize))
    data[:,:] = np.roll(data, -rows, axis=0)
    tmp = np.zeros(xSize*rows, dtype=np.uint16)
    mandelbrot(xMin, yMax + yScale, xScale, yScale / ySize * rows, xSize, rows, maxIter, color, tmp)
    data[ySize-rows:ySize,:] = tmp.reshape((rows, xSize))
    data = data.reshape(xSize*ySize) 
