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

def mandelbrot(xmin, xmax, ymin, ymax, xsize, ysize, maxIter, color, data):
    dx = (xmax - xmin) / xsize
    dy = (ymin - ymax) / ysize
    
    output_cl = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)
            
    prg.mandelbrot(queue, (xsize//8, ysize), None, np.float64(xmin), np.float64(ymax), 
                   np.float64(dx), np.float64(dy), np.uint16(maxIter), np.uint16(color), output_cl)

    cl.enqueue_copy(queue, data, output_cl).wait()

def move_right(colums, xmin, xmax, ymin, ymax, xsize, ysize, maxIter, color, data):
    if colums == 0: 
        return
    xscale = (xmax - xmin) / xsize
    data = data.reshape((ysize,xsize))
    data[:,:] = np.roll(data, -colums, axis=1)
    tmp = np.zeros(ysize*colums, dtype=np.uint16)
    mandelbrot(xmax, xmax + xscale*colums, ymin, ymax, colums, ysize, maxIter, color, tmp)
    data[:,xsize-colums:xsize] = tmp.reshape((ysize, colums))
    data = data.reshape(xsize*ysize)

def move_left(colums, xmin, xmax, ymin, ymax, xsize, ysize, maxIter, color, data):
    if colums == 0: 
        return
    xscale = (xmax - xmin) / xsize
    data = data.reshape((ysize,xsize))
    data[:,:] = np.roll(data, colums, axis=1)
    tmp = np.zeros(ysize*colums, dtype=np.uint16)
    mandelbrot(xmin - xscale*colums, xmin, ymin, ymax, colums, ysize, maxIter, color, tmp)
    data[:,0:colums] = tmp.reshape((ysize, colums))
    data = data.reshape(xsize*ysize)

def move_up(rows, xmin, xmax, ymin, ymax, xsize, ysize, maxIter, color, data):
    if rows == 0: 
        return
    yscale = (ymax - ymin) / ysize
    data = data.reshape((ysize,xsize))
    data[:,:] = np.roll(data, rows, axis=0)
    tmp = np.zeros(xsize*rows, dtype=np.uint16)
    mandelbrot(xmin, xmax, ymax , ymax + yscale*rows, xsize, rows, maxIter, color, tmp)
    data[0:rows,:] = tmp.reshape((rows, xsize))
    data = data.reshape(xsize*ysize)

def move_down(rows, xmin, xmax, ymin, ymax, xsize, ysize, maxIter, color, data):
    if rows == 0: 
        return
    yscale = (ymax - ymin) / ysize
    data = data.reshape((ysize,xsize))
    data[:,:] = np.roll(data, -rows, axis=0)
    tmp = np.zeros(xsize*rows, dtype=np.uint16)
    mandelbrot(xmin, xmax, ymin - yscale*rows, ymin, xsize, rows, maxIter, color, tmp)
    data[ysize-rows:ysize,:] = tmp.reshape((rows, xsize))
    data = data.reshape(xsize*ysize) 
