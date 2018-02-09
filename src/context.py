# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:45:39 2015

@author: yboetz
"""

import os, sys
import numpy as np
import pyopencl as cl

# choose platform and device
platforms = cl.get_platforms()
if not platforms:
    print('No platforms found, exiting...')
    sys.exit(0)

if len(platforms) > 1:
    print('Choose platform to use:')
    for i,p in enumerate(platforms):
        print(f'[{i}]: {p.name}')

    while True:
        try:
            devices = platforms[int(input())].get_devices()
            break
        except (ValueError, IndexError):
            print('Please enter a valid number...')
        except KeyboardInterrupt:
            print('Interrupted...')
            sys.exit(0)
else:
    devices = platforms[0].get_devices()

if not devices:
    print('No devices found, exiting...')
    sys.exit(0)

if len(devices) > 1:
    print('Choose device to use:')
    for i,d in enumerate(devices):
        print(f'[{i}]: {d.name}')

    while True:
        try:
            dev = devices[int(input())]
            break
        except (ValueError, IndexError):
            print('Please enter a valid number...')
        except KeyboardInterrupt:
            print('Interrupted...')
            sys.exit(0)
else:
    dev = devices[0]

ctx = cl.Context(devices=[dev])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

TYPE = cl.device_type.to_string(dev.type)
TYPE = TYPE if TYPE in ['CPU', 'GPU'] else 'CPU'
VEC_SIZE = 8 if TYPE=='CPU' else 1

dirname = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dirname, f'kernel_{TYPE}.cl'), 'r') as file:
    code = file.read()

prg = cl.Program(ctx, code).build()

def mandelbrot(xmin, xmax, ymin, ymax, xsize, ysize, maxIter, color, data):
    dx = (xmax - xmin) / xsize
    dy = (ymax - ymin) / ysize

    output_cl = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    prg.mandelbrot(queue, (xsize//VEC_SIZE, ysize), None, np.float64(xmin), np.float64(ymax),
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
