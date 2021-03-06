# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:30:47 2018

@author: yboetz
"""

import os
import sys
from time import time
import numpy as np
import pyopencl as cl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from utils import get_color_map, generate_lut
from context import *


class Fractal:
    def __init__(self, xsize, ysize, xmin, xmax, ymin, ymax, maxIter, color, data):
        self.xsize, self.ysize = xsize, ysize
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.maxIter = maxIter
        self.color = color
        self.data = data
        self.redraw()
    
    def redraw(self):
        mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax, self.xsize, self.xsize, 
                   self.maxIter, self.color, self.data)

    def setExtent(self, x, y):
        xscale = (self.xmax - self.xmin);
        yscale = (self.ymax - self.ymin);
        xmiddle = self.xmin + xscale / self.xsize * x;
        ymiddle = self.ymax - yscale / self.ysize * y;
        self.xmin = xmiddle - 0.5 * xscale;
        self.xmax = xmiddle + 0.5 * xscale;
        self.ymin = ymiddle - 0.5 * yscale;
        self.ymax = ymiddle + 0.5 * yscale;
        
    def zoom(self, zoomFactor):
        xscale = (self.xmax - self.xmin)
        yscale = (self.ymax - self.ymin)
        self.xmin += 0.5 * (1-zoomFactor) * xscale
        self.xmax -= 0.5 * (1-zoomFactor) * xscale
        self.ymin += 0.5 * (1-zoomFactor) * yscale
        self.ymax -= 0.5 * (1-zoomFactor) * yscale
        self.redraw()
           
    def moveR(self, columns):
        move_right(columns, self.xmin, self.xmax, self.ymin, self.ymax, self.xsize, self.xsize, 
                   self.maxIter, self.color, self.data)
        self.xmin += (self.xmax - self.xmin)/self.xsize * columns
        self.xmax += (self.xmax - self.xmin)/self.xsize * columns
        
    def moveL(self, columns):
        move_left(columns, self.xmin, self.xmax, self.ymin, self.ymax, self.xsize, self.xsize, 
                  self.maxIter, self.color, self.data)
        self.xmin -= (self.xmax - self.xmin)/self.xsize * columns
        self.xmax -= (self.xmax - self.xmin)/self.xsize * columns
        
    def moveU(self, rows):
        move_up(rows, self.xmin, self.xmax, self.ymin, self.ymax, self.xsize, self.xsize, 
                self.maxIter, self.color, self.data)
        self.ymin += (self.ymax - self.ymin)/self.ysize * rows
        self.ymax += (self.ymax - self.ymin)/self.ysize * rows
        
    def moveD(self, rows):
        move_down(rows, self.xmin, self.xmax, self.ymin, self.ymax, self.xsize, self.xsize, 
                  self.maxIter, self.color, self.data)
        self.ymin -= (self.ymax - self.ymin)/self.ysize * rows
        self.ymax -= (self.ymax - self.ymin)/self.ysize * rows

    def setMaxIt(self, maxIter):
        self.maxIter = maxIter
        self.redraw()

    def setCol(self, color):
        self.color = color
        self.redraw()


class FractalWidget(pg.GraphicsLayoutWidget):
    """GraphicsLayoutWidget class with controls & fractal data"""
    def __init__(self):
        super(FractalWidget, self).__init__()

        self.xsize, self.ysize = 1024, 1024
        self.xmin, self.xmax = -2, 1
        self.ymin, self.ymax =  -1.5, 1.5
        self.maxit, self.col = 200, 200

        self.moveSpeed = 24
        self.zoomSpeed = 1.0667

        # Array to hold iteration count for each pixel
        self.data = np.zeros(self.xsize*self.ysize, dtype=np.uint16)

        # Add view box
        view = self.addViewBox(lockAspect=True, enableMouse=False, enableMenu=False, invertY=True)

        # Create image item
        self.ip = pg.ImageItem(border='w')
        lut = generate_lut(color_map=get_color_map(os.path.dirname(os.path.abspath(__file__)) + '/../viridis'))
        self.ip.setLookupTable(lut, update=False)
        self.createFractal()
        view.addItem(self.ip)

        # Create TextItem to display fps and coordinates
        self.ti = pg.TextItem()
        self.ti.setPos(10, self.ysize)
        view.addItem(self.ti)

        # Connect mouse click event to function mouseEvent
        self.scene().sigMouseClicked.connect(self.mouseEvent)

        # Dictionary with functions to call at keypress
        self.staticKeyList = {
                              QtCore.Qt.Key_R: self.createFractal
                             }
        # Functions which get continuously called at keypress
        self.movementKeyList = {
                                QtCore.Qt.Key_E: self.zoomIn,
                                QtCore.Qt.Key_Q: self.zoomOut,
                                QtCore.Qt.Key_A: self.moveL,
                                QtCore.Qt.Key_D: self.moveR,
                                QtCore.Qt.Key_S: self.moveD,
                                QtCore.Qt.Key_W: self.moveU
                               }
        # A set of currently pressed keys
        self.pressedKeys = set()

        # Timer which calls update function at const framerate
        self.tickRate = 1000 / 30
        self.timer = QtCore.QTimer()
        self.timer.start(self.tickRate)
        self.timer.timeout.connect(self.move)
        self.timer.timeout.connect(self.renderText)

        # Init fps counter
        self.fps = 1000 / self.tickRate
        self.lastTime = time()
        self.timer.timeout.connect(self.fpsCounter)

    # Renders text to display
    def renderText(self):
        self.ti.setText('Fps: %.0f\tRange: (%.15f, %.15f; %.15f, %.15f)'
                        %(self.fps, self.fractal.xmin, self.fractal.xmax,
                          self.fractal.ymin, self.fractal.ymax))

    # Calculates current fps
    def fpsCounter(self):
        self.now = time()
        dt = self.now - self.lastTime
        self.lastTime = self.now
        s = np.clip(dt*2., 0, 1)
        self.fps = self.fps * (1-s) + (1.0/dt) * s

    # Calls moveX functions according to which key is pressed
    def move(self):
        for key in self.pressedKeys:
            self.movementKeyList.get(key, self.doNothing)()

    # Empty function to call if any other key is pressed
    def doNothing(self):
        pass

    # Appends key to list when pressed
    def keyPressEvent(self, e):
        if e.isAutoRepeat():
            pass
        else:
            self.pressedKeys.add(e.key())
            self.staticKeyList.get(e.key(), self.doNothing)()

    # Removes key from list when released
    def keyReleaseEvent(self, e):
        if e.isAutoRepeat():
            pass
        else:
            try:
                self.pressedKeys.remove(e.key())
            except KeyError:
                pass

    # Takes mouse click and zooms in or out at position
    def mouseEvent(self, e):
        pos = self.ip.mapFromScene(e.scenePos())
        if not (0 < pos.x() < self.xsize) or not (0 < pos.y() < self.ysize):
            return
        bts = {1:self.zoomIn, 2:self.zoomOut}
        if e.button() in bts:
            self.fractal.setExtent(pos.x(), pos.y())
            print(pos.x(), pos.y())
            bts.get(e.button())()

    def updateImage(self):
        self.ip.setImage(self.data.reshape((self.ysize,self.xsize)).transpose(),
                         levels = (0,self.col))

    def createFractal(self):
        args = (self.xsize, self.ysize, self.xmin, self.xmax, self.ymin,
                self.ymax, self.maxit, self.col, self.data)
        self.fractal = Fractal(*args)
        self.updateImage()

    def zoomIn(self):
        self.fractal.zoom(1.0/self.zoomSpeed)
        self.updateImage()

    def zoomOut(self):
        self.fractal.zoom(self.zoomSpeed)
        self.updateImage()

    def moveL(self):
        self.fractal.moveL(self.moveSpeed)
        self.updateImage()

    def moveR(self):
        self.fractal.moveR(self.moveSpeed)
        self.updateImage()

    def moveD(self):
        self.fractal.moveD(self.moveSpeed)
        self.updateImage()

    def moveU(self):
        self.fractal.moveU(self.moveSpeed)
        self.updateImage()

    def setMaxIt(self):
        text, ok = QtGui.QInputDialog.getText(self, 'Iterations', 'Set number of iterations')
        if ok:
            try:
                self.fractal.setMaxIt(int(text))
                self.updateImage()
            except ValueError:
                pass

    def setCol(self):
        text, ok = QtGui.QInputDialog.getText(self, 'Colors', 'Set number of colors')
        if ok:
            try:
                self.fractal.setCol(int(text))
                self.col = int(text)
                self.updateImage()
            except ValueError:
                pass


class MainWindow(QtGui.QMainWindow):
    """Main window with file menu & statusbar"""
    def __init__(self):
        super(MainWindow, self).__init__()
        self.init()
        self.show()

    def init(self):
        self.resize(1024, 1024)
        self.setWindowTitle('Mandelbrot')

        self.window = FractalWidget()
#        self.keyPressEvent = self.window.keyPressEvent
        self.setCentralWidget(self.window)
#        self.statusBar()
        # Menubar entries
        closeApp = QtGui.QAction(QtGui.QIcon('quit.png'), 'Quit', self)
        closeApp.setShortcut('Escape')
#        closeApp.setStatusTip('Exit application')
        closeApp.triggered.connect(pg.exit)
        setIter = QtGui.QAction(QtGui.QIcon('quit.png'), 'Iterations', self)
        setIter.setShortcut('I')
#        setIter.setStatusTip('Set number of iterations to calculate')
        setIter.triggered.connect(self.window.setMaxIt)
        setCol = QtGui.QAction(QtGui.QIcon('quit.png'), 'Colors', self)
        setCol.setShortcut('C')
#        setCol.setStatusTip('Set number of colors')
        setCol.triggered.connect(self.window.setCol)
        # Add menubar
#        menubar = self.menuBar()
#        fileMenu = menubar.addMenu('&File')
#        fileMenu.addAction(setIter)
#        fileMenu.addAction(setCol)
#        fileMenu.addAction(closeApp)
        self.addAction(setIter)
        self.addAction(setCol)
        self.addAction(closeApp)
