# README #

This is a simulation calculating the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set).
Number crunching code is done in OpenCL, which is then called in Python and visualized using pyqtgraph.


#### Requirements ####

You need python 3.x with the following packages:

    numpy
    pyopencl
    PyQt5
    pyqtgraph

I suggest installing [virtualenv & virtualenvwrapper](http://docs.python-guide.org/en/latest/dev/virtualenvs/),
so you don't clutter your system python installation with additional packages.


#### How do I set it up? ####

Clone the git repository

    git clone git@github.com:yboetz/mandelbrot.git

Then install the required python packages (best in your virtualenv)

    cd mandelbrot
    pip install -r requirements.txt

Run the widget

	python main.py


#### Key controls ####

Basic controls:

+ W/A/S/D - move around
+ E - zoom in
+ Q - zoom out
+ C - set number of color steps
+ I - set number of iterations to calculate
+ R - recenter and zoom out
+ Esc - quit app