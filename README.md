# README #

This is a simulation calculating the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set).
Number crunching code is done in OpenCL, which is then called in Python and visualized using pyqtgraph.


#### Requirements ####

You will need an installation of OpenCL. For Intel CPUs/GPUs on Ubuntu you can follow this post on
[askubuntu](https://askubuntu.com/questions/850281/opencl-on-ubuntu-16-04-intel-sandy-bridge-cpu).
For other distributions and Nvidia or AMD GPUs you can follow Andreas Klöckner's
[wiki](https://wiki.tiker.net/OpenCLHowTo).

You also need python 3.x with the following packages:

    numpy
    pyopencl
    PyQt5
    pyqtgraph

I suggest installing [virtualenv & virtualenvwrapper](http://docs.python-guide.org/en/latest/dev/virtualenvs/),
so you don't clutter your system python installation with additional packages.


#### How do I set it up? ####

Clone the git repository

    git clone git@github.com:yboetz/mandelbrot_opencl.git

Then install the required python packages (best in your virtualenv)

    cd mandelbrot_opencl
    pip install -r requirements.txt

Run the widget

	python src/main.py


#### Key controls ####

Basic controls:

+ W/A/S/D - move around
+ E - zoom in
+ Q - zoom out
+ C - set number of color steps
+ I - set number of iterations to calculate
+ R - recenter and zoom out
+ Esc - quit app
