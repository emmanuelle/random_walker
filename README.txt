Random walker algorithm
=======================

from *Random walks for image segmentation*, Leo Grady, IEEE Trans 
Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.

Dependencies
------------

* numpy >= 1.4, scipy

* optional: pyamg, numexpr

Installing pyamg and using the 'cg_mg' mode of random_walker improves
significantly the performance.

Installing numexpr makes only a slight improvement. 

Installing
----------

sudo python setup.py install

Testing
-------

nosetests test_random_walker.py
