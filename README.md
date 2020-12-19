
LaimbioNet
=============

**LaimbioNet** is a library and script collection for deep learning using medical images in Python, providing functionalities for **tfrecords**, **training** and **testing**
using **tensorflow**.


**Troubles?** Feel free to write me with any questions / comments / suggestions: javier.vera@urjc.es




Requirements
============
Python 3

Dependencies
------------
* `tensorflow 1.13`
* `scipy`
* `numpy`
* `nibabel` or `medpy`
* `opencv` (cv2)



Usage
--------
The data must be stored in a folder that must contains groups, for example:

````console
Paris_data-
          -train-
                -subject01
                -subject02
                -subject03
          -test -
                -subject_04
                -subject_05
              
````              
On the other hand the subjects folders (subject01,subject02...) must contains all the images for that subject such as: MRI.nii.gz,CT.nii.gz, Mask.nii.gz

