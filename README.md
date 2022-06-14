# Deep-Kernel-Method-for-PET-Image-Reconstruction


This is a package showing how to train a deep kernel model and test it on dynamic PET reconstruction. The method is described in:

S. Q.  Li, and G. B. Wang, Deep Kernel Representation for Image Reconstruction in PET. 
IEEE Transactions on Medical Imaging, in press, May 2022 (doi: 10.1109/TMI.2022.3176002).

Programer: Siqi Li

Last updated date: 6/13/2022

Prerequistites:
	Python 3.7 (or 3.x)
	PyTorch
	Matlab R2021a

Note that, the current deep kernel is a single-subject learning method. The training and testing projection data must be same!

1. Deep kernel training:

a). 	You can re-train the deep kernel model to obtain the kernel matrix by running Train_deel_kernel.py
	An initial model is stored in 'initialization model' folder.
	Otherwise you can use my trained model which is stored in 'trained models' folder.


b).	The training data is stored in 'training data' folder, including composite prior images (CIP), 
	corrupted CIP as noise image, and pre-defined neighbor index.

2. Kernelized EM for PET reconstruction:

a).	To use this package, you need to add the KER_v0.2 package into your matlab path by
  	running setup.m in matlab. KER_v0.2 package can be downloaded from:

	https://wanglab.faculty.ucdavis.edu/code

b).	To test the algorithms in the package, run "demo/demo_Deep_Kernel.m" in the demo folder. You
  	may need your own system matrix G or use Jeff Fessler's IRT matlab toolbox to 
  	generate one. IRT can downloaded from 
  
      	http://web.eecs.umich.edu/~fessler/code/index.html

c).	We used our trained deep kernel model to generate a kernel matrix in demo and compared proposed deep kernel with ML-EM and
	conventional kernel method. The example result is stored in 'fig' folder. 
	Please load your trained pairwise weight to generate the kernel matrix in the line 141 in demo_Deep_Kernel.m for testing if you are interesting.


This package is the proprietary property of The Regents of the University of California.
 
Copyright © 2019 The Regents of the University of California, Davis. 
All Rights Reserved. 
 
This software may be patent pending.
 
The software program and documentation are suppluntitled.mied "as is", without any 
accompanying services from The Regents, for purposes of confidential discussions 
only. The Regents does not warrant that the operation of the program will be 
uninterrupted or error-free. The end-user understands that the program was 
developed for research purposes and is advised not to rely exclusively on 
the program for any reason.
 
IN NO EVENT SHALL THE REGENTS OF THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY
PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, 
INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, 
EVEN IF THE REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE REGENTS 
SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE REGENTS HAS NO 
OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS. 



Please feel free to contact me (Siqi Li) if you have any questions: sqlli@ucdavis.edu
