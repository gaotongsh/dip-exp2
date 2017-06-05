    The Matlab Package for Landmark-Based Domain Adaptation
                   v.1, Dec, 2013

                   by Boqing Gong 
                 (boqinggo@usc.edu)


I n s t a l l a t i o n

- Download SVM-KM from http://asi.insa-rouen.fr/enseignants/~arakoto/toolbox/index.html
- Extract it to the subdirectory "SVM-KM"

- Download SimpleMKL from http://asi.insa-rouen.fr/enseignants/~arakoto/code/mklindex.html
- Extract it to the subdirectory "simplemkl"
- Perhaps you need re-compile some files in simplemkl:
    
    mex devectorize.c	
    mex devectorize_single.c
    mex vectorize.c
    mex vectorize_single.c


U s a g e

Ex_Landmark.m   :   to reproduce the results in [1]
Ex_GFK.m        :   to reproduce the results in [2]


C o n t a c t

Please send me a message via boqinggo@usc.edu if you have any suggestions
or questions with the code or reproducing the results.


R e f e r e n c e s 

[1] Connecting the Dots with Landmarks: Discriminatively Learning Domain-
    Invariant Features for Unsupervised Domain Adaptation. 
    B. Gong, K. Grauman, and F. Sha.
    Proceedings of the International Conference on Machine Learning (ICML), 
    Atlanta, GA, June 2013.

[2] Geodesic Flow Kernel for Unsupervised Domain Adaptation.  
    B. Gong, Y. Shi, F. Sha, and K. Grauman.  
    Proceedings of the IEEE Conference on Computer Vision and Pattern 
    Recognition (CVPR), Providence, RI, June 2012.

