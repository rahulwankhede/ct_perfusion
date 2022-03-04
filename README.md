# CT Perfusion by the Truncated-SVD based Deconvolution method parallelized using CUDA

This project takes Computed Tomography data obtained by a CT scanner and an Arterial Input Function (AIF) vector and gives the corresponding perfusion map in a span of seconds

You'll need to have CUDA installed on your system with a dedicated NVIDIA GPU. Also make sure gcc and nvcc are installed properly.

To use it, run the compile_script (first make it executable if it's not already using the command "chmod u+x compile_script.sh"), then run a.out.
You'll get a "cbfmatrix.txt" file which you can easily plot in any plotting program such as MATLAB.

Sample output on CT scanner data taken from file "mydata.txt" and AIF vector taken from "aifvector.txt":
