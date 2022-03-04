# CT Perfusion by the Truncated-SVD based Deconvolution method parallelized using CUDA
This project takes Computed Tomography data obtained by a CT scanner and an Arterial Input Function (AIF) vector and gives the corresponding perfusion map in a span of seconds.

## How to run
Firstly, you'll need to have CUDA installed on your system (and a dedicated NVIDIA GPU of course). Also make sure gcc and nvcc are installed properly.

To use it, run the compile_script (first make it executable if it's not already using the command "chmod u+x compile_script.sh"), then run a.out.
You'll get a "cbfmatrix.txt" file which you can easily plot in any plotting program (or preferably in MATLAB with the command "ctshow").

## Sample output
Sample output on a brain CT scanner data taken from file "mydata.txt" and AIF vector taken from "aifvector.txt":
![Perfusion map obtained on sample data and AIF](images/parallel_deconvolution.jpg?raw=true "Perfusion map")
## For more details
If you have no idea what these terms such as "Perfusion" and "Deconvolution" mean, please read this nice paper over at https://www.hindawi.com/journals/ijbi/2011/467563/
