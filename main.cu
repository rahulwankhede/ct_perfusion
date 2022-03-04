/*
 * 	nvcc -c -I /usr/local/cuda/include Main_SPD.cu
 * 	gcc -o a.out Main_SPD.o -L /usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -lstdc++
 */

#include <time.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define L 109
#define height 366
#define width 280

#define LAMBDA 0.06
#define AIF_SCALE 2
#define N (2*109)

double *transpose(double *A, int n){
	double temp;
	for(int i = 0; i < n; i++){
		for(int j = i+1; j < n; j++){
			temp = A[i*n + j];
			A[i*n + j] = A[j*n + i];
			A[j*n + i] = temp;
		}
	}
	return A;
}

double max_element(double *A, int n){
	double temp = A[0];
	for(int i = 1; i < n; i++){
		temp = (A[i] > temp) ? A[i] : temp;
	}
	return temp;
}


double *pinv(double *A, int n){
	for(int i = 0; i < n/2; i++){
		A[i] = (A[i] != 0) ? (1/A[i]) : 0;
	}
	return A;
}

__global__ void truncate_and_inverse_vector(double *S, double lambda){
	int id = threadIdx.x;
	double max = S[0];
	double thresh = max*lambda;

	if(S[id] < thresh){
		S[id] = 0.0;
	}
	else{
		S[id] = 1.0/S[id];
	}
}

__global__ void transpose_matrix(double *A, int n){
	int i = threadIdx.x;
	int j = blockIdx.x;

	double temp;
	if(i > j){
		temp = A[i + n*j];
		A[i + n*j] = A[j +n*i];
		A[j + n*i] = temp;
	}
}


__global__ void multiply_dVdS(double *dV, double *dS, int n){
	int i = threadIdx.x;
	int j = blockIdx.x;

	dV[i + n*j] *= dS[j];
}

__global__ void multiply_matrices(double *A, double *B, double *C, int n){
	int i = threadIdx.x;
	int j = blockIdx.x;

	double temp = 0.0;
	for(int k = 0; k < n; k++){
		temp += A[i + n*k] * B[k + n*j];
	}
	C[i + n*j] = temp;
}

__global__ void matVecMultKernel(double *invD, double *tempvector, double *R){
	int row = threadIdx.x;
	
	double temp = 0.0;

	for(int col = 0; col < N; col++){
		temp += invD[row + N*col] * tempvector[col];
	}

	R[row] = temp;
}


__global__ void init_tempvector(double *tempvector, double *inmap, int i, int j){
	int row = threadIdx.x;
	tempvector[row] = (row < L) ? inmap[row*height*width + i*width + j] : 0.0;
}


__global__ void shift_vector(double *R, double *Rshift){
	int maxindex = 0;
	for(int l = 0; l < N; l++){
		if(R[l] > R[maxindex]){
			maxindex = l;
		}
	}
	for(int l = 0; l < N; l++){
		Rshift[l] = R[(l + maxindex) % N];
	}
}

__global__ void store_vector(double *d_Rshift, double *d_ptr1, int i, int j){
	for(int l = 0; l < N; l++){
		d_ptr1[l*height*width + i*width + j] = d_Rshift[l];
	}
}

__global__ void shift_and_store(double *R, double *Rshift, double *ptr1, int i, int j){
	
	if(R[N-1] > R[0]){
		int maxindex = 0;
		for(int l = 0; l < N; l++){
			if(R[l] > R[maxindex]){
				maxindex = l;
			}
		}
		for(int l = 0; l < N; l++){
			Rshift[l] = R[(l + maxindex) % N];
		}
		R = Rshift;
	}

	for(int l = 0; l < N; l++){
		ptr1[l*height*width + i*width + j] = R[l];
	}
	

	if(i == 142 && j == 265){
		for(int l = 0; l < N; l++){
			printf("ptr1[%d, %d, %d] = %2.3lf\n", l, i, j, ptr1[l*height*width + i*width + j]);
		}
	}

}

__global__ void divKernel(double *d_ptr1, int dt){
	int i = threadIdx.x;
	int j = threadIdx.y;
	int k = threadIdx.z;

	d_ptr1[i*height*width + j*width + k] /= dt;

}


__global__ void printVecKernel(double *tempvector){
	int i = threadIdx.x;
	printf("tempvector[%d] = %2.3lf\n", i, tempvector[i]);
}

__global__ void set_zeros(double *A){
	for(int i = 0; i < N*height*width; i++){
		A[i] = 0.0;
	}
}

double ***pct_bsvd(double *inmap, double *aif, int dt, double lambda, int aif_scale){
	double *D = (double *) malloc(N * N * sizeof(double));
	for(int i = 0; i < N * N; i++){
		D[i] = 0.0;
	}
	for(int col = 0; col < N; col++){
		for(int i = 0; i < L; i++){
			D[col*N + ((i+col) % N)] = aif[i];
		}
	}


	// Making A = D so I don't have to change A to D everywhere
	double *A = D;

	//time start
	clock_t s, e;
	double used;
	
	s = clock();


	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;

	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;
	cudaError_t cudaStat6 = cudaSuccess;
	cudaError_t cudaStat7 = cudaSuccess;
	cudaError_t cudaStat8 = cudaSuccess;
	
	const int m = N;
	const int n = N;
	const int lda = m;

	double *d_A = NULL;
	double *d_S = NULL;
	double *d_U = NULL;
	double *d_VT = NULL;
	int *devInfo = NULL;
	double *d_work = NULL;
	double *d_rwork = NULL;
	double *d_W = NULL; // W = S*VT

	int lwork = 0;
	int info_gpu = 0;

	// Step 1: Create cusolverDn/cublas handle
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);


	// Step 2: Copy A to device
	cudaStat1 = cudaMalloc((void **) &d_A, sizeof(double) * lda * n);
	cudaStat2 = cudaMalloc((void **) &d_S, sizeof(double) * n);
	cudaStat3 = cudaMalloc((void **) &d_U, sizeof(double) * lda * m);
	cudaStat4 = cudaMalloc((void **) &d_VT, sizeof(double) * lda * n);
	cudaStat5 = cudaMalloc((void **) &d_W, sizeof(double) * lda * n);
	cudaStat6 = cudaMalloc((void **) &devInfo, sizeof(int));


	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	assert(cudaSuccess == cudaStat5);
	assert(cudaSuccess == cudaStat6);

	cudaStat7 = cudaMemcpy(d_A, A, sizeof(double) * lda * n, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat7);

	// Step 3: Query working space of SVD
	cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	cudaStat8 = cudaMalloc((void **) &d_work, sizeof(double) * lwork);
	assert(cudaSuccess == cudaStat8);


	// Step 4: Compute SVD
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n rows of VT

	cusolver_status = cusolverDnDgesvd(
		cusolverH,
		jobu,
		jobvt,
		m,
		n,
		d_A,
		lda,
		d_S,
		d_U,
		lda,	// ldu
		d_VT,
		lda, 	// ldvt
		d_work,
		lwork,
		d_rwork,
		devInfo
	);

	cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

	assert(cudaSuccess == cudaStat1);

	printf("after gesvd: info_gpu = %d\n", info_gpu);
	assert(0 == info_gpu);
	printf("=====\n");
	

	//time end
	e = clock();
	used = ((double) (e - s)) / CLOCKS_PER_SEC;

	printf("%lf\n", used);

	truncate_and_inverse_vector<<<1, N>>>(d_S, lambda);

	cudaDeviceSynchronize();


	double *dS_inv = d_S;

	transpose_matrix<<<N, N>>>(d_U, N);
	cudaDeviceSynchronize();


	transpose_matrix<<<N, N>>>(d_VT, N);
	cudaDeviceSynchronize();


	double *dU_inv = d_U;
	double *dV_inv = d_VT;	//V;

	multiply_dVdS<<<N, N>>>(dV_inv, dS_inv, N);

	cudaDeviceSynchronize();

	double *dW = dV_inv;

	double *d_invD;
	cudaError_t cudaStat9 = cudaMalloc((void **) &d_invD, N * N * sizeof(double));
    assert(cudaSuccess == cudaStat9);

	multiply_matrices<<<N, N>>>(dW, dU_inv, d_invD, N);
	
	cudaDeviceSynchronize();

	
	//k = zeros(N, height, width);
	double *d_ptr1;
	cudaError_t cudaStat10 = cudaMalloc((void **) &d_ptr1, N * height * width * sizeof(double));
    assert(cudaSuccess == cudaStat10);

	//set d_ptr1 to all zeros here just for completeness sake
	set_zeros<<<1,1>>>(d_ptr1);
	cudaDeviceSynchronize();



	double *d_tempvector;
	cudaError_t cudaStat13 = cudaMalloc((void **) &d_tempvector, N * sizeof(double));
    assert(cudaSuccess == cudaStat13);

	double *d_R;
	cudaError_t cudaStat14 = cudaMalloc((void **) &d_R, N * sizeof(double));
    assert(cudaSuccess == cudaStat14);

	double *d_Rshift;
	cudaError_t cudaStat15 = cudaMalloc((void **) &d_Rshift, N * sizeof(double));
    assert(cudaSuccess == cudaStat15);

	double *d_inmap;
	cudaError_t cudaStat16 = cudaMalloc((void **) &d_inmap, L * height * width * sizeof(double));
	assert(cudaSuccess == cudaStat16);


	//Now copy inmap to d_inmap here
	cudaError_t cudaStat17 = cudaMemcpy(d_inmap, inmap, L * height * width * sizeof(double), cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat17);

	/*Deconvolve*/

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			init_tempvector<<<1, N>>>(d_tempvector, d_inmap, i ,j);
			cudaDeviceSynchronize();

			//multiply d_invD with d_tempvector and store in d_R
			matVecMultKernel<<<1,N>>>(d_invD, d_tempvector, d_R);
			cudaDeviceSynchronize();

			shift_vector<<<1,1>>>(d_R, d_Rshift);
			cudaDeviceSynchronize();

			store_vector<<<1,1>>>(d_Rshift, d_ptr1, i, j);
			cudaDeviceSynchronize();
		}
	}


	double *ptr1 = (double *) malloc(N * height * width * sizeof(double));

	double **ptr2 = (double **) malloc(N * height * sizeof(double *));
	for(int i = 0; i < N*height; i++){
		ptr2[i] = ptr1 + i*width;
	}

	double ***ptr3 = (double ***) malloc(N * sizeof(double **)); //ptr3 = k
	for(int i = 0; i < N; i++){
		ptr3[i] = ptr2 + i*height;
	}
 
	//here, copy d_ptr1 to ptr1 and return ptr3
	cudaError_t cudaStat18 = cudaMemcpy(ptr1, d_ptr1, N * height * width * sizeof(double), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat18);
    
    // Free resources
    if(d_A) cudaFree(d_A);
    if(d_S) cudaFree(d_S);
    if(d_U) cudaFree(d_U);
    if(d_VT) cudaFree(d_VT);
    if(devInfo) cudaFree(devInfo);
    if(d_work) cudaFree(d_work);
    if(d_rwork) cudaFree(d_rwork);
    if(d_W) cudaFree(d_W);

    if(d_invD) cudaFree(d_invD);
    if(d_ptr1) cudaFree(d_ptr1);
	if(d_inmap) cudaFree(d_inmap);
	
	if(cublasH) cublasDestroy(cublasH);
    if(cusolverH) cusolverDnDestroy(cusolverH);

    cudaDeviceReset();


	return ptr3; // = k

}

void pct_cbf(double ***rmap, double rho, int **mask, double **cbf_map){
	double scaling_factor = 6000.0 / rho;
	for(int h = 0; h < height; h++){
		for(int w = 0; w < width; w++){
			double temp = 0.0;
			for(int t = 0; t < L; t++){
				temp = (rmap[t][h][w] > temp) ? rmap[t][h][w] : temp;
			}
			if(!mask[h][w]){
				temp = 0.0;
			}
			cbf_map[h][w] = scaling_factor * temp;
		}
	}
}


double **pct_map(double *data, double *AIF, int **Mask, double lambda, int aif_scale){
	/*PCT Parameters*/
	double rho = 1.05;
	int POST_bbbp = 90;
	POST_bbbp = (POST_bbbp < L) ? POST_bbbp : L;
	int dt = 1; //time interval between samples in second, int for now, try double later

	/*SVD Parameters*/

	/*Calculate the residue functions*/
	//R = pct_bsvd(data, AIF, dt, lambda, m, Mask);
	//R is a 109x366x280 double matrix
	double ***R = pct_bsvd(data, AIF, dt, lambda, aif_scale);

	/*Calculate a CBF map*/
	//CBF = pct_cbf();
	double **CBF = (double **) malloc(height * sizeof(double *));
    CBF[0] = (double *) malloc(height * width * sizeof(double));
    for(int i = 1; i < height; i++){
        CBF[i] = CBF[0] + i*width;
    }

	pct_cbf(R, rho, Mask, CBF);

	return CBF;

}

int main(){

	/*Parameters for you to change*/
	double lambda = LAMBDA;
	int aif_scale = AIF_SCALE;

	/*Load data here*/
	//C0, C1, AIF0, AIF1, Mask
	double *C0 = (double *) malloc(L * height * width * sizeof(double));

	//read from mydata.txt here and set C0 = that new array
	FILE *fp;
	fp = fopen("mydata.txt", "r");
	double *array = (double *) malloc(L*height*width * sizeof(double));
	size_t n1 = fread((void *) array, sizeof(double), L*height*width, fp);
	for(int i = 0; i < L; i++){
		for(int j = 0; j < height; j++){
			for(int k = 0; k < width; k++){
				C0[i*height*width + j*width + k] = array[i + j*L + k*L*height];
			}
		}
	}

	int **Mask = (int **) malloc(height * sizeof(int *));
	Mask[0] = (int *) malloc(height * width * sizeof(int));
	for(int i = 1; i < height; i++){
		Mask[i] = Mask[0] + width*i;
	}
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			Mask[i][j] = 1;
		}
	}

	double *AIF0 = (double *) malloc(L * sizeof(double));
	FILE *fp2;
	fp2 = fopen("aifvector.txt", "r");
	size_t n2 = fread((void *) AIF0, sizeof(double), L, fp2);


	clock_t start, end;
	double cpu_time_used;
	
	start = clock();

	/*Here be pct_map*/
	double **cbf = pct_map(C0, AIF0, Mask, lambda, aif_scale);
	
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("%lf\n", cpu_time_used);

	//Now, save this to a file that matlab can read
	//cbf is a height x width matrix of doubles

	FILE *fp3;
	fp3 = fopen("cbfmatrix.txt", "w");
	size_t n3 = fwrite((void *) cbf[0], sizeof(double), height * width, fp3);

	return 0;
}
