#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include <math.h>
#include <string.h>

#include "matrix.h"

#define OPTIMIAL_THREAD 10

#define M_IDENTITY 1.0
#define M_SEQUENCE 2.0
#define M_UNIFORM 3.0
#define M_SORTED 4.0
#define M_RANDOM 5.0

#define CACHE_SIZE 1

#define CACHE_TYPE (g_elements)
#define CACHE_SORTED (g_elements+1)
#define CACHE_MIN (g_elements+2)
#define CACHE_MAX (g_elements+3)
#define CACHE_SUM (g_elements+4)
#define CACHE_DET (g_elements+5)
#define CACHE_TRACE (g_elements+6)

static ssize_t g_width = 0;
static ssize_t g_elements = 0;
static ssize_t g_nthreads = 1;


// void set_cache(float* matrix, float type, float sorted, float min, float max, float sum, float det, float trace){
	// matrix[CACHE_TYPE] = type;
	// matrix[CACHE_SORTED] = sorted;
	// matrix[CACHE_MIN] = min;
	// matrix[CACHE_MAX] = max;
	// matrix[CACHE_SUM] = sum;
	// matrix[CACHE_DET] =  det;
	// matrix[CACHE_TRACE] =  trace;
// }

////////////////////////////////
///    THREADING FUNCTIONS   ///
////////////////////////////////

// threads for void* make_matrix(void) operations.
void spawn_threads(void*(*funcptr)(void*), thread_args argv){
	const int nwidth = g_width;
	const int nelements = g_elements;
	const int nthreads = g_nthreads;
	
	pthread_t thread_ids[nthreads];
	void* args = NULL;
	
	int start = 0;
	int end = 0;
	
	float* result = argv.result;
	thread_type method = argv.type;
	int incre = 0;

	if(method == MMULTHREAD){
		args = (d_mthread*)malloc(sizeof(d_mthread)*nthreads);
		incre = sizeof(d_mthread);
		for(int id=0; id < nthreads; id++){
			end = id == nthreads - 1 ? nwidth : (id + 1) * (nwidth / nthreads);
			((d_mthread*)args)[id] = (d_mthread) {
				.result = result,
				.start = start,
				.end = end,
				.matrix_a = argv.args.matrix.matrix_a,
				.matrix_b = argv.args.matrix.matrix_b,
			};
			start = end;
		}
	}
	else if(method == MADDTHREAD){
		args = (d_mthread*)malloc(sizeof(d_mthread)*nthreads);
		incre = sizeof(d_mthread);
		for(int id=0; id < nthreads; id++){
			end = id == nthreads - 1 ? nelements : (id + 1) * (nelements / nthreads);
							
			((d_mthread*)args)[id] = (d_mthread) {
				.result = result,
				.start = start,
				.end = end,
				.matrix_a = argv.args.matrix.matrix_a,
				.matrix_b = argv.args.matrix.matrix_b,
			};
			start = end;
		}
	}
	else if(method == STHREAD){
		args = (d_sthread*)malloc(sizeof(d_sthread)*nthreads);
		incre = sizeof(d_sthread);
		for(int id=0; id < nthreads; id++){
			end = id == nthreads - 1 ? nelements : (id + 1) * (nelements / nthreads);
							
			((d_sthread*)args)[id] = (d_sthread) {
				.result = result,
				.start = start,
				.end = end,
				.value = argv.args.scalar.scalar,
				.matrix = argv.args.scalar.matrix,
			};
			start = end;
		}
	}
	else if(method == OTHREAD){
		args = (d_othread*)malloc(sizeof(d_othread)*nthreads);
		incre = sizeof(d_othread);
		for(int id=0; id < nthreads; id++){
			end = id == nthreads - 1 ? nelements : (id + 1) * (nelements / nthreads);
							
			((d_othread*)args)[id] = (d_othread) {
				.result = result,
				.start = start,
				.end = end,
				.matrix = argv.args.operation.matrix,
			};
			start = end;
		}
		}
	else if(method == IMTHREAD){
		args = (d_imthread*)malloc(sizeof(d_imthread)*nthreads);
		incre = sizeof(d_imthread);		
		
		for(int id=0; id < nthreads; id++){
			end = id == nthreads - 1 ? nwidth : (id + 1) * (nwidth / nthreads);
							
			((d_imthread*)args)[id] = (d_imthread) {
				.result = result,
				.start = start,
				.end = end,
			};
			start = end;
		}
	}
	else if(method == UMTHREAD){
		args = (d_umthread*)malloc(sizeof(d_umthread)*nthreads);
		incre = sizeof(d_umthread);
		for(int id=0; id < nthreads; id++){
			end = id == nthreads - 1 ? nelements : (id + 1) * (nelements / nthreads);
							
			((d_umthread*)args)[id] = (d_umthread) {
				.result = result,
				.start = start,
				.end = end,
				.value = argv.args.uniform.value,
			};
			start = end;
		}
	}
	else if(method == SMTHREAD){
		args = (void*)malloc(sizeof(d_smthread)*nthreads);
		incre = sizeof(d_smthread);
		for(int id=0; id < nthreads; id++){
			end = id == nthreads - 1 ? nelements : (id + 1) * (nelements / nthreads);
							
			((d_smthread*)args)[id] = (d_smthread) {
				.result = result,
				.start = start,
				.end = end,
				.initial = argv.args.sequence.initial,
				.step = argv.args.sequence.step,
			};
			start = end;
		}
	}else{
		return;
	}
	
	// launch threads
	for (int i = 0; i < nthreads; i++) {
		pthread_create(thread_ids + i, NULL, funcptr, args+(incre*i) );
	}

	// wait for threads to finish
	for (size_t i = 0; i < nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}
	//if(method == MMULTHREAD) free(argv.args.matrix.matrix_b);
	free(args);
	
	return;
}


////////////////////////////////
///    THREADING WORKERS     ///
////////////////////////////////
/**
 *	Identity Matrix Thread Process
 */
void* identity_thread(void* argv){
	d_imthread* data = (d_imthread*) argv;
	
	const int start = data->start;
	const int end = data->end;
	float* result = data->result;
	const int nwidth = g_width;
		
	for(int i = start; i < end; i++){
		result[i * nwidth + i] = 1.0;
	}
	
	return NULL;
}

/**
 *	Uniform Matrix Thread Process
 */
void* uniform_thread(void* argv){
	d_umthread* data = (d_umthread*) argv;
	
	const int start = data->start;
	const int end = data->end;
	float* result = data->result;
	
	const float value = data->value;
	
	for(int i = start; i < end; i++){
		result[i] = value;
	}
	
	return NULL;
}

/**
 *	Sequence Matrix Thread Process
 */
void* sequence_thread(void* argv){
	d_smthread* data = (d_smthread*) argv;
	
	const int start = data->start;
	const int end = data->end;
	float* result = data->result;
	
	const float initial = data->initial;
	const float step = data->step;
	
	for(int i = start; i < end; i++){
		result[i] = initial + (step*i);
	}
	
	return NULL;
}

/**
 *	Scalar Multiply Thread Process
 */
void* scalar_mul_thread(void* argv){
	d_sthread* data = (d_sthread*) argv;
	
	const int start = data->start;
	const int end = data->end;
	float* result = data->result;
	const float* const matrix = data->matrix;
	
	const float scalar = data->value;
		
	for(int i = start; i < end; i++){
		result[i] = matrix[i]*scalar;
	}
	
	return NULL;
}

/**
 *	Scalar Add Thread Process
 */
void* scalar_add_thread(void* argv){
	d_sthread* data = (d_sthread*) argv;
	
	const int start = data->start;
	const int end = data->end;
	float* result = data->result;
	const float* const matrix = data->matrix;
	
	const float scalar = data->value;
		
	for(int i = start; i < end; i++){
		result[i] = matrix[i]+scalar;
	}
	
	return NULL;
}

/**
 *	Matrix Multiply Thread Process
 */
void* matrix_mul_thread(void* argv){
	d_mthread* data = (d_mthread*) argv;
	
	const int start = data->start;
	const int end = data->end;
	const int nwidth = g_width;
	
	const float* matrix_a = data->matrix_a;
	const float* matrix_b = data->matrix_b;
	float* result = data->result;
	
	float sum = 0;
	
	for(int i=start; i < end; i++){
		for(int k=0; k < nwidth; k++){
			sum = 0;
			for(int j=0; j < nwidth; j++){
				sum += matrix_a[i * nwidth + j]*matrix_b[k * nwidth + j];
			}
			result[i * nwidth + k] = sum;
		}
	}
	
	return NULL;
}

/**
 *	Matrix Add Thread Process
 */
void* matrix_add_thread(void* argv){
	d_mthread* data = (d_mthread*) argv;
	
	const int start = data->start;
	const int end = data->end;
	
	const float* matrix_a = data->matrix_a;
	const float* matrix_b = data->matrix_b;
	float* result = data->result;
		
	for(int i=start; i < end; i++){
		result[i] = matrix_a[i] + matrix_b[i];
	}
	
	return NULL;
}

/**
 *	Get Sum Thread Process  TODO FIX
 */



////////////////////////////////
///     UTILITY FUNCTIONS    ///
////////////////////////////////
/**
 *	Description:	Compare function for qsort
 *	Return:			difference between two ints
 * 	Source:			http://www.tutorialspoint.com/c_standard_library/c_function_qsort.htm
 */
int sortcmp(const void * a, const void * b){
	return ((int)( *(float*)a - *(float*)b ));
}


/**
 * Sets the number of threads available.
 */
void set_nthreads(ssize_t count) {

	g_nthreads = count;
}

/**
 * Sets the dimensions of the matrix.
 */
void set_dimensions(ssize_t order) {

	g_width = order;
	g_elements = order * order;
}

/**
 * Displays given matrix.
 */
void display(const float* matrix) {
	const int nwidth = g_width;
	
	for (ssize_t y = 0; y < nwidth; y++) {
		for (ssize_t x = 0; x < nwidth; x++) {
			if (x > 0) printf(" ");
			printf("%.2f", matrix[y * nwidth + x]);
		}

		printf("\n");
	}
}

/**
 * Displays given matrix row.
 */
void display_row(const float* matrix, ssize_t row) {
	const int nwidth = g_width;
	
	for (ssize_t x = 0; x < nwidth; x++) {
		if (x > 0) printf(" ");
		printf("%.2f", matrix[row * nwidth + x]);
	}

	printf("\n");
}

/**
 * Displays given matrix column.
 */
void display_column(const float* matrix, ssize_t column) {
	const int nwidth = g_width;
	
	for (ssize_t i = 0; i < nwidth; i++) {
		printf("%.2f\n", matrix[i * nwidth + column]);
	}
}

/**
 * Displays the value stored at the given element index.
 */
void display_element(const float* matrix, ssize_t row, ssize_t column) {
	
	printf("%.2f\n", matrix[row * g_width + column]);
}


////////////////////////////////
///   MATRIX INITALISATIONS  ///
////////////////////////////////

/**
 * Returns new matrix with all elements set to zero.
 */
float* new_matrix(void) {
	return calloc(g_elements+CACHE_SIZE, sizeof(float));
}

/**
 * Returns new matrix with all elements not set.
 */
float* empty_matrix(void) {
	return malloc((g_elements+CACHE_SIZE)*sizeof(float));
}

/**
 * Returns new identity matrix.
 */
float* identity_matrix(void) {
	/*
		1 0
		0 1
	*/
	float* result = new_matrix();
		
	result[CACHE_TYPE] = M_IDENTITY;
	
	if(g_width > OPTIMIAL_THREAD && g_nthreads > 1){
		void* (*functionPtr)(void*);
		functionPtr = &identity_thread;
		thread_args data = (thread_args){
				.result = result,
				.type = IMTHREAD,
				};
		
		spawn_threads(functionPtr, data);
		
	}else{
		for(int i = 0; i < g_width; i++){
			result[i * g_width + i] = 1.0;
		}
	}
	
	return result;
}


/**
 * Returns new matrix with elements generated at random using given seed. DONE!!
 */
float* random_matrix(int seed) {

	float* matrix = empty_matrix();

	int rand = seed;
	
	matrix[CACHE_TYPE] = M_RANDOM;
	
	for (ssize_t i = 0; i < g_elements; i++) {
		rand = (214013 * rand + 2531011);
		matrix[i] = (rand >> 16) & 0x7FFF;
	}
	
	return matrix;
}

/**
 * Returns new matrix with all elements set to given value. DONE!!
 */
float* uniform_matrix(float value) {
	/*
		     1 1
		1 => 1 1
	*/
	float* result = empty_matrix();
	const int nelements = g_elements;
	
	result[CACHE_TYPE] = M_UNIFORM;
	
	if(g_width > OPTIMIAL_THREAD && g_nthreads > 1){
		void* (*functionPtr)(void*);
		functionPtr = &uniform_thread;
		thread_args data = (thread_args){
				.result = result,
				.type = UMTHREAD,
				.args.uniform.value = value,
				};
		
		spawn_threads(functionPtr, data);
	}else{
		for(int i = 0; i < nelements; i++){
			result[i] = value;
		}
	}
	
	return result;
}

/**
 * Returns new matrix with elements in sequence from given start and step DONE!!
 */
float* sequence_matrix(float start, float step) {
	/*
		       1 2
		1 1 => 3 4
	*/
	float* result = empty_matrix();
	const int nelements = g_elements;
	
	result[CACHE_TYPE] = M_SEQUENCE;
	
	if(g_width > OPTIMIAL_THREAD && g_nthreads > 1){
		void* (*functionPtr)(void*);
		functionPtr = &sequence_thread;
		thread_args data = (thread_args) {
				.result = result,
				.type = SMTHREAD,
				.args.sequence.initial = start,
				.args.sequence.step = step,
				};
		spawn_threads(functionPtr, data);
	}else{
		for(int i = 0; i < nelements; i++){
			result[i] = start+(step*i);
		}
	}
	
	return result;
}


////////////////////////////////
///     MATRIX OPERATIONS    ///
////////////////////////////////

/**
 * Returns new matrix with elements cloned from given matrix. DONE!!
 */
float* cloned(const float* matrix) {

	float* result = empty_matrix();
	
	memcpy(result, matrix, sizeof(float)*(g_elements+CACHE_SIZE));

	return result;
}


/**
 * Returns new matrix with elements in ascending order.
 */
float* sorted(const float* matrix) {
	/*
		3 4    1 2
		2 1 => 3 4

	*/
	float* result = cloned(matrix);
	// clone and sort clone
	if(matrix[CACHE_TYPE] == M_UNIFORM || matrix[CACHE_TYPE] == M_SORTED){
		return result;
	}else{
		result[CACHE_TYPE] = M_SORTED;
		const int nelements = g_elements;
		MergeSort(result, nelements);
		//qsort(result, nelements, sizeof(float), sortcmp);
		return result;
	}
}

/**
 *  Merge Sort sourced from:
 *	https://gist.github.com/mycodeschool/9678029
 */
// Function to Merge Arrays L and R into A.  
// lefCount = number of elements in L 
// rightCount = number of elements in R.  
void Merge(float *A,float *L,int leftCount,float *R,int rightCount) { 
	int i,j,k; 
 
 	// i - to mark the index of left aubarray (L) 
 	// j - to mark the index of right sub-raay (R) 
 	// k - to mark the index of merged subarray (A) 
 	i = 0; j = 0; k =0; 
 
 	while(i<leftCount && j< rightCount) { 
 		if(L[i]  < R[j]) A[k++] = L[i++]; 
 		else A[k++] = R[j++]; 
 	} 
 	while(i < leftCount) A[k++] = L[i++]; 
 	while(j < rightCount) A[k++] = R[j++]; 
 } 
 
 
 // Recursive function to sort an array of integers.  
 void MergeSort(float* A,int n) { 
 	int mid;
	float *L, *R;
	const int nthreads = g_nthreads;
 	if(n < 2) return; // base condition. If the array has less than two element, do nothing.  
 
 	mid = n/2;  // find the mid index.  
 
	// create left and right subarrays 
 	// mid elements (from index 0 till mid-1) should be part of left sub-array  
 	// and (n-mid) elements (from mid to n-1) will be part of right sub-array 
 	L = (float*)malloc(mid*sizeof(float));  
 	R = (float*)malloc((n- mid)*sizeof(float));
	
	pthread_t thread_ids[nthreads];
	quicksort_type args[nthreads];
	
	void* (*functionPtr)(void*);
	functionPtr = &parallel_qsort;

	args[0] = (quicksort_type) {
		.cmatrix = A,
		.matrix = L,
		.n = mid,
		.start = 0,
		.end = mid,
	};
	
	args[1] = (quicksort_type) {
		.cmatrix = A,
		.matrix = R,
		.n = n-mid,
		.start = mid,
		.end = n,
	};
	printf("ready to spawn threads\n");
	// launch threads
	for (int i = 0; i < 2; i++) {
		pthread_create(thread_ids + i, NULL, functionPtr, args+i );
	}
	
	// wait for threads to finish
	for (size_t i = 0; i < 2; i++) {
		pthread_join(thread_ids[i], NULL);
	}
	printf("ready to merge\n");
 	//MergeSort(L,mid);  // sorting the left subarray 
 	//MergeSort(R,n-mid);  // sorting the right subarray

 	Merge(A,L,mid,R,n-mid);  // Merging L and R into A as sorted list. 
    free(L); 
    free(R); 
} 

// custom function to call qsort
void* parallel_qsort(void* args){
	quicksort_type* data = (quicksort_type*) args;
	
	const float* A = data->cmatrix;
	float* D = data->matrix;
	const int size = data->n;
	const int start = data->start;
	const int end = data->end;
	
	for(int i = start;i<end;i++) D[i] = A[i]; // creating left subarray
	printf("sub array created.\n");
	qsort(D, size, sizeof(float), sortcmp);
	
	return NULL;
}


/**
 * Returns new matrix with elements rotated 90 degrees clockwise.
 */
float* rotated(const float* matrix) {
	/*
		1 2    3 1
		3 4 => 4 2
		
		1 2 3	 7 4 1
		4 5 6    8 5 2
		7 8 9 => 9 6 3
		
	*/
	float* result = empty_matrix();
	result[CACHE_TYPE] = M_RANDOM;
	const int nwidth = g_width;
	
	for(int row=0; row < nwidth; row++){
		for(int col=0; col < nwidth; col++){
			result[row*nwidth+col] = matrix[(nwidth-col-1)*nwidth+(row)];
		}
	}
	
	return result;
}

/**
 * Returns new matrix with elements ordered in reverse.  DONE!!
 */
float* reversed(const float* matrix) {
	/*
		1 2    4 3
		3 4 => 2 1
	*/
	float* result = empty_matrix();
	result[CACHE_TYPE] = M_RANDOM;
	const int nelements = g_elements;
	const int last = nelements - 1;
	
	for(int i = 0; i < nelements; i++){
		result[i] = matrix[last-i];
	}

	return result;
}

/**
 * Returns new transposed matrix.
 */
float* transposed(const float* matrix) {
	/*
		1 2    1 3
		3 4 => 2 4
	*/
	float* result = empty_matrix();
	result[CACHE_TYPE] = M_RANDOM;
	const int nwidth = g_width;
	
	for(int row=0; row < nwidth; row++){
		for(int col=0; col < nwidth; col++){
			result[row*nwidth+col] = matrix[col*nwidth+row];
		}
	}

	return result;
}

/**
 * Returns new matrix with scalar added to each element.
 */
float* scalar_add(const float* matrix, float scalar) {
	/*
		1 0        2 1
		0 1 + 1 => 1 2

		1 2        5 6
		3 4 + 4 => 7 8
	*/
	const int nelements = g_elements;
	if(scalar == 0) return cloned(matrix);
	else{
		float* result = empty_matrix();
		result[CACHE_TYPE] = matrix[CACHE_TYPE];
		
		if(g_width > OPTIMIAL_THREAD && g_nthreads > 1){
			void* (*functionPtr)(void*);
			functionPtr = &scalar_add_thread;
			thread_args data = (thread_args){
					.result = result,
					.type = STHREAD,
					.args.scalar.scalar = scalar,
					.args.scalar.matrix = matrix,
					};
			
			spawn_threads(functionPtr, data);
		}else{
			for(int i = 0; i < nelements; i++){
				result[i] = matrix[i] + scalar;
			}
		}

		return result;
	}
}

/**
 * Returns new matrix with scalar multiplied to each element.
 */
float* scalar_mul(const float* matrix, float scalar) {	
	/*
		1 0        2 0
		0 1 x 2 => 0 2

		1 2        2 4
		3 4 x 2 => 6 8
	*/
	const int nelements = g_elements;
	if(scalar == 0) return cloned(matrix);
	else{
		float* result = empty_matrix();
		result[CACHE_TYPE] = matrix[CACHE_TYPE];
		if(g_width > OPTIMIAL_THREAD && g_nthreads > 1){
			void* (*functionPtr)(void*);
			functionPtr = &scalar_mul_thread;
			thread_args data = (thread_args){
					.result = result,
					.type = STHREAD,
					.args.scalar.scalar = scalar,
					.args.scalar.matrix = matrix,
					};
			
			spawn_threads(functionPtr, data);
		}else{
			for(int i = 0; i < nelements; i++){
				result[i] = matrix[i] * scalar;
			}
		}
		
		return result;
	}
}


/**
 * Returns new matrix that is the result of
 * adding the two given matrices together.
 */
float* matrix_add(const float* matrix_a, const float* matrix_b) {
	/*
		1 0   0 1    1 1
		0 1 + 1 0 => 1 1

		1 2   4 4    5 6
		3 4 + 4 4 => 7 8
	*/
	float* result = empty_matrix();
	result[CACHE_TYPE] = M_RANDOM;
	const int nelements = g_elements;
	
	if(g_width > OPTIMIAL_THREAD-10 && g_nthreads > 1){
		void* (*functionPtr)(void*);
		functionPtr = &matrix_add_thread;
		thread_args data = (thread_args){
				.result = result,
				.type = MADDTHREAD,
				.args.matrix.matrix_a = matrix_a,
				.args.matrix.matrix_b = (float *)matrix_b,
				};
		
		spawn_threads(functionPtr, data);
	}else{
		
		for(int i = 0; i < nelements; i++){
			result[i] = matrix_a[i] + matrix_b[i];
		}
	}
	

	return result;
}


/**
 * Returns new matrix that is the result of
 * multiplying the two matrices together.
 */
float* matrix_mul(const float* matrix_a, const float* matrix_b) {
	/*
		1 2   1 0    1 2
		3 4 x 0 1 => 3 4

		1 2   5 6    19 22
		3 4 x 7 8 => 43 50
	*/
	if(matrix_b[CACHE_TYPE] == M_IDENTITY) return cloned(matrix_a);
	
	const int nwidth = g_width;
	
	float* result = empty_matrix();
	result[CACHE_TYPE] = M_RANDOM;
	if(nwidth > OPTIMIAL_THREAD-10 && g_nthreads > 1){
		void* (*functionPtr)(void*);
		functionPtr = &matrix_mul_thread;
		float* transpose = transposed(matrix_b);
		thread_args data = (thread_args){
				.result = result,
				.type = MMULTHREAD,
				.args.matrix.matrix_a = matrix_a,
				.args.matrix.matrix_b = transpose,
				};
		spawn_threads(functionPtr, data);
		free(transpose);
	}else{
		// very slow method
		float sum;
		float* transpose = transposed(matrix_b);
		for(int i=0; i < nwidth; i++){
			for(int k=0; k < nwidth; k++){
				sum = 0;
				for(int j=0; j < nwidth; j++){
					sum += matrix_a[i * nwidth + j]*transpose[k * nwidth + j];
				}
				result[i * nwidth + k] = sum;
			}
		}
		free(transpose);
	}

	return result;
}

/**
 * Returns new matrix that is the result of
 * powering the given matrix to the exponent.
 */
float* matrix_pow(const float* matrix, int exponent) {

	float* result = NULL;
	float* temp;

	/*
		1 2        1 0
		3 4 ^ 0 => 0 1

		1 2        1 2
		3 4 ^ 1 => 3 4

		1 2        199 290
		3 4 ^ 4 => 435 634
	*/
		
	if(exponent == 0){
		return identity_matrix();
	}else if(exponent == 1){
		result = cloned(matrix);
		return result;
	}else if(exponent > 1){
		result = cloned(matrix);
		for(int i=1; i < exponent; i++){
			temp = matrix_mul(result, matrix);
			free(result);
			result = temp;
		}
		result[CACHE_TYPE] = M_RANDOM;
	}

	return result;
}


/**
 * Returns new matrix that is the result of
 * convolving given matrix with a 3x3 kernel matrix.
 */
float* matrix_conv(const float* matrix, const float* kernel) {
	/*
		Convolution is the process in which the values of a matrix are
		computed according to the weighted sum of each value and it's
		neighbours, where the weights are given by the kernel matrix.
	
		1  2  3  4				-4 -2 -1  1
		5  6  7  8				 4  6  7  9
		9  10 11 12				 8  10 11 13
		13 14 15 16 :: sharpen =>  16 18 19 21
		
	*/
	float* result = new_matrix();
	result[CACHE_TYPE] = M_RANDOM;
	
	float sum = 0;
	const int width_kernel = 3;
	const int nwidth = g_width;
	int offset_col = 0;
	int offset_row = 0;
	
	for(int row=0; row < nwidth; row++){
		// traverse the array one element at a time.
		for(int col=0; col < nwidth; col++){
			// start working out the convolution of one element,
			//   moving along the kernel one at a time.
			//  Kernel starts in it's top corner.
						
			sum = 0;
			for(int c_row=0; c_row < width_kernel; c_row++){
				for(int c_col=0; c_col < width_kernel; c_col++){
					// account for padding / over hang
					// account for fact that kernel needs to be over the centre of the matrix

					offset_col = 0;
					offset_row = 0;
					if( row == 0 && c_row == 0) offset_row = 1;  // whole top row
					if( row == nwidth-1 && c_row == 2) offset_row = -1; // whole bottom row
					
					if( col == 0 && c_col == 0) offset_col = 1; //whole left side
					if( col == nwidth-1 && c_col == 2) offset_col = -1; // whole right side					

					// reference (centre):  matrix[row*nwidth+col];
					sum += kernel[c_row * width_kernel + c_col] * matrix[(row+c_row-1+offset_row) * nwidth + (col+c_col-1+offset_col)];
				}
				result[row * nwidth + col] = sum;
			}
		}
	}
	
	return result;
}

////////////////////////////////
///       COMPUTATIONS       ///
////////////////////////////////

/**
 * Returns the sum of all elements.
 */
float get_sum(const float* matrix) {

	/*
		2 1
		1 2 => 6

		1 1
		1 1 => 4
	*/
	const int nelements = g_elements;
	if(matrix[nelements] == M_IDENTITY) return g_width;
	else if(matrix[nelements] == M_UNIFORM) return matrix[0]*nelements;
	else if(matrix[nelements] == M_SEQUENCE) return ((nelements/2.0)*(matrix[0]+matrix[nelements-1]));
	else{
		float sum = 0;
		for(int i = 0; i < nelements; i++){
			sum += matrix[i];
		}
		return sum;
	}
}

/**
 * Returns the trace of the matrix.
 */
float get_trace(const float* matrix) {

	/*
		1 0
		0 1 => 2

		2 1
		1 2 => 4
	*/
	const int nwidth = g_width;
	if(matrix[CACHE_TYPE] == M_IDENTITY) return nwidth;
	else if(matrix[CACHE_TYPE] == M_UNIFORM) return matrix[0]*nwidth;
	else if(nwidth == 1) return matrix[0];
	else{
		float sum = 0;
		for(int i = 0; i < nwidth; i++){
			sum += matrix[i * nwidth + i];
		}
		return sum;
	}
}

/**
 * Returns the smallest value in the matrix.
 */
float get_minimum(const float* matrix) {

	/*
		1 2
		3 4 => 1

		4 3
		2 1 => 1
	*/
	const int nelements = g_elements;
	if(matrix[CACHE_TYPE] == M_IDENTITY) return 0.0;
	else if(matrix[CACHE_TYPE] == M_UNIFORM) return matrix[0];
	else if(matrix[CACHE_TYPE] == M_SEQUENCE) return matrix[0];
	else if(matrix[CACHE_TYPE] == M_SORTED) return matrix[0];
	else{
		float min = matrix[0];
		for(int i = 0; i < nelements; i++){
			if(matrix[i]<min){
				min = matrix[i];
			}
		}
		return min;
	}	
}

/**
 * Returns the largest value in the matrix.
 */
float get_maximum(const float* matrix) {

	/*
		1 2
		3 4 => 4

		4 3
		2 1 => 4
	*/
	const int nelements = g_elements;
	if(matrix[CACHE_TYPE] == M_IDENTITY) return 1.0;
	else if(matrix[CACHE_TYPE] == M_UNIFORM) return matrix[0];
	else if(matrix[CACHE_TYPE] == M_SEQUENCE) return matrix[g_elements-1];
	else if(matrix[CACHE_TYPE] == M_SORTED) return matrix[g_elements-1];
	else{
		float max = matrix[0];
		for(int i = 0; i < nelements; i++){
			if(matrix[i]>max){
				max = matrix[i];
			}
		}
		return max;
	}	
}


/**
 * Displays given matrix.  TODO: REMOVE
 */
void display_c(const float* matrix, int width) {
	const int nwidth = g_width;
	for (int y = 0; y < nwidth; y++) {
		for (int x = 0; x < nwidth; x++) {
			if (x > 0) printf(" ");
			printf("%.2f", matrix[y * nwidth + x]);
		}

		printf("\n");
	}
}


/**
 *  Builds a matrix for the determinant function
 */
float* build_matrix(const float* matrix, const int crow, const int width){
	if(width == 0){
		return NULL;
	}

	float* result = malloc(sizeof(float)*(width*width));

	if(width == 1){
		result[0*width+0] = matrix[0*width+0];
		//display_c(result, width);
		return result;
	}

	// columns before...
	int offset = 0;
	for(int row = 0; row < width; row++){
		for(int col = 0; col < width; col++){
			if(col == crow){
				// when we hit the column of the current element
				offset=1;
			}
			result[(row) * width + (col)] = matrix[(row+1) * (width+1) + (col+offset)];
		}
		offset = 0;
	}
	return result;
}


/**
 *	Recursive method for finding determinants greater than n>2
 */

float determinant_calc(const float* matrix, int width){
	if(width == 3){
		// do not call recurrsion again...
		return ((matrix[0*width+0]*((matrix[1*width+1]*matrix[2*width+2])-(matrix[2*width+1]*matrix[1*width+2])))
				- (matrix[0*width+1]*((matrix[1*width+0]*matrix[2*width+2])-(matrix[2*width+0]*matrix[1*width+2])))
				+ (matrix[0*width+2]*((matrix[1*width+0]*matrix[2*width+1])-(matrix[2*width+0]*matrix[1*width+1]))));
	}else if(width == 2){
		// do not call recurrsion again...
		return (matrix[0*width+0]*matrix[1*width+1])
				-(matrix[0*width+1]*matrix[1*width+0]);
	}else{
		float determinant = 0;
		float* smatrix = NULL;
		float result;
		float element;
		for(int i = 0; i < width; i++){
			smatrix = build_matrix(matrix, i, width - 1);
			result = (determinant_calc(smatrix, width - 1));
			element = (pow(-1,i))*(matrix[0*width+i]);
			determinant += element*result;
			//printf("POWER:  %f  || DETA: %f  || result: %f  || element: %f\n", pow(-1,i), determinant, result, element);
			free(smatrix);
		}
		return determinant;
	}
}

/**
 * Returns the determinant of the matrix.
 */
float get_determinant(const float* matrix) {

	/*
		1 0
		0 1 => 1

		1 2
		3 4 => -2

		8 0 2
		0 4 0
		2 0 8 => 240
	*/
	const int nwidth = g_width;
	//det search for zero on column, go along column with zero
	if(nwidth == 1){
		return matrix[0];
	}else if(nwidth == 2){
		return ((matrix[0*nwidth+0]*matrix[1*nwidth+1])-(matrix[0*nwidth+1]*matrix[1*nwidth+0]));
	}else{
		return determinant_calc(matrix, nwidth);
	}
}

/**
 * Returns the frequency of the given value in the matrix.
 */
ssize_t get_frequency(const float* matrix, float value) {

	/*
		1 1
		1 1 :: 1 => 4

		1 0
		0 1 :: 2 => 0
	*/
	ssize_t freq = 0;
	const int nelements = g_elements;
	for(int i = 0; i < nelements; i++){
		if(matrix[i]==value){
			freq++;
		}
	}

	return freq;
}