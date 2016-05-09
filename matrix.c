#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include <math.h>

#include "matrix.h"

static int g_seed = 0;

static ssize_t g_width = 0;
static ssize_t g_height = 0;
static ssize_t g_elements = 0;

static ssize_t g_nthreads = 1;

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
 * Returns pseudorandom number determined by the seed.
 */
int fast_rand(void) {

	g_seed = (214013 * g_seed + 2531011);
	return (g_seed >> 16) & 0x7FFF;
}

/**
 * Sets the seed used when generating pseudorandom numbers.
 */
void set_seed(int seed) {

	g_seed = seed;
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
	g_height = order;

	g_elements = g_width * g_height;
}

/**
 * Displays given matrix.
 */
void display(const float* matrix) {

	for (ssize_t y = 0; y < g_height; y++) {
		for (ssize_t x = 0; x < g_width; x++) {
			if (x > 0) printf(" ");
			printf("%.2f", matrix[y * g_width + x]);
		}

		printf("\n");
	}
}

/**
 * Displays given matrix row.
 */
void display_row(const float* matrix, ssize_t row) {

	for (ssize_t x = 0; x < g_width; x++) {
		if (x > 0) printf(" ");
		printf("%.2f", matrix[row * g_width + x]);
	}

	printf("\n");
}

/**
 * Displays given matrix column.
 */
void display_column(const float* matrix, ssize_t column) {

	for (ssize_t i = 0; i < g_height; i++) {
		printf("%.2f\n", matrix[i * g_width + column]);
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

	return calloc(g_elements, sizeof(float));
}

/**
 * Returns new matrix with all elements not set.
 */
float* empty_matrix(void) {

	return malloc(g_elements*sizeof(float));
}

/**
 * Returns new identity matrix. DONE!!
 */
float* identity_matrix(void) {

	float* result = new_matrix();

	/*
		1 0
		0 1
	*/

	for(int i = 0; i < g_width; i++){
		result[i * g_width + i] = 1.0;
	}
	return result;
}

/**
 * Returns new matrix with elements generated at random using given seed. DONE!!
 */
float* random_matrix(int seed) {

	float* matrix = new_matrix();

	set_seed(seed);

	for (ssize_t i = 0; i < g_elements; i++) {
		matrix[i] = fast_rand();
	}

	return matrix;
}

/**
 * Returns new matrix with all elements set to given value. DONE!!
 */
float* uniform_matrix(float value) {

	float* result = empty_matrix();

	/*
		     1 1
		1 => 1 1
	*/
	for(int i = 0; i < g_elements; i++){
		result[i] = value;
	}
	return result;
}

/**
 * Returns new matrix with elements in sequence from given start and step DONE!!
 */
float* sequence_matrix(float start, float step) {

	float* result = empty_matrix();

	/*
		       1 2
		1 1 => 3 4
	*/
	for(int i = 0; i < g_elements; i++){
		result[i] = start+(step*i);
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

	for (int y = 0; y < g_height; y++) {
		for (int x = 0; x < g_width; x++) {
			result[y * g_width + x] = matrix[y * g_width + x];
		}
	}

	return result;
}

/**
 * Returns new matrix with elements in ascending order. DONE!!
 */
float* sorted(const float* matrix) {

	float* result = cloned(matrix);

	/*
		3 4    1 2
		2 1 => 3 4

	*/
	// clone and sort clone
	qsort(result, g_elements, sizeof(float), sortcmp);

	return result;
}


/**
 * Returns new matrix with elements rotated 90 degrees clockwise.
 */
float* rotated(const float* matrix) {

	float* result = empty_matrix();

	/*
		1 2    3 1
		3 4 => 4 2
		
		1 2 3	 7 4 1
		4 5 6    8 5 2
		7 8 9 => 9 6 3
		
	*/
	for(int row=0; row < g_width; row++){
		for(int col=0; col < g_width; col++){
			result[row*g_width+col] = matrix[(g_width-col-1)*g_width+(row)];
		}
	}
	
	return result;
}

/**
 * Returns new matrix with elements ordered in reverse.  DONE!!
 */
float* reversed(const float* matrix) {

	float* result = empty_matrix();

	/*
		1 2    4 3
		3 4 => 2 1
	*/
	int last = g_elements - 1;
	for(int i = 0; i < g_elements; i++){
		result[i] = matrix[last-i];
	}

	return result;
}

/**
 * Returns new transposed matrix.
 */
float* transposed(const float* matrix) {

	float* result = empty_matrix();

	/*
		TODO

		1 2    1 3
		3 4 => 2 4
	*/
	
	for(int row=0; row < g_width; row++){
		for(int col=0; col < g_width; col++){
			result[row*g_width+col] = matrix[col*g_width+row];
		}
	}

	return result;
}

/**
 * Returns new matrix with scalar added to each element.
 */
float* scalar_add(const float* matrix, float scalar) {

	float* result = empty_matrix();

	/*
		1 0        2 1
		0 1 + 1 => 1 2

		1 2        5 6
		3 4 + 4 => 7 8
	*/
	for(int i = 0; i < g_elements; i++){
		result[i] = matrix[i] + scalar;
	}

	return result;
}

/**
 * Returns new matrix with scalar multiplied to each element.
 */
float* scalar_mul(const float* matrix, float scalar) {

	float* result = empty_matrix();

	/*
		1 0        2 0
		0 1 x 2 => 0 2

		1 2        2 4
		3 4 x 2 => 6 8
	*/
	for(int i = 0; i < g_elements; i++){
		result[i] = matrix[i] * scalar;
	}

	return result;
}

/**
 * Returns new matrix that is the result of
 * adding the two given matrices together.
 */
float* matrix_add(const float* matrix_a, const float* matrix_b) {

	float* result = empty_matrix();

	/*
		1 0   0 1    1 1
		0 1 + 1 0 => 1 1

		1 2   4 4    5 6
		3 4 + 4 4 => 7 8
	*/
	for(int i = 0; i < g_elements; i++){
		result[i] = matrix_a[i] + matrix_b[i];
	}

	return result;
}

int min(const int a, const int b){
	if(a<b) return b;
	return a;
}

/**
 * Returns new matrix that is the result of
 * multiplying the two matrices together.
 */
float* matrix_mul(const float* matrix_a, const float* matrix_b) {

	float* result = empty_matrix();

	/*
		TODO

		1 2   1 0    1 2
		3 4 x 0 1 => 3 4

		1 2   5 6    19 22
		3 4 x 7 8 => 43 50
	*/
	float sum;
	int tile_size = (int)sqrt(g_width);

	//second method - still very slow... fixes cache misses for row-major
	for(int I=0; I < g_width; I+=tile_size){
		for(int J=0; J < g_width; J+=tile_size){
			for(int K=0; K < g_width; K+=tile_size){
				for(int i=0; i < min(I+tile_size, g_width); i++){
					for(int j=0; j < min(J+tile_size, g_width); j++){
						sum = 0;
						for(int k = 0; k < min(K+tile_size, g_width); k++){
							//if(matrix_a[i * g_width + k] != 0 || matrix_b[k * g_width + j] != 0){
								sum = sum+(matrix_a[i * g_width + k]*matrix_b[k * g_width + j]);
							//}
						}
						result[i * g_width + j] = sum;
					}
				}
			}
		}
	}

	// very slow method
	// for(int i=0; i < g_width; i++){
		// for(int j=0; j < g_width; j++){
			// sum = 0;
			// for(int k=0; k < g_width; k++){
				// sum += (matrix_a[i * g_width + k]*matrix_b[k * g_width + j]);
			// }
			// result[i * g_width + j] = sum;
		// }
	// }

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
	}

	return result;
}


/**
 * Returns new matrix that is the result of
 * convolving given matrix with a 3x3 kernel matrix.
 */
float* matrix_conv(const float* matrix, const float* kernel) {

	float* result = new_matrix();

	/*
		TODO

		Convolution is the process in which the values of a matrix are
		computed according to the weighted sum of each value and it's
		neighbours, where the weights are given by the kernel matrix.
	*/

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
	float sum = 0;
	for(int i = 0; i < g_elements; i++){
		sum += matrix[i];
	}
	return sum;
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

	
	if(g_width == 1){
		return matrix[0];
	}else{
		float sum = 0;
		for(int i = 0; i < g_width; i++){
			sum += matrix[i * g_width + i];
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
	ssize_t min = matrix[0];
	for(int i = 0; i < g_elements; i++){
		if(matrix[i]<min){
			min = matrix[i];
		}
	}
	return min;
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

	ssize_t max = matrix[0];
	for(int i = 0; i < g_elements; i++){
		if(matrix[i]>max){
			max = matrix[i];
		}
	}
	return max;
}





/**
 * Displays given matrix.  TODO: REMOVE
 */
void display_c(const float* matrix, int width) {

	for (ssize_t y = 0; y < width; y++) {
		for (ssize_t x = 0; x < width; x++) {
			if (x > 0) printf(" ");
			printf("%.2f", matrix[y * width + x]);
		}

		printf("\n");
	}
}


/**
 *  Builds a matrix for the determinant function
 */
float* build_matrix(const float* matrix, int crow, int width){
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
	if(width == 2){
		return (matrix[0*width+0]*matrix[1*width+1])-(matrix[0*width+1]*matrix[1*width+0]); //pow(-1,power)*
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

	if(g_width == 1){
		return matrix[0];
	}else if(g_width == 2){
		return ((matrix[0*g_width+0]*matrix[1*g_width+1])-(matrix[0*g_width+1]*matrix[1*g_width+0]));
	}else{
		return determinant_calc(matrix, g_width);
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
	for(int i = 0; i < g_elements; i++){
		if(matrix[i]==value){
			freq++;
		}
	}

	return freq;
}