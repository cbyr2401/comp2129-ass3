#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>

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
 * Returns new identity matrix. DONE!!
 */
float* identity_matrix(void) {

	float* result = new_matrix();

	/*
		1 0
		0 1
	*/
	int row = 0;
	for(int i = 0; row < g_width; i++){
		result[row * g_width + i] = 1.0;
		row++;
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

	float* result = new_matrix();

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

	float* result = new_matrix();

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

	float* result = new_matrix();

	for (ssize_t y = 0; y < g_height; y++) {
		for (ssize_t x = 0; x < g_width; x++) {
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

	float* result = new_matrix();

	/*
		TODO

		1 2    3 1
		3 4 => 4 2
	*/

	return result;
}

/**
 * Returns new matrix with elements ordered in reverse.  DONE!!
 */
float* reversed(const float* matrix) {

	float* result = new_matrix();

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

	float* result = new_matrix();

	/*
		TODO

		1 2    1 3
		3 4 => 2 4
	*/

	return result;
}

/**
 * Returns new matrix with scalar added to each element.
 */
float* scalar_add(const float* matrix, float scalar) {

	float* result = new_matrix();

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

	float* result = new_matrix();

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

	float* result = new_matrix();

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

/**
 * Returns new matrix that is the result of
 * multiplying the two matrices together.
 */
float* matrix_mul(const float* matrix_a, const float* matrix_b) {

	float* result = new_matrix();

	/*
		TODO

		1 2   1 0    1 2
		3 4 x 0 1 => 3 4

		1 2   5 6    19 22
		3 4 x 7 8 => 43 50
	*/
	ssize_t sum;

	for(int i=0; i < g_width; i++){
		for(int j=0; j < g_width; j++){
			sum = 0;
			for(int k = 0; k < g_width; k++){
				sum = sum*(matrix_a[k * g_width + i]*matrix_b[j * g_width + k]);
			}
			result[j * g_width + i] = sum;
		}
	}

	return result;
}

/**
 * Returns new matrix that is the result of
 * powering the given matrix to the exponent.
 */
float* matrix_pow(const float* matrix, float exponent) {

	float* result = new_matrix();

	/*
		TODO

		1 2        1 0
		3 4 ^ 0 => 0 1

		1 2        1 2
		3 4 ^ 1 => 3 4

		1 2        199 290
		3 4 ^ 4 => 435 634
	*/

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
		TODO

		2 1
		1 2 => 6

		1 1
		1 1 => 4
	*/

	return 0;
}

/**
 * Returns the trace of the matrix.
 */
float get_trace(const float* matrix) {

	/*
		TODO

		1 0
		0 1 => 2

		2 1
		1 2 => 4
	*/

	return 0;
}

/**
 * Returns the smallest value in the matrix.
 */
float get_minimum(const float* matrix) {

	/*
		TODO

		1 2
		3 4 => 1

		4 3
		2 1 => 1
	*/

	return 0;
}

/**
 * Returns the largest value in the matrix.
 */
float get_maximum(const float* matrix) {

	/*
		TODO

		1 2
		3 4 => 4

		4 3
		2 1 => 4
	*/

	return 0;
}

/**
 * Returns the determinant of the matrix.
 */
float get_determinant(const float* matrix) {

	/*
		TODO

		1 0
		0 1 => 1

		1 2
		3 4 => -2

		8 0 2
		0 4 0
		2 0 8 => 240
	*/

	return 0;
}

/**
 * Returns the frequency of the given value in the matrix.
 */
ssize_t get_frequency(const float* matrix, float value) {

	/*
		TODO

		1 1
		1 1 :: 1 => 4

		1 0
		0 1 :: 2 => 0
	*/

	return 0;
}