#ifndef MATRIX_H
#define MATRIX_H

#include <unistd.h>

// utility functions

int fast_rand(void);

void set_seed(int value);
void set_nthreads(ssize_t count);
void set_dimensions(ssize_t width);

void display(const float* matrix);
void display_row(const float* matrix, ssize_t row);
void display_column(const float* matrix, ssize_t column);
void display_element(const float* matrix, ssize_t row, ssize_t column);

// matrix operations

float* new_matrix(void);
float* identity_matrix(void);

float* random_matrix(int seed);
float* uniform_matrix(float value);
float* sequence_matrix(float start, float step);

float* cloned(const float* matrix);
float* sorted(const float* matrix);
float* rotated(const float* matrix);
float* reversed(const float* matrix);
float* transposed(const float* matrix);

float* scalar_add(const float* matrix, float scalar);
float* scalar_mul(const float* matrix, float scalar);
float* matrix_pow(const float* matrix, int exponent);
float* matrix_conv(const float* matrix, const float* kernel);
float* matrix_add(const float* matrix_a, const float* matrix_b);
float* matrix_mul(const float* matrix_a, const float* matrix_b);

// compute operations

float get_sum(const float* matrix);
float get_trace(const float* matrix);
float get_minimum(const float* matrix);
float get_maximum(const float* matrix);
float get_determinant(const float* matrix);
ssize_t get_frequency(const float* matrix, float value);

// custom function:
float* build_matrix(const float* matrix, int current_row, int width);
void display_c(const float* matrix, int width);
float determinant_calc(const float* matrix, int width);
int sortcmp(const void * a, const void * b);
int min(const int a, const int b);


typedef struct {
	float* result;
	int start;
	int end;
} d_imthread;

typedef struct {
	float* result;
	int start;
	int end;
	float value;
} d_umthread;

typedef struct {
	float* result;
	int start;
	int end;
	float initial;
	float step;
} d_smthread;

typedef struct {
	float* result;
	const float* matrix_a;
	const float* matrix_b;
	int start;
	int end;
} d_mthread;

typedef struct {
	float* result;
	const float* matrix;
	int start;
	int end;
} d_othread;

typedef struct {
	float* result;
	const float* matrix;
	float value;
	int start;
	int end;
} d_sthread;

typedef enum {
	MTHREAD, STHREAD, OTHREAD, IMTHREAD, UMTHREAD, SMTHREAD
} thread_type;

typedef struct {
	int partition;
	thread_type type;
	float* result;
	
	union{
		struct
		{
			const float* matrix_a;
			const float* matrix_b;
		} matrix;
		
		struct
		{
			const float* matrix;
			float scalar;
		} scalar;
		
		struct
		{
			const float* matrix;
			const float* kernel;
		} conv;
		
		struct
		{
			const float* matrix;
		} operation;
		
		struct
		{
			float initial;
			float step;
		} sequence;
		
		struct
		{
			float value;
		} uniform;
	} args;
} thread_args;


// thread structs (defined in typedef below)
// typedef struct d_mthread d_mthread; // matrix mul
// typedef struct d_sthread d_sthread; // scalar mul + add
// typedef struct d_othread d_othread; // operations
// typedef struct d_imthread d_imthread; // identity matrix
// typedef struct d_umthread d_umthread; // uniform matrix
// typedef struct d_smthread d_smthread; // sequence matrix
//typedef enum thread_data_types thread_type; // type of thread struct to use

//typedef struct thread_args thread_args;

void spawn_threads(void*(*funcptr)(void*), thread_args argv);
void* identity_thread(void* argv);
void* uniform_thread(void* argv);
void* sequence_thread(void* argv);
void* scalar_mul_thread(void* argv);


#endif