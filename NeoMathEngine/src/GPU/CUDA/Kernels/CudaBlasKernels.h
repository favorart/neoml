/* Copyright © 2017-2023 ABBYY

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#pragma once

#include <Kernels/CudaGrid.h>
#include <CudaCommon.h>
#include <Kernels/CudaReduce.h>

namespace NeoML {

__global__ void SetVectorToMatrixRowsKernel(float* result,
	int matrixHeight, int matrixWidth, const float* __restrict__ vector)
{
	int index;
	if( GetCudaTaskIndex( matrixHeight * matrixWidth, index ) ) {
		PRINT_HEAD2_F( index, 0, 0, "SetVectorToMatrixRowsKernel", vector, result, matrixHeight * matrixWidth );

		result[index] = vector[index % matrixWidth];
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

const int AddVectorToMatrixElementsCombine = 4;
__global__ void AddVectorToMatrixElementsKernel( float* matrix, int height, int width,
	const int* __restrict__ indices, const float* __restrict__ vector, size_t calls_counter, void* historyKernels )
{
	int jPos;
	int step;
	const int count = GetCudaTaskCountAndIndex( height, AddVectorToMatrixElementsCombine, jPos, step );
	PRINT_HEAD2_CNT_F( jPos, 0, 0, "AddVectorToMatrixElementsKernel1", vector, matrix,
		height, calls_counter, historyKernels, AddVectorToMatrixElementsKernel1Id );

	for( int i = 0; i < count; ++i ) {
		const int index = indices[jPos];
		if( index >= 0 && index < width ) {
			float result = matrix[jPos * width + index] += vector[jPos];

			WARN2_CNT_F( "AddVectorToMatrixElementsKernel1", vector[jPos], vector, result, matrix,
				height, width, index, calls_counter, historyKernels );
		}
		jPos += step;
	}
}

const int AddVectorToMatrixElementsMulCombine = 4;
__global__ void AddVectorToMatrixElementsKernel( float* __restrict__ matrix, int /*height*/, int width,
	const int* __restrict__ rowIndices, const int* __restrict__ columnIndices,
	const float* __restrict__ vector, int vectorSize, size_t calls_counter, void* historyKernels )
{
	int index;
	int step;
	const int count = GetCudaTaskCountAndIndex( vectorSize, AddVectorToMatrixElementsMulCombine, index, step );

	PRINT_HEAD2_CNT_F( index, 0, 0, "AddVectorToMatrixElementsKernel2", vector, matrix,
		vectorSize, calls_counter, historyKernels, AddVectorToMatrixElementsKernel2Id );
	for( int i = 0; i < count; ++i ) {
		float* result = matrix + rowIndices[index] * width + columnIndices[index];
		atomicAdd( result, vector[index] );

		WARN2_CNT_F( "AddVectorToMatrixElementsKernel2", vector[index], vector, *result, matrix,
			vectorSize, width, index, calls_counter, historyKernels );
		index += step;
	}
}

// Assigns the values matrix[rowIndices[i], columnIndices[i]] = vector[i].
const int SetVectorToMatrixElementsMulCombine = 4;
__global__ void SetVectorToMatrixElementsKernel(
	float* __restrict__ matrix, int /*height*/, int width,
	const int* __restrict__ rowIndices, const int* __restrict__ columnIndices,
	const float* __restrict__ vector, int vectorSize )
{
	int index;
	int step;
	const int count = GetCudaTaskCountAndIndex(
		vectorSize, SetVectorToMatrixElementsMulCombine, index, step );

	PRINT_HEAD2_F( index, 0, 0, "SetVectorToMatrixElementsKernel", vector, matrix, vectorSize );
	for( int i = 0; i < count; ++i ) {
		float result = matrix[rowIndices[index] * width + columnIndices[index]] = vector[index];
		assert( isfinite( result ) );
		assert( result > -BigFloatNumber );
		assert( result <  BigFloatNumber );
		index += step;
	}
}

const int AddMatrixElementsToVectorCombine = 4;
__global__ void AddMatrixElementsToVectorKernel( const float* __restrict__ matrix, int height, int width,
	const int* __restrict__ indices, float* result, size_t calls_counter, void* historyKernels )
{
	int jPos;
	int step;
	const int count = GetCudaTaskCountAndIndex( height, AddMatrixElementsToVectorCombine, jPos, step );

	PRINT_HEAD2_CNT_F( jPos, 0, 0, "AddMatrixElementsToVectorKernel1", matrix, result, height, /*width,*/ calls_counter, historyKernels, AddMatrixElementsToVectorKernel1Id );
	for( int i = 0; i < count; ++i ) {
		const int index = indices[jPos];
		if( index >= 0 && index < width ) {
			result[jPos] += matrix[jPos * width + index];

			WARN2_CNT_F( "AddMatrixElementsToVectorKernel1", matrix[jPos * width + index], matrix, result[jPos], result, height, width, index, /*jPos,*/ calls_counter, historyKernels );
		}
		jPos += step;
	}
}

const int AddMatrixElementsToVectorMulCombine = 4;
__global__ void AddMatrixElementsToVectorKernel(const float* __restrict__ matrix, int /*height*/, int width,
	const int* __restrict__ rowIndices, const int* __restrict__ columnIndices, float* result, int vectorSize, size_t calls_counter, void* historyKernels)
{
	int index;
	int step;
	const int count = GetCudaTaskCountAndIndex(vectorSize, AddMatrixElementsToVectorMulCombine, index, step);

	PRINT_HEAD2_CNT_F( index, 0, 0, "AddMatrixElementsToVectorKernel2", matrix, result, /*rowIndices columnIndices, width,*/ vectorSize, calls_counter, historyKernels, AddMatrixElementsToVectorKernel2Id );
	for(int i = 0; i < count; ++i) {
		float val = result[index] += matrix[rowIndices[index] * width + columnIndices[index]];

		WARN2_CNT_F( "AddMatrixElementsToVectorKernel2", val, matrix, val, result, vectorSize, width, index, calls_counter, historyKernels );
		index += step;
	}
}

const int AddMatrixElementsToMatrixCombine = 4;
__global__ void AddMatrixElementsToMatrixKernel(const float* __restrict__ matrix, int height, int width,
	float* result, const int* __restrict__ indices)
{
	int jPos;
	int step;
	const int count = GetCudaTaskCountAndIndex(height, AddMatrixElementsToMatrixCombine, jPos, step);

	PRINT_HEAD2_F( jPos, 0, 0, "AddMatrixElementsToMatrixKernel", matrix, result, height /*, width*/ );

	for(int i = 0; i < count; ++i) {
		const int index = indices[jPos];
		if(index >= 0 && index < width) {
			float val = result[jPos * width + index] += matrix[jPos * width + index];
			assert( isfinite( val ) );
			assert( val > -BigFloatNumber );
			assert( val <  BigFloatNumber );
		}
		jPos += step;
	}
}

const int BatchAddVectorToMatrixRowsCombine = 4;
__global__ void AddVectorToMatrixRowsKernel(int batchSize,
	const float* __restrict__ matrix, float* result, int matrixHeight,
	int matrixWidth, const float* __restrict__ vector, size_t calls_counter, void* historyKernels)
{
	const int yPos = blockIdx.y * blockDim.y + threadIdx.y;
	if(yPos < batchSize * matrixHeight) {
		const int matrixBaseIndex = yPos * matrixWidth;
		const int batch = yPos / matrixHeight;
		const int vectorBaseIndex = batch * matrixWidth;

		int index;
		int step;
		const int count = GetCudaTaskCountAndIndexX(matrixWidth, BatchAddVectorToMatrixRowsCombine, index, step);
		PRINT_HEAD3_CNT_F( yPos, index, 0, "AddVectorToMatrixRowsKernel", matrix, vector, result,
			matrixHeight, /*matrixWidth, batchSize,*/ calls_counter, historyKernels, AddVectorToMatrixRowsKernelId );

		for(int i = 0; i < count; ++i) {
			const int matrixIndex = matrixBaseIndex + index;
			float val = result[matrixIndex] = matrix[matrixIndex] + vector[vectorBaseIndex + index];

			WARN3_CNT_F( "AddVectorToMatrixRowsKernel", matrix[matrixIndex], matrix, vector[vectorBaseIndex + index], vector, val, result,
				matrixHeight, matrixWidth, index, calls_counter, historyKernels );
			index += step;
		}
	}
}

template<class T>
__global__ void AddVectorToMatrixColumnsKernel( const T* __restrict__ matrix, T* result,
	int matrixHeight, int matrixWidth, const T* __restrict__ vector )
{
	int i;
	int j;
	if( GetCudaTaskIndex2D( matrixHeight, matrixWidth, j, i ) ) {
		PRINT_HEAD3_T( i, j, 0, "AddVectorToMatrixColumnsKernel", matrix, vector, result, matrixHeight /*, matrixWidth*/ );

		const int index = matrixWidth * j + i;
		result[index] = matrix[index] + vector[j];
		if constexpr( std::is_same_v<T, float> ) {
			assert( isfinite( result[index] ) );
			assert( result[index] > -BigFloatNumber );
			assert( result[index] <  BigFloatNumber );
		}
	}
}

__global__ void SubVectorFromMatrixColumnsKernel(const float* __restrict__ matrix, float* result,
	int matrixHeight, int matrixWidth, const float* __restrict__ vector)
{
	int i;
	int j;
	if(GetCudaTaskIndex2D(matrixHeight, matrixWidth, j, i)) {
		PRINT_HEAD3_F( i, j, 0, "SubVectorFromMatrixColumnsKernel", matrix, vector, result, matrixHeight /*, matrixWidth*/ );

		const int index = matrixWidth * j + i;
		result[index] = matrix[index] - vector[j];
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

const int SumMatrixRowsAddCombineCount = 128;
template<class T>
__global__ void SumMatrixRowsAddKernel(
	int batchSize, T* result, const T* __restrict__ matrix,
	int matrixHeight, int matrixWidth, size_t calls_counter, void* historyKernels )
{
	const int height = ( matrixHeight + SumMatrixRowsAddCombineCount - 1 ) / SumMatrixRowsAddCombineCount;

	int batchIndex = -1;
	int rowIndex = -1;
	int colIndex = -1;
	if( !GetCudaTaskIndex3D( batchSize, height, matrixWidth, batchIndex, rowIndex, colIndex ) ) {
		return;
	}
	PRINT_HEAD2_CNT_T( batchIndex, rowIndex, colIndex, "SumMatrixRowsAddKernel", matrix, result,
		batchSize /*, matrixHeight, matrixWidth*/, calls_counter, historyKernels, SumMatrixRowsAddKernelId );

	rowIndex *= SumMatrixRowsAddCombineCount;
	if( rowIndex >= matrixHeight ) {
		return;
	}

	const int rowEndIndex = min( matrixHeight, rowIndex + SumMatrixRowsAddCombineCount );

	[[maybe_unused]] const T* __restrict__ base_matrix = matrix;

	matrix += ( batchIndex * matrixHeight + rowIndex ) * matrixWidth + colIndex;
	T sum = *matrix;
	for(int j = rowIndex + 1; j < rowEndIndex; ++j) {
		matrix += matrixWidth;
		sum += *matrix;
		//assert( sum > -BigFloatNumber );
		//assert( sum <  BigFloatNumber );
	}

	auto *val = result + batchIndex * matrixWidth + colIndex;
	atomicAdd( val, sum );

	WARN2_CNT_T( "SumMatrixRowsAddKernel", sum, base_matrix, *val, result,
		matrixHeight, matrixWidth, batchIndex/*, colIndex, rowIndex, rowEndIndex*/, calls_counter, historyKernels );
}

const int SumMatrixColumnsCombine = 4;
const int SumMatrixColumnsPartial = 8;
const int SumMatrixColumnsMaxAtomic = 64;
__global__ void SumMatrixColumnsKernel(float* result, const float* __restrict__ matrix,
	int matrixHeight, int matrixWidth, bool isNeg, int widthNorm, int combine)
{
	assert( threadIdx.z == 0 );

	extern __shared__ float buffer[];
	float* const acc = &buffer[threadIdx.y * blockDim.x + threadIdx.x];
	*acc = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	int index;
	int y;
	GetCudaTaskIndex2D(matrixHeight, widthNorm, y, index);
	PRINT_HEAD2_F( index, y, 0, "SumMatrixColumnsKernel", matrix, result, matrixHeight /*, matrixWidth, widthNorm, combine*/ );

	if(y < matrixHeight) {
		// Calculate partial sums
		result += y;
		matrix += y * matrixWidth;

		int step;
		const int count = GetCudaTaskCountAndIndex(matrixWidth, combine, index, step);
		matrix += index;

		for(int i = 0; i < count; ++i) {
			*acc += *matrix;
			assert( isfinite( *acc ) );
			assert( isfinite( *matrix ) );
			matrix += step;
		}
	}

	int partial = 1;
	do {
		// Put the partial sums into buffer[0] (with SumMatrixColumnsPartial stride)
		__syncthreads();
		const int nextPartial = partial * SumMatrixColumnsPartial;
		if((threadIdx.x % nextPartial) == 0) {
			for(int i = 1; i < SumMatrixColumnsPartial; ++i) {
				const int index = i * partial;
				if(threadIdx.x + index >= blockDim.x) {
					break;
				}
				*acc += acc[index];
				assert( isfinite( *acc ) );
				assert( isfinite( acc[index] ) );
			}
		}
		partial = nextPartial;
	} while(partial < blockDim.x);

	if(threadIdx.x == 0 && y < matrixHeight) {
		// Put buffer[0] into result
		if(gridDim.x > 1) {
			if(isNeg) {
				atomicAdd(result, -*acc);
			} else {
				atomicAdd(result, *acc);
			}
		} else {
			*result = isNeg ? -*acc : *acc;
		}
		assert( isfinite( *acc ) );
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
	}
}

const int MatrixLogSumExpByRowsCombine = 2;
__global__ void MatrixLogSumExpByRowsKernel(const float* __restrict__ matrix,
	int height, int width, float* result, int widthNorm, size_t calls_counter, void* historyKernels)
{
	assert( threadIdx.z == 0 );

	extern __shared__  float buffer[];
	float& my = buffer[threadIdx.y * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	const int combineCount = (width + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	const int count = GetCudaTaskCountAndIndexX(width, combineCount, index, step);
	PRINT_HEAD2_CNT_F( index, threadIdx.y * blockDim.x + threadIdx.x, blockIdx.x, "MatrixLogSumExpByRowsKernel", matrix, result,
		height /*, width, widthNorm*/, calls_counter, historyKernels, MatrixLogSumExpByRowsKernelId );

	[[maybe_unused]] const float* __restrict__ base_matrix = matrix;

	int xPos;
	int yPos;
	if(GetCudaTaskIndex2D(height, widthNorm, yPos, xPos) && count > 0) {
		matrix += yPos * width; // get the correct row
								// find the maximum
		my = matrix[index];
		for(int i = 1; i < count; ++i) {
			const float val = matrix[index + i * step];
			if(val > my) {
				my = val;
			}
		}
	} else {
		my = -FLT_MAX;
	}

	const float maxVal = ReduceMaxXSharedBuffer(buffer);
	assert( isfinite( maxVal ) );

	// Add up the needed part
	if(yPos < height && count > 0) {
		my = ExponentFunc(matrix[index] - maxVal);
		for(int i = 1; i < count; ++i) {
			my += ExponentFunc(matrix[index + i * step] - maxVal);
		}
		assert( isfinite( my ) );
	} else {
		my = 0.f;
	}

	const float sumVal = ReduceSumXSharedBuffer(buffer);

	if(yPos < height && threadIdx.x == 0) {
		if( !isfinite( sumVal ) ) {
			printf( "MatrixLogSumExpByRowsKernel: ReduceSumXSharedBuffer=%f  x=%d y=%d z=%d  count=%d index=%d step=%d  height=%d yPos=%d  call=%llu \n",
				sumVal, threadIdx.x, threadIdx.y, threadIdx.z, count, index, step, height, yPos, calls_counter );
		}
		assert( isfinite( sumVal ) );

		float val = result[yPos] = maxVal + log(sumVal);

		WARN2_CNT_F( "MatrixLogSumExpByRowsKernel", maxVal /*, sumVal*/, base_matrix, val, result, height, width, yPos /*, widthNorm*/, calls_counter, historyKernels );
	}
}

const int MatrixSoftmaxByRowsCombine = 2;
__global__ void MatrixSoftmaxByRowsKernel(const float* matrix,
	int height, int width, float* result, int widthNorm, size_t calls_counter, void* historyKernels )
{
	assert( threadIdx.z == 0 );

	extern __shared__ float buffer[];
	float& my = buffer[threadIdx.y * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	const int combineCount = (width + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	const int count = GetCudaTaskCountAndIndexX(width, combineCount, index, step);
	PRINT_HEAD2_CNT_F( index, threadIdx.y * blockDim.x + threadIdx.x, blockIdx.x, "MatrixSoftmaxByRowsKernel", matrix, result,
		height /*, width, widthNorm*/, calls_counter, historyKernels, MatrixSoftmaxByRowsKernelId );

	[[maybe_unused]] const float* __restrict__ base_matrix = matrix;

	int xPos;
	int yPos;
	if(GetCudaTaskIndex2D(height, widthNorm, yPos, xPos) && count > 0) {
		matrix += yPos * width; // get the correct row
		result += yPos * width;

		// Find the maximum
		my = matrix[index];
		for(int i = 1; i < count; ++i) {
			const float val = matrix[index + i * step];
			if(val > my) {
				my = val;
			}
		}
	} else {
		my = -FLT_MAX;
	}

	const float maxVal = ReduceMaxXSharedBuffer(buffer);
	assert( isfinite( maxVal ) );

	// Put the exponent into result and add up the needed part
	if(yPos < height && count > 0) {
		my = result[index] = ExponentFunc(matrix[index] - maxVal);
		for(int i = 1; i < count; ++i) {
			const float val = ExponentFunc(matrix[index + i * step] - maxVal);
			result[index + i * step] = val;
			my += val;
		}
		assert( isfinite( my ) );
	} else {
		my = 0.f;
	}

	const float reduce = ReduceSumXSharedBuffer( buffer );

	if(yPos < height && count > 0) {
		assert( reduce != 0.f );
		const float sumVal = 1.f / reduce;
		if( !isfinite( reduce ) || !isfinite( sumVal ) ) {
			printf( "MatrixSoftmaxByRowsKernel: ReduceSumXSharedBuffer=%f sumVal=%f x=%d y=%d z=%d count=%d index=%d step=%d height=%d yPos=%d  call=%llu \n",
				reduce, sumVal, threadIdx.x, threadIdx.y, threadIdx.z, count, index, step, height, yPos, calls_counter );
		}
		assert( isfinite( reduce ) );
		assert( isfinite( sumVal ) );

		// Divide the needed part by the total
		for(int i = 0; i < count; ++i) {
			float val = result[index + i * step] *= sumVal;

			WARN2_CNT_F( "MatrixSoftmaxByRowsKernel", /*maxVal,*/ sumVal, base_matrix, val, result, height, width, index /*, i, step, widthNorm*/, calls_counter, historyKernels );
		}
	}
}

const int MatrixSoftmaxDiffOpByRowsCombine = 2;
__global__ void MatrixSoftmaxDiffOpByRowsKernel(const float* __restrict__ first,
	const float* __restrict__ second, int height, int width, float* result, int widthNorm)
{
	assert( threadIdx.z == 0 );

	extern __shared__ float buffer[];
	float& my = buffer[threadIdx.y * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	const int combineCount = (width + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	const int count = GetCudaTaskCountAndIndexX(width, combineCount, index, step);
	PRINT_HEAD3_F( index, threadIdx.y * blockDim.x + threadIdx.x, blockIdx.x, "MatrixSoftmaxDiffOpByRowsKernel", first, second, result, width /*, height, widthNorm*/ );

	int xPos;
	int yPos;
	if(GetCudaTaskIndex2D(height, widthNorm, yPos, xPos) && count > 0) {
		first += yPos * width; // get the correct row
		second += yPos * width;
		result += yPos * width;

		// Find the dot product
		for(int i = 0; i < count; ++i) {
			my += first[index + i * step] * second[index + i * step];
		}
		assert( isfinite( my ) );
	}

	const float dotProd = ReduceSumXSharedBuffer(buffer);

	// Store the result and add up the needed part
	if(yPos < height && count > 0) {
		if( !isfinite( dotProd ) ) {
			printf( "MatrixSoftmaxDiffOpByRowsKernel: ReduceSumXSharedBuffer=%f x=%d y=%d z=%d count=%d index=%d step=%d height=%d yPos=%d \n",
				dotProd, threadIdx.x, threadIdx.y, threadIdx.z, count, index, step, height, yPos );
		}
		assert( isfinite( dotProd ) );

		for(int i = 0; i < count; ++i) {
			auto val = result[index + i * step] =
				first[index + i * step] * (second[index + i * step] - dotProd);
			assert( isfinite( val ) );
			assert( val > -BigFloatNumber );
			assert( val <  BigFloatNumber );
		}
	}
}

const int MatrixSoftmaxByColumnsCombine = 2;
__global__ void MatrixSoftmaxByColumnsKernel(const float* __restrict__ matrix,
	int height, int width, float* result, int heightNorm)
{
	assert( threadIdx.z == 0 );

	extern __shared__ float buffer[];
	float& my = buffer[threadIdx.y * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	const int combineCount = (height + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	const int count = GetCudaTaskCountAndIndexX(height, combineCount, index, step);
	PRINT_HEAD2_F( index, threadIdx.y * blockDim.x + threadIdx.x, blockIdx.x, "MatrixSoftmaxByColumnsKernel", matrix, result, height /*, width, heightNorm*/ );

	index *= width;
	step *= width;

	int xPos;
	int yPos;
	// x and y swapped
	if(GetCudaTaskIndex2D(width, heightNorm, xPos, yPos) && count > 0) {
		matrix += xPos; // get the correct column
		result += xPos;

		// Find the maximum
		my = matrix[index];
		for(int i = 1; i < count; ++i) {
			const float val = matrix[index + i * step];
			if(val > my) {
				my = val;
			}
		}
	} else {
		my = -FLT_MAX;
	}

	const float maxVal = ReduceMaxXSharedBuffer(buffer);
	assert( isfinite( maxVal ) );

	// Put the exponent into result and add up the needed part
	if(xPos < width && count > 0) {
		my = result[index] = ExponentFunc(matrix[index] - maxVal);
		for(int i = 1; i < count; ++i) {
			const float val = ExponentFunc(matrix[index + i * step] - maxVal);
			result[index + i * step] = val;
			my += val;
		}
		assert( isfinite( my ) );
	} else {
		my = 0.f;
	}

	const float reduce = ReduceSumXSharedBuffer( buffer );

	if(xPos < width && count > 0) {
		const float sumVal = 1.f / reduce;
		if( !isfinite( reduce ) || !isfinite( sumVal ) ) {
			printf( "MatrixSoftmaxByColumnsKernel: ReduceSumXSharedBuffer=%f sumVal=%f x=%d y=%d z=%d count=%d index=%d step=%d height=%d yPos=%d \n",
				reduce, sumVal, threadIdx.x, threadIdx.y, threadIdx.z, count, index, step, height, yPos );
		}
		assert( reduce != 0.f );
		assert( isfinite( sumVal ) );

		// Divide the needed part by the total
		for(int i = 0; i < count; ++i) {
			auto val = result[index + i * step] *= sumVal;
			assert( isfinite( val ) );
			assert( val > -BigFloatNumber );
			assert( val <  BigFloatNumber );
		}
	}
}

const int MatrixSoftmaxDiffOpByColumnsCombine = 2;
__global__ void MatrixSoftmaxDiffOpByColumnsKernel(const float* __restrict__ first,
	const float* __restrict__ second, int height, int width, float* result, int heightNorm)
{
	assert( threadIdx.z == 0 );

	extern __shared__ float buffer[];
	float& my = buffer[threadIdx.y * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	const int combineCount = (height + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	const int count = GetCudaTaskCountAndIndexX(height, combineCount, index, step);
	PRINT_HEAD3_F( index, threadIdx.y * blockDim.x + threadIdx.x, blockIdx.x, "MatrixSoftmaxDiffOpByColumnsKernel", first, second, result, height /*, width, heightNorm*/ );

	index *= width;
	step *= width;

	int xPos;
	int yPos;
	if(GetCudaTaskIndex2D(width, heightNorm, xPos, yPos) && count > 0) {
		first += xPos; // get the correct row
		second += xPos;
		result += xPos;

		// Find the dot product
		for(int i = 0; i < count; ++i) {
			my += first[index + i * step] * second[index + i * step];
		}
		assert( isfinite( my ) );
	}

	const float dotProd = ReduceSumXSharedBuffer(buffer);

	// Store the result and add up the needed part
	if(xPos < width && count > 0) {
		if( !isfinite( dotProd ) ) {
			printf( "MatrixSoftmaxDiffOpByColumnsKernel: ReduceSumXSharedBuffer=%f x=%d y=%d z=%d count=%d index=%d step=%d height=%d yPos=%d \n",
				dotProd, threadIdx.x, threadIdx.y, threadIdx.z, count, index, step, height, yPos );
		}
		assert( isfinite( dotProd ) );

		for(int i = 0; i < count; ++i) {
			auto val = result[index + i * step] =
				first[index + i * step] * (second[index + i * step] - dotProd);
			assert( isfinite( val ) );
			assert( val > -BigFloatNumber );
			assert( val <  BigFloatNumber );
		}
	}
}

const int FindMaxValueInRowsCombine = 4;
__global__ void FindMaxValueWithIndicesInRowsKernel(const float* __restrict__ matrix,
	int matrixHeight, int matrixWidth, float* result, int* indices, int widthNorm, size_t calls_counter, void* historyKernels)
{
	assert( threadIdx.z == 0 );

	[[maybe_unused]] const float* __restrict__ base_matrix = matrix;

	extern __shared__ CValueWithIndex threadBuffer[];
	CValueWithIndex& res = threadBuffer[threadIdx.y * blockDim.x + threadIdx.x];
	// NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum
	res.Index = 0;
	res.Value = -FLT_MAX;

	int yPos;
	int xPos;
	if(GetCudaTaskIndex2D(matrixHeight, widthNorm, yPos, xPos)) {
		PRINT_HEAD2_CNT_F( yPos, xPos, 0, "FindMaxValueWithIndicesInRowsKernel", matrix, result,
			matrixHeight /*, matrixWidth, widthNorm*/, calls_counter, historyKernels, FindMaxValueWithIndicesInRowsKernelId );

		// Find the maximum in the needed part of the row
		matrix += yPos * matrixWidth;

		const int combineCount = (matrixWidth + blockDim.x - 1) / blockDim.x;

		int index;
		int step;
		const int count = GetCudaTaskCountAndIndexX(matrixWidth, combineCount, index, step);

		for(int i = 0; i < count; ++i) {
			const float value = matrix[index];
			if(value > res.Value) {
				res.Value = value;
				assert( isfinite( res.Value ) );
				res.Index = index;
			}
			index += step;
		}
	}

	const CValueWithIndex maxVal = ReduceMaxWithIndexXSharedBuffer(threadBuffer);
	assert( isfinite( maxVal.Value ) );

	if(yPos < matrixHeight && threadIdx.x == 0) {
		float val = result[yPos] = maxVal.Value;

		WARN2_CNT_F( "FindMaxValueWithIndicesInRowsKernel", maxVal.Value, base_matrix, val, result,
			matrixHeight, matrixWidth, yPos /*, xPos, widthNorm */, calls_counter, historyKernels );
		indices[yPos] = maxVal.Index;
	}
}

__global__ void FindMaxValueInRowsKernel(const float* __restrict__ matrix,
	int matrixHeight, int matrixWidth, float* result, int widthNorm)
{
	assert( threadIdx.z == 0 );

	extern __shared__ float buffer[];
	float& my = buffer[threadIdx.y * blockDim.x + threadIdx.x];
	my = -FLT_MAX; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	int yPos;
	int xPos;
	if(GetCudaTaskIndex2D(matrixHeight, widthNorm, yPos, xPos)) {
		PRINT_HEAD2_F( yPos, xPos, 0, "FindMaxValueInRowsKernel", matrix, result, matrixHeight /*, matrixWidth, widthNorm*/);

		// Find the maximum in the needed part of the row
		matrix += yPos * matrixWidth;

		const int combineCount = (matrixWidth + blockDim.x - 1) / blockDim.x;

		int index;
		int step;
		const int count = GetCudaTaskCountAndIndexX(matrixWidth, combineCount, index, step);

		for(int i = 0; i < count; ++i) {
			const float value = matrix[index];
			if(value > my) {
				my = value;
				assert( isfinite( my ) );
			}
			index += step;
		}
	}

	const float maxVal = ReduceMaxXSharedBuffer( buffer );
	assert( isfinite( maxVal ) );

	if(yPos < matrixHeight && threadIdx.x == 0) {
		result[yPos] = maxVal;
		assert( isfinite( result[yPos] ) );
		assert( result[yPos] > -BigFloatNumber );
		assert( result[yPos] <  BigFloatNumber );
	}
}

const int FindMaxValueInColumnsCombine = 16;
__global__ void FindMaxValueInColumnsKernel( int batchSize, const float* __restrict__ matrix,
	int height, int width, float* result, int* indices, int heightNorm )
{
	extern __shared__ CValueWithIndex threadBuffer[];
	CValueWithIndex& res = threadBuffer[(threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
	// NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum
	res.Value = -FLT_MAX;
	res.Index = 0;

	int batchIndex;
	int colIndex;
	int rowIndex;
	if( GetCudaTaskIndex3D( batchSize, width, heightNorm, batchIndex, colIndex, rowIndex ) ) {
		PRINT_HEAD2_F( batchIndex, colIndex, rowIndex, "FindMaxValueInColumnsKernel", matrix, result, batchSize /*, height, width, heightNorm*/ );

		matrix += batchIndex * height * width + colIndex;

		const int combineCount = ( height + blockDim.x - 1 ) / blockDim.x;

		int step;
		const int count = GetCudaTaskCountAndIndexX( height, combineCount, rowIndex, step );

		matrix += rowIndex * width;
		for( int i = 0; i < count; ++i ) {
			if( *matrix > res.Value ) {
				res.Value = *matrix;
				assert( isfinite( res.Value ) );
				res.Index = rowIndex;
			}
			rowIndex += step;
			matrix += step * width;
		}
	}

	const CValueWithIndex maxVal = ReduceMaxWithIndexXSharedBuffer( threadBuffer );
	assert( isfinite( maxVal.Value ) );

	if( batchIndex < batchSize && colIndex < width && threadIdx.x == 0 ) {
		auto val = result[batchIndex * width + colIndex] = maxVal.Value;
		assert( isfinite( val ) );
		assert( val > -BigFloatNumber );
		assert( val <  BigFloatNumber );
		indices[batchIndex * width + colIndex] = maxVal.Index;
	}
}

static __global__ void FindMinValueInColumnsKernel( const float* matrixHandle, int matrixHeight, int matrixWidth,
	float* resultHandle, int* columnIndices )
{
	int index = 0;
	if( GetCudaTaskIndex( matrixWidth, index ) ) {
		PRINT_HEAD2_F( index, 0, 0, "FindMinValueInColumnsKernel", matrixHandle, resultHandle, matrixWidth /*, matrixHeight*/ );

		matrixHandle += index;
		resultHandle += index;
		columnIndices += index;

		for( int i = 0; i < matrixHeight; ++i ) {
			if( *matrixHandle < *resultHandle ) {
				*resultHandle = *matrixHandle;
				assert( isfinite( *resultHandle ) );
				*columnIndices = i;
			}
			matrixHandle += matrixWidth;
		}
	}
}

const int BatchVectorLookupAndCopyCombineBatch = 4;
template<class TInput, class TLookup>
__global__ void VectorChannelLookupAndCopyKernel(int batchSize, const TInput* __restrict__ input, int inputChannels,
	const TLookup* __restrict__ lookup, int vectorSize, TLookup* output, int outputChannels, int batchNorm, size_t calls_counter, void* historyKernels)
{
	int b;
	int index;
	if(!GetCudaTaskIndex2D(batchNorm, vectorSize, b, index)) {
		return;
	}
	using T = TLookup;
	PRINT_HEAD3_CNT_T( b, index, 0, "VectorChannelLookupAndCopyKernel", (T*)input, lookup, output,
	 vectorSize /*, inputChannels, batchSize, outputChannels, batchNorm*/, calls_counter, historyKernels, VectorChannelLookupAndCopyKernelId );

	[[maybe_unused]] const TInput* __restrict__ base_input = input;
	[[maybe_unused]] const TLookup* __restrict__ base_lookup = lookup;
	[[maybe_unused]] TLookup* base_output = output;

	b *= BatchVectorLookupAndCopyCombineBatch;
	const int bLast = min( batchSize, b + BatchVectorLookupAndCopyCombineBatch );
	const int count = bLast - b;

	input += b * inputChannels;
	output += b * outputChannels + index;
	lookup += index;
	for(int k = 0; k < count; ++k) {
		const int tableIndex = (int)(*input);
		input += inputChannels;
		*output = lookup[tableIndex * vectorSize];

		WARN3_CNT_T( "VectorChannelLookupAndCopyKernel", *((T*)&tableIndex), base_input, lookup[tableIndex * vectorSize], base_lookup, *output, base_output,
			vectorSize, k, index /*, b, inputChannels, batchSize, outputChannels, batchNorm*/, calls_counter, historyKernels );
		output += outputChannels;
	}
}

template<class TInput, class TLookup>
__global__ void BatchVectorChannelCopyKernel(int batchSize, const TInput* __restrict__ input,
	int inputChannels, int vectorSize, TLookup* output, int outputChannels, int batchNorm)
{
	int b;
	int index;
	if(!GetCudaTaskIndex2D(batchNorm, vectorSize, b, index)) {
		return;
	}
	using T = TLookup;
	PRINT_HEAD2_T( b, index, 0, "BatchVectorChannelCopyKernel", (T*)input, output, vectorSize );

	b *= BatchVectorLookupAndCopyCombineBatch;
	const int bLast = min( batchSize, b + BatchVectorLookupAndCopyCombineBatch );
	const int count = bLast - b;

	input += b * inputChannels;
	output += b * outputChannels + index;
	for(int k = 0; k < count; ++k) {
		*output = *input;
		if constexpr( std::is_same_v<TLookup, float> ) {
			assert( isfinite( *output ) );
			assert( *output > -BigFloatNumber );
			assert( *output <  BigFloatNumber );
		}
		input += inputChannels;
		output += outputChannels;
	}
}

const int BatchVectorLookupAndAddToTableCombine = 8;
template<class T>
__global__ void VectorChannelLookupAndAddToTableKernel(int batchSize, const T* __restrict__ input, int inputChannel,
	float* lookup, int vectorSize, float mult, const float* __restrict__ matrix, int outputChannel, int batchNorm)
{
	int b;
	int index;
	if(!GetCudaTaskIndex2D(batchNorm, vectorSize, b, index)) {
		return;
	}
	PRINT_HEAD3_F( b, index, 0, "VectorChannelLookupAndAddToTableKernel", (float*)input, matrix, lookup,
		vectorSize /*, inputChannels, batchSize, outputChannels, batchNorm*/ );

	b *= BatchVectorLookupAndAddToTableCombine;
	const int bLast = min( batchSize, b + BatchVectorLookupAndAddToTableCombine );
	const int count = bLast - b;

	input += b * inputChannel;
	matrix += b * outputChannel + index;
	lookup += index;
	for(int k = 0; k < count; ++k) {
		const int tableIndex = (int)(*input);
		input += inputChannel;

		float* res = lookup + tableIndex * vectorSize;
		atomicAdd( res, *matrix * mult);

		assert( isfinite( *res ));
		assert( *res > -BigFloatNumber );
		assert( *res <  BigFloatNumber );
		matrix += outputChannel;
	}
}

__global__ void LookupAndSumKernel( const int* __restrict__ indices, int batchSize, int indexCount,
	const float* __restrict__ table, int vectorSize, float* result )
{
	int batch;
	int elem;
	if( GetCudaTaskIndex2D( batchSize, vectorSize, batch, elem ) ) {
		PRINT_HEAD2_F( batch, elem, 0, "LookupAndSumKernel", table, result, vectorSize /*, indexCount, batchSize*/ );

		result += batch * vectorSize + elem;
		indices += batch * indexCount;
		table += elem;
		if( *indices >= 0 ) {
			*result = table[*indices * vectorSize];
			assert( isfinite( *result ) );
			assert( *result > -BigFloatNumber );
			assert( *result <  BigFloatNumber );
		} else {
			*result = 0.f;
		}
		for( int i = 1; i < indexCount; ++i ) {
			++indices;
			if( *indices >= 0 ) {
				*result += table[*indices * vectorSize];
				assert( isfinite( *result ) );
				assert( *result > -BigFloatNumber );
				assert( *result <  BigFloatNumber );
			}
		}
	}
}

__global__ void LookupAndAddToTableKernel( const int* __restrict__ indices, int batchSize, int indexCount,
	const float* __restrict__ additions, int vectorSize, float* table )
{
	int indexCoord;
	int batch;
	int vectorCoord;
	if( GetCudaTaskIndex3D( batchSize, indexCount, vectorSize, batch, indexCoord, vectorCoord ) ) {
		PRINT_HEAD2_F( indexCoord, batch, vectorCoord, "LookupAndAddToTableKernel", additions, table, vectorSize /*, indexCount, batchSize*/ );

		indices += batch * indexCount + indexCoord;
		if( *indices >= 0 ) {
			float* res = table + *indices * vectorSize + vectorCoord;
			atomicAdd( res, *( additions + batch * vectorSize + vectorCoord ) );
			assert( isfinite( *res ) );
			assert( *res > -BigFloatNumber );
			assert( *res <  BigFloatNumber );
		}
	}
}

const int EnumBinarizationCombine = 16;
template<class T>
__global__ void EnumBinarizationKernel(int batchSize, const T* __restrict__ input, int enumSize, float* result)
{
	int index;
	int step;
	const int count = GetCudaTaskCountAndIndex(batchSize * enumSize, EnumBinarizationCombine, index, step);
	PRINT_HEAD2_F( index, 0, 0, "EnumBinarizationKernel", (float*)input, result, enumSize );

	for(int i = 0; i < count; ++i) {
		const int batch = index / enumSize;
		const int pos = index % enumSize;
		if(batch >= batchSize) {
			break;
		}
		result[index] = ((int)input[batch] == pos) ? 1 : 0;
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
		index += step;
	}
}

__global__ void BitSetBinarizationKernel(int batchSize, int bitSetElementCount,
	const int* __restrict__ input, int outputVectorSize, float* result)
{
	const int BitsPerElement = sizeof(int) * CHAR_BIT;

	int index;
	int step;
	const int count = GetCudaTaskCountAndIndex( batchSize * outputVectorSize, 1, index, step );
	PRINT_HEAD2_F( index, 0, 0, "EnumBinarizationKernel", ( float* )input, result, batchSize * outputVectorSize );

	for( int i = 0; i < count; ++i, index += step ) {
		const int batchIndex = index / outputVectorSize;
		const int inputBatchBegin = batchIndex * bitSetElementCount;
		const int globalBitIndex = index % outputVectorSize;
		const int elementIndex = globalBitIndex / BitsPerElement;

		const int inputElement = input[inputBatchBegin + elementIndex];
		const int bitIndex = globalBitIndex % 32;

		result[index] = inputElement & ( 1 << bitIndex ) ? 1.0f : 0.0f;
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

const int MultiplyLookupMatrixByLookupVectorCombine = 4;
__global__ void MultiplyLookupMatrixByLookupVectorKernel(int batchSize, const float* __restrict__ matrixTable,
	int /*matrixVectorCount*/, int vectorSize, const int* __restrict__ rows, int rowCount,
	const float* __restrict__ vectorTable, int /*vectorVectorCount*/, const int* __restrict__ vector,
	float* result, int /*resultSize*/, int widthNorm)
{
	assert( threadIdx.z == 0 );

	extern __shared__  float buffer[];
	float& my = buffer[threadIdx.y * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	const int totalY = batchSize * rowCount;

	int yPos;
	int xPos;
	if(GetCudaTaskIndex2D(totalY, widthNorm, yPos, xPos)) {
		PRINT_HEAD3_F( yPos, xPos, 0, "MultiplyLookupMatrixByLookupVectorKernel", matrixTable, vectorTable, result, totalY );

		const int matrixBaseIndex = rows[yPos] * vectorSize;
		const int vectorBaseIndex = vector[yPos / rowCount] * vectorSize;

		int index;
		int step;
		const int count = GetCudaTaskCountAndIndexX(vectorSize, MultiplyLookupMatrixByLookupVectorCombine, index, step);

		for(int i = 0; i < count; ++i) {
			my += matrixTable[matrixBaseIndex + index] * vectorTable[vectorBaseIndex + index];
			assert( my );
			index += step;
		}
		assert( isfinite( my ) );
	}

	const float sum = ReduceSumXSharedBuffer(buffer);

	if(yPos < totalY && threadIdx.x == 0) {
		if( !isfinite( sum ) ) {
			printf( "MultiplyLookupMatrixByLookupVectorKernel: ReduceSumXSharedBuffer=%f x=%d y=%d z=%d totalY=%d yPos=%d \n",
				sum, threadIdx.x, threadIdx.y, threadIdx.z, totalY, yPos );
		}
		assert( isfinite( sum ) );

		// Store the result
		if(gridDim.x > 0) {
			// Several GPUs are adding in the same row, atomic operations needed
			atomicAdd(result + yPos, sum);
		} else {
			result[yPos] = sum;
		}
		assert( isfinite( result[yPos] ) );
		assert( result[yPos] > -BigFloatNumber );
		assert( result[yPos] <  BigFloatNumber );
	}
}

const int MultiplyTransposedLookupMatrixByVectorCombine = 4;
__global__  void MultiplyTransposedLookupMatrixByVectorKernel(int batchSize, const float* __restrict__ matrixTable,
	int /*matrixVectorCount*/, int width, const int* __restrict__ rows, int height,
	const float* __restrict__ vector, float* result, bool isAdd, int heightNorm)
{
	assert( threadIdx.z == 0 );

	extern __shared__  float buffer[];
	float& my = buffer[threadIdx.y * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	// The X coordinate corresponds to Height
	const int totalX = batchSize * width;
	int yPos;
	int xPos;
	GetCudaTaskIndex2D(totalX, heightNorm, xPos, yPos);
	PRINT_HEAD3_F( yPos, xPos, 0, "MultiplyTransposedLookupMatrixByVectorKernel", matrixTable, vector, result, totalX );

	const int batch = xPos / width;
	const int resultIndex = xPos;
	xPos %= width;

	if(batch < batchSize && yPos < heightNorm) {
		// Calculate the needed part of the total
		const int rowBaseIndex = batch * height;

		int index;
		int step;
		const int count = GetCudaTaskCountAndIndexX(height, MultiplyTransposedLookupMatrixByVectorCombine, index, step);

		index += rowBaseIndex;
		for(int i = 0; i < count; ++i) {
			my += matrixTable[rows[index] * width + xPos] * vector[index];
			assert( isfinite( my ) );
			index += step;
		}
	}

	const float sum = ReduceSumXSharedBuffer(buffer);

	if(batch < batchSize && yPos < heightNorm && threadIdx.x == 0) {
		if( !isfinite( sum ) ) {
			printf( "MultiplyTransposedLookupMatrixByVectorKernel: ReduceSumXSharedBuffer=%f x=%d y=%d z=%d height=%d yPos=%d \n",
				sum, threadIdx.x, threadIdx.y, threadIdx.z, height, yPos );
		}
		assert( isfinite( sum ) );

		if(gridDim.x > 1) {
			// Several GPUs are adding in the same column, atomic operations needed
			atomicAdd(result + resultIndex, sum);
		} else if(isAdd){
			result[resultIndex] += sum;
		} else {
			result[resultIndex] = sum;
		}
		assert( isfinite( result[resultIndex] ) );
		assert( result[resultIndex] > -BigFloatNumber );
		assert( result[resultIndex] <  BigFloatNumber );
	}
}

const int MultiplyVectorByTransposedLookupVectorAndAddToTableCombine = 8;
__global__ void MultiplyVectorByTransposedLookupVectorAndAddToTableKernel(int batchSize,
	float* table, int /*vectorCount*/, int vectorSize, const int* __restrict__ tableIndices,
	const float* __restrict__ first, int firstSize,
	const float* __restrict__ secondTable, const int* __restrict__ secondIndices, int vectorSizeNorm)
{
	int yPos;
	int xPos;
	GetCudaTaskIndex2D( batchSize * firstSize, vectorSizeNorm, yPos, xPos );
	PRINT_HEAD3_F( yPos, xPos, 0, "MultiplyVectorByTransposedLookupVectorAndAddToTableKernel", first, secondTable, table, batchSize * firstSize );

	if( yPos < batchSize * firstSize ) {
		const int batch = yPos / firstSize;
		const int tableIndex = tableIndices[yPos] * vectorSize;
		const int secondIndex = secondIndices[batch] * vectorSize;

		int index;
		int step;
		const int count = GetCudaTaskCountAndIndexX(vectorSize,
			MultiplyVectorByTransposedLookupVectorAndAddToTableCombine, index, step);

		const float mul = first[yPos];

		for(int i = 0; i < count; ++i) {
			const float val = secondTable[secondIndex + index] * mul;
			atomicAdd(table + tableIndex + index, val);
			index += step;
		}
	}
}

__global__ void MultiplyDiagMatrixByMatrixKernel(const float* __restrict__ first, int firstSize,
	const float* __restrict__ second, int secondWidth, float* result, size_t calls_counter, void* historyKernels )
{
	int i;
	int j;
	if(GetCudaTaskIndex2D(firstSize, secondWidth, j, i)) {
		PRINT_HEAD3_CNT_F( i, j, 0, "MultiplyDiagMatrixByMatrixKernel", first, second, result,
			firstSize, calls_counter, historyKernels, MultiplyDiagMatrixByMatrixKernelId );

		const int index = j * secondWidth + i;
		result[index] = second[index] * first[j];

		WARN3_CNT_F( "MultiplyDiagMatrixByMatrixKernel", first[j], first, second[index], second, result[index], result,
			i, j, index, /*, firstSize, secondWidth*/ calls_counter, historyKernels );
	}
}

const int Multiply1DiagMatrixByMatrixCombine = 8;
__global__ void Multiply1DiagMatrixByMatrixKernel(int batchSize, const float* __restrict__ first,
	int firstSize, const float* __restrict__ second, int secondWidth, float* result, int batchNorm)
{
	int b;
	int index;
	int matrixSize = firstSize * secondWidth;
	if(!GetCudaTaskIndex2D(batchNorm, matrixSize, b, index)) {
		return;
	}
	PRINT_HEAD3_F( index, b, 0, "Multiply1DiagMatrixByMatrixKernel", first, second, result, batchNorm );

	b *= Multiply1DiagMatrixByMatrixCombine;
	const int bLast = min( batchSize, b + Multiply1DiagMatrixByMatrixCombine );
	const int count = bLast - b;

	const int j = index / secondWidth;
	index += b * matrixSize;
	result += index;
	second += index;
	const float mult = first[j];

	for(int c = 0; c < count; ++c) {
		*result = mult * (*second);
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		second += matrixSize;
		result += matrixSize;
	}
}

const int TransposeMatrixCombine = 8;
template<class T>
__global__ void TransposeMatrixKernel(int batchSize,
	const T* __restrict__ first, int height, int medium, int width, int channels, T* result, int size)
{
	int index;
	int step;
	const int count = GetCudaTaskCountAndIndex(size, TransposeMatrixCombine, index, step);
	PRINT_HEAD2_T( index, 0, 0, "TransposeMatrixKernel", first, result, size );

	for(int i = 0; i < count; ++i) {
		const int resChannel = index % channels;
		int cur = index / channels;
		const int resHeight = cur % width;
		cur = cur / width;
		const int resMed = cur % medium;
		cur /= medium;
		const int resWidth = cur % height;
		const int resBatch = cur / height;

		int j = ( ( ( resBatch * width + resHeight ) * medium + resMed ) * height + resWidth ) * channels + resChannel;
		result[j] = first[index];

		if constexpr( std::is_same_v<T, float> ) {
			assert( isfinite( result[j] ) );
			assert( result[j] > -BigFloatNumber );
			assert( result[j] <  BigFloatNumber );
		}
		index += step;
	}
}

const int MultiplyDiagMatrixByMatrixAndSumCombine = 16;
__global__ void MultiplyDiagMatrixByMatrixAndSumKernel( int batchSize, const float* __restrict__ first,
	int firstSize, const float* __restrict__ second, int secondWidth, float* result, int batchSizeNorm, size_t calls_counter, void* historyKernels )
{
	extern __shared__ float buffer[];
	float& my = buffer[( threadIdx.z * blockDim.y + threadIdx.y ) * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	int batch;
	int column;
	int row;
	GetCudaTaskIndex3D( firstSize, secondWidth, batchSizeNorm, row, column, batch );
	PRINT_HEAD3_CNT_F( batch, column, row, "MultiplyDiagMatrixByMatrixAndSumKernel", first, second, result,
		firstSize /*, secondWidth, batchSizeNorm, batchSize*/, calls_counter, historyKernels, MultiplyDiagMatrixByMatrixAndSumKernelId );

	const bool isValidZY = row < firstSize && column < secondWidth;

	if( isValidZY ) {
		int step;
		const int count = GetCudaTaskCountAndIndex( batchSize, MultiplyDiagMatrixByMatrixAndSumCombine, batch, step );

		const float* __restrict__ currFirst = first + row + batch * firstSize;
		const float* __restrict__ currSecond = second + column + row * secondWidth + batch * secondWidth * firstSize;

		for( int i = 0; i < count; ++i ) {
			my += *currFirst * *currSecond;

			WARN3_CNT_F( "MultiplyDiagMatrixByMatrixAndSumKernel", *currFirst, first, *currSecond, second, my, result,
				firstSize, secondWidth, batchSize, calls_counter, historyKernels );
			currFirst += step * firstSize;
			currSecond += step * secondWidth * firstSize;
		}
	}

	const float sum = ReduceSumXSharedBuffer( buffer );

	if( isValidZY && threadIdx.x == 0 ) {
		if( !isfinite( sum ) ) {
			printf( "MultiplyDiagMatrixByMatrixAndSumKernel: ReduceSumXSharedBuffer=%f x=%d y=%d z=%d row=%d firstSize=%d column=%d secondWidth=%d batch=%d  call=%llu \n",
				sum, threadIdx.x, threadIdx.y, threadIdx.z, row, firstSize, column, secondWidth, batch, ( unsigned long long )calls_counter );
		}
		assert( isfinite( sum ) );

		float* const currResult = result + row * secondWidth + column;
		if( gridDim.x > 1 ) {
			atomicAdd( currResult, sum );
		} else {
			*currResult += sum;
		}
		WARN2_CNT_F( "MultiplyDiagMatrixByMatrixAndSumKernel", sum, first, *currResult, result, firstSize, secondWidth, batchSizeNorm, calls_counter, historyKernels );
	}
}

const int RowMultiplyMatrixByMatrixCombine = 32;
const int RowMultiplyMatrixByMatrixPartial = 64;
__global__ void RowMultiplyMatrixByMatrixKernel( const float* __restrict__ first,
	const float* __restrict__ second, int height, int width, float* result, int widthNorm )
{
	assert( threadIdx.z == 0 );

	extern __shared__ float buffer[];
	float* const acc = &buffer[threadIdx.y * blockDim.x + threadIdx.x];
	*acc = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	int row;
	int column;
	GetCudaTaskIndex2D( height, widthNorm, row, column );
	PRINT_HEAD3_F( column, row, 0, "RowMultiplyMatrixByMatrixKernel", first, second, result, height /*, width, widthNorm*/ );

	if( row < height ) {
		first += row * width;
		second += row * width;

		int step;
		const int count = GetCudaTaskCountAndIndex(width, RowMultiplyMatrixByMatrixCombine, column, step);

		first += column;
		second += column;
		for(int i = 0; i < count; ++i) {
			*acc += (*first) * (*second);
			assert( isfinite( *acc ) );
			assert( *acc > -BigFloatNumber );
			assert( *acc <  BigFloatNumber );
			first += step;
			second += step;
		}
	}

	__syncthreads();

	if( row < height && (threadIdx.x % RowMultiplyMatrixByMatrixPartial ) == 0 ) {
		float tmpRes = *acc;
		for(int i = 1; i < RowMultiplyMatrixByMatrixPartial && (threadIdx.x + i) < blockDim.x; ++i) {
			tmpRes += acc[i];
		}
		atomicAdd(result + row, tmpRes);
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
	}
}

const int MatrixSpreadRowsCombine = 16;
template<class T>
__global__ void MatrixSpreadRowsKernel(const T* __restrict__ source, int height, int width,
	T* result, const int* __restrict__ indices, int widthNorm, size_t calls_counter, void* historyKernels )
{
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	if( j >= height || indices[j] < 0 ) {
		return;
	}

	int i;
	int step;
	const int count = GetCudaTaskCountAndIndex(width, MatrixSpreadRowsCombine, i, step);
	PRINT_HEAD2_CNT_T( i, j, 0, "MatrixSpreadRowsKernel", source, result,
		height /*, width, widthNorm*/, calls_counter, historyKernels, MatrixSpreadRowsKernelId );

	[[maybe_unused]] const T* __restrict__ base_source = source;
	[[maybe_unused]] T* base_result = result;

	source += j * width + i;
	result += indices[j] * width + i;
	for(int c = 0; c < count; ++c) {
		*result = *source;

		WARN2_CNT_T( "MatrixSpreadRowsKernel", *source, base_source, *result, base_result,
			/*i, j,*/ height, width, step, calls_counter, historyKernels );
		source += step;
		result += step;
	}
}

__global__ void MatrixSpreadRowsAddKernel(const float* __restrict__ source, int height, int width,
	float* result, const int* __restrict__ indices, int widthNorm)
{
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	if( j >= height || indices[j] < 0 ) {
		return;
	}

	int i;
	int step;
	const int count = GetCudaTaskCountAndIndex(width, MatrixSpreadRowsCombine, i, step);
	PRINT_HEAD2_F( i, j, 0, "MatrixSpreadRowsAddKernel", source, result, height /*, width, widthNorm*/ );

	int sourceIndex = j * width + i;
	result += indices[j] * width + i;
	for(int c = 0; c < count; ++c) {
		atomicAdd(result, source[sourceIndex]);
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		sourceIndex += step;
		result += step;
	}
}

const int AddDiagMatrixToMatrixCombine = 16;
__global__ void AddDiagMatrixToMatrixKernel( const float* __restrict__ diagMatrix, const float*  __restrict__ matrix,
	int height, int width, int widthNorm, float* result )
{
	int row;
	int col;
	if( !GetCudaTaskIndex2D( height, widthNorm, row, col ) ) {
		return;
	}
	PRINT_HEAD3_F( col, row, 0, "AddDiagMatrixToMatrixKernel", diagMatrix, matrix, result, height /*, width, widthNorm*/ );

	col *= AddDiagMatrixToMatrixCombine;
	matrix += row * width + col;
	result += row * width + col;
	for( int i = col; i < min( width, col + AddDiagMatrixToMatrixCombine ); i++ ) {
		*result = *matrix;
		if( row == i ) {
			*result += diagMatrix[row];
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		matrix++;
		result++;
	}
}

const int MatrixColumnsEltwiseDivideCombine = 16;
__global__ void MatrixColumnsEltwiseDivideKernel( const float* __restrict__ matrix,
	int matrixHeight, int matrixWidth, int widthNorm,
	const float* __restrict__ vector, float* result )
{
	int row;
	int col;
	if( !GetCudaTaskIndex2D( matrixHeight, widthNorm, row, col ) ) {
		return;
	}
	PRINT_HEAD3_F( col, row, 0, "MatrixColumnsEltwiseDivideKernel", matrix, vector, result, matrixHeight /*, matrixWidth, widthNorm*/ );

	col *= MatrixColumnsEltwiseDivideCombine;
	matrix += row * matrixWidth + col;
	result += row * matrixWidth + col;
	for( int i = col; i < min( matrixWidth, col + MatrixColumnsEltwiseDivideCombine ); i++ ) {
		*result++ = *matrix++ / vector[row];
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
	}
}

const int MultiplyMatrixByDiagMatrixCombine = 16;
__global__ void MultiplyMatrixByDiagMatrixKernel( int batchSize, const float* __restrict__ first, int height,
	int width, int firstMatrixOffset, const float* __restrict__ second, int secondMatrixOffset, float* result )
{
	const int matrixSize = height * width;
	const int resultSize = batchSize * matrixSize;

	int index;
	int step;
	const int count = GetCudaTaskCountAndIndex( resultSize, MultiplyMatrixByDiagMatrixCombine, index, step );
	PRINT_HEAD3_F( index, 0, 0, "MultiplyMatrixByDiagMatrixKernel", first, second, result, resultSize /*, height, width, firstMatrixOffset, secondMatrixOffset*/ );

	for( int i = 0; i < count; ++i ) {
		const int b = index / matrixSize;
		const int row = ( index % matrixSize ) / width;
		const int col = ( index % matrixSize ) % width;
		result[index] = first[b * firstMatrixOffset + row * width + col] * second[b * secondMatrixOffset + col];
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
		index += step;
	}
}

} // namespace NeoML
