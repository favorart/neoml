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
		result[index] = vector[index % matrixWidth];
	}
}

const int AddVectorToMatrixElementsCombine = 4;
__global__ void AddVectorToMatrixElementsKernel( float* matrix, int height, int width,
	const int* __restrict__ indices, const float* __restrict__ vector )
{
	int jPos;
	int step;
	int count = GetCudaTaskCountAndIndex( height, AddVectorToMatrixElementsCombine, jPos, step );

	for( int i = 0; i < count; ++i ) {
		int index = indices[jPos];
		if( index >= 0 && index < width ) {
			matrix[jPos * width + index] += vector[jPos];
		}
		jPos += step;
	}
}

const int AddVectorToMatrixElementsMulCombine = 4;
__global__ void AddVectorToMatrixElementsKernel( float* __restrict__ matrix, int /*height*/, int width,
	const int* __restrict__ rowIndices, const int* __restrict__ columnIndices,
	const float* __restrict__ vector, int vectorSize )
{
	int index;
	int step;
	int count = GetCudaTaskCountAndIndex( vectorSize, AddVectorToMatrixElementsMulCombine, index, step );

	for( int i = 0; i < count; ++i ) {
		atomicAdd( matrix + rowIndices[index] * width + columnIndices[index], vector[index] );
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
	int count = GetCudaTaskCountAndIndex(
		vectorSize, SetVectorToMatrixElementsMulCombine, index, step );

	for( int i = 0; i < count; ++i ) {
		matrix[rowIndices[index] * width + columnIndices[index]] = vector[index];
		index += step;
	}
}

const int AddMatrixElementsToVectorCombine = 4;
__global__ void AddMatrixElementsToVectorKernel( const float* __restrict__ matrix, int height, int width,
	const int* __restrict__ indices, float* result )
{
	int jPos;
	int step;
	int count = GetCudaTaskCountAndIndex( height, AddMatrixElementsToVectorCombine, jPos, step );

	for( int i = 0; i < count; ++i ) {
		int index = indices[jPos];
		if( index >= 0 && index < width ) {
			result[jPos] += matrix[jPos * width + index];
		}
		jPos += step;
	}
}

const int AddMatrixElementsToVectorMulCombine = 4;
__global__ void AddMatrixElementsToVectorKernel(const float* __restrict__ matrix, int /*height*/, int width,
	const int* __restrict__ rowIndices, const int* __restrict__ columnIndices, float* result, int vectorSize)
{
	int index;
	int step;
	int count = GetCudaTaskCountAndIndex(vectorSize, AddMatrixElementsToVectorMulCombine, index, step);

	for(int i = 0; i < count; ++i) {
		result[index] += matrix[rowIndices[index] * width + columnIndices[index]];
		index += step;
	}
}

const int AddMatrixElementsToMatrixCombine = 4;
__global__ void AddMatrixElementsToMatrixKernel(const float* __restrict__ matrix, int height, int width,
	float* result, const int* __restrict__ indices)
{
	int jPos;
	int step;
	int count = GetCudaTaskCountAndIndex(height, AddMatrixElementsToMatrixCombine, jPos, step);

	for(int i = 0; i < count; ++i) {
		int index = indices[jPos];
		if(index >= 0 && index < width) {
			result[jPos * width + index] += matrix[jPos * width + index];
		}
		jPos += step;
	}
}

const int BatchAddVectorToMatrixRowsCombine = 4;
__global__ void AddVectorToMatrixRowsKernel(int batchSize,
	const float* __restrict__ matrix, float* result, int matrixHeight,
	int matrixWidth, const float* __restrict__ vector)
{
	const int yPos = blockIdx.y * blockDim.y + threadIdx.y;
	if(yPos < batchSize * matrixHeight) {
		int matrixBaseIndex = yPos * matrixWidth;
		int batch = yPos / matrixHeight;
		int vectorBaseIndex = batch * matrixWidth;

		int index;
		int step;
		int count = GetCudaTaskCountAndIndexX(matrixWidth, BatchAddVectorToMatrixRowsCombine, index, step);

		for(int i = 0; i < count; ++i) {
			int matrixIndex = matrixBaseIndex + index;
			result[matrixIndex] = matrix[matrixIndex] + vector[vectorBaseIndex + index];
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
		int index = matrixWidth * j + i;
		result[index] = matrix[index] + vector[j];
	}
}

__global__ void SubVectorFromMatrixColumnsKernel(const float* __restrict__ matrix, float* result,
	int matrixHeight, int matrixWidth, const float* __restrict__ vector)
{
	int i;
	int j;
	if(GetCudaTaskIndex2D(matrixHeight, matrixWidth, j, i)) {
		int index = matrixWidth * j + i;
		result[index] = matrix[index] - vector[j];
	}
}

const int SumMatrixRowsAddCombineCount = 128;
template<class T>
__global__ void SumMatrixRowsAddKernel(
	int batchSize, T* result, const T* __restrict__ matrix,
	int matrixHeight, int matrixWidth )
{
	const int height = ( matrixHeight + SumMatrixRowsAddCombineCount - 1 ) / SumMatrixRowsAddCombineCount;

	int batchIndex = -1;
	int rowIndex = -1;
	int colIndex = -1;
	if( !GetCudaTaskIndex3D( batchSize, height, matrixWidth, batchIndex, rowIndex, colIndex ) ) {
		return;
	}
	rowIndex *= SumMatrixRowsAddCombineCount;
	if( rowIndex >= matrixHeight ) {
		return;
	}

	int rowEndIndex = rowIndex + SumMatrixRowsAddCombineCount;
	if( rowEndIndex > matrixHeight ) {
		rowEndIndex = matrixHeight;
	}

	matrix += ( batchIndex * matrixHeight + rowIndex ) * matrixWidth + colIndex;
	T sum = *matrix;
	for(int j = rowIndex + 1; j < rowEndIndex; ++j) {
		matrix += matrixWidth;
		sum += *matrix;
	}

	atomicAdd( result + batchIndex * matrixWidth + colIndex, sum );
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
	if(y < matrixHeight) {
		// Calculate partial sums
		result += y;
		matrix += y * matrixWidth;

		int step;
		int count = GetCudaTaskCountAndIndex(matrixWidth, combine, index, step);
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
		int nextPartial = partial * SumMatrixColumnsPartial;
		if((threadIdx.x % nextPartial) == 0) {
			for(int i = 1; i < SumMatrixColumnsPartial; ++i) {
				int index = i * partial;
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
	}
}

const int MatrixLogSumExpByRowsCombine = 2;
__global__ void MatrixLogSumExpByRowsKernel(const float* __restrict__ matrix, int height, int width, float* result, int widthNorm)
{
	assert( threadIdx.z == 0 );

	extern __shared__  float buffer[];
	float& my = buffer[threadIdx.y * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	int combineCount = (width + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	int count = GetCudaTaskCountAndIndexX(width, combineCount, index, step);

	int xPos;
	int yPos;
	if(GetCudaTaskIndex2D(height, widthNorm, yPos, xPos) && count > 0) {
		matrix += yPos * width; // get the correct row
								// find the maximum
		my = matrix[index];
		for(int i = 1; i < count; ++i) {
			float val = matrix[index + i * step];
			if(val > my) {
				my = val;
			}
		}
	} else {
		my = -FLT_MAX;
	}

	float maxVal = ReduceMaxXSharedBuffer(buffer);
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

	float sumVal = ReduceSumXSharedBuffer(buffer);

	if(yPos < height && threadIdx.x == 0) {
		if( !isfinite( sumVal ) ) {
			printf( "MatrixLogSumExpByRowsKernel: ReduceSumXSharedBuffer=%f x=%d y=%d z=%d count=%d index=%d step=%d height=%d yPos=%d \n",
				sumVal, threadIdx.x, threadIdx.y, threadIdx.z, count, index, step, height, yPos );
		}
		assert( isfinite( sumVal ) );
		result[yPos] = maxVal + log(sumVal);
		assert( isfinite( result[yPos] ) );
	}
}

const int MatrixSoftmaxByRowsCombine = 2;
__global__ void MatrixSoftmaxByRowsKernel(const float* matrix,
	int height, int width, float* result, int widthNorm)
{
	assert( threadIdx.z == 0 );

	extern __shared__ float buffer[];
	float& my = buffer[threadIdx.y * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	int combineCount = (width + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	int count = GetCudaTaskCountAndIndexX(width, combineCount, index, step);

	int xPos;
	int yPos;
	if(GetCudaTaskIndex2D(height, widthNorm, yPos, xPos) && count > 0) {
		matrix += yPos * width; // get the correct row
		result += yPos * width;

		// Find the maximum
		my = matrix[index];
		for(int i = 1; i < count; ++i) {
			float val = matrix[index + i * step];
			if(val > my) {
				my = val;
			}
		}
	} else {
		my = -FLT_MAX;
	}

	float maxVal = ReduceMaxXSharedBuffer(buffer);
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
			printf( "MatrixSoftmaxByRowsKernel: ReduceSumXSharedBuffer=%f sumVal=%f x=%d y=%d z=%d count=%d index=%d step=%d height=%d yPos=%d \n",
				reduce, sumVal, threadIdx.x, threadIdx.y, threadIdx.z, count, index, step, height, yPos );
		}
		assert( isfinite( reduce ) );
		assert( isfinite( sumVal ) );

		// Divide the needed part by the total
		for(int i = 0; i < count; ++i) {
			result[index + i * step] *= sumVal;
			assert( isfinite( result[index + i * step] ) );
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

	int combineCount = (width + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	int count = GetCudaTaskCountAndIndexX(width, combineCount, index, step);

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

	float dotProd = ReduceSumXSharedBuffer(buffer);

	// Store the result and add up the needed part
	if(yPos < height && count > 0) {
		if( !isfinite( dotProd ) ) {
			printf( "MatrixSoftmaxDiffOpByRowsKernel: ReduceSumXSharedBuffer=%f x=%d y=%d z=%d count=%d index=%d step=%d height=%d yPos=%d \n",
				dotProd, threadIdx.x, threadIdx.y, threadIdx.z, count, index, step, height, yPos );
		}
		assert( isfinite( dotProd ) );

		for(int i = 0; i < count; ++i) {
			result[index + i * step] =
				first[index + i * step] * (second[index + i * step] - dotProd);
			assert( isfinite( result[index + i * step] ) );
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

	int combineCount = (height + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	int count = GetCudaTaskCountAndIndexX(height, combineCount, index, step);
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
			float val = matrix[index + i * step];
			if(val > my) {
				my = val;
			}
		}
	} else {
		my = -FLT_MAX;
	}

	float maxVal = ReduceMaxXSharedBuffer(buffer);
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
			result[index + i * step] *= sumVal;
			assert( isfinite( result[index + i * step] ) );
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

	int combineCount = (height + blockDim.x - 1) / blockDim.x;

	int index;
	int step;
	int count = GetCudaTaskCountAndIndexX(height, combineCount, index, step);
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

	float dotProd = ReduceSumXSharedBuffer(buffer);

	// Store the result and add up the needed part
	if(xPos < width && count > 0) {
		if( !isfinite( dotProd ) ) {
			printf( "MatrixSoftmaxDiffOpByColumnsKernel: ReduceSumXSharedBuffer=%f x=%d y=%d z=%d count=%d index=%d step=%d height=%d yPos=%d \n",
				dotProd, threadIdx.x, threadIdx.y, threadIdx.z, count, index, step, height, yPos );
		}
		assert( isfinite( dotProd ) );

		for(int i = 0; i < count; ++i) {
			result[index + i * step] =
				first[index + i * step] * (second[index + i * step] - dotProd);
			assert( isfinite( result[index + i * step] ) );
		}
	}
}

const int FindMaxValueInRowsCombine = 4;
__global__ void FindMaxValueWithIndicesInRowsKernel(const float* __restrict__ matrix,
	int matrixHeight, int matrixWidth, float* result, int* indices, int widthNorm)
{
	assert( threadIdx.z == 0 );

	extern __shared__ CValueWithIndex threadBuffer[];
	CValueWithIndex& res = threadBuffer[threadIdx.y * blockDim.x + threadIdx.x];
	// NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum
	res.Index = 0;
	res.Value = -FLT_MAX;

	int yPos;
	int xPos;
	if(GetCudaTaskIndex2D(matrixHeight, widthNorm, yPos, xPos)) {
		// Find the maximum in the needed part of the row
		matrix += yPos * matrixWidth;
		int combineCount = (matrixWidth + blockDim.x - 1) / blockDim.x;

		int index;
		int step;
		int count = GetCudaTaskCountAndIndexX(matrixWidth, combineCount, index, step);

		for(int i = 0; i < count; ++i) {
			float value = matrix[index];
			if(value > res.Value) {
				res.Value = value;
				res.Index = index;
			}

			index += step;
		}
	}

	CValueWithIndex maxVal = ReduceMaxWithIndexXSharedBuffer(threadBuffer);
	assert( isfinite( maxVal.Value ) );

	if(yPos < matrixHeight && threadIdx.x == 0) {
		result[yPos] = maxVal.Value;
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
		// Find the maximum in the needed part of the row
		matrix += yPos * matrixWidth;
		int combineCount = (matrixWidth + blockDim.x - 1) / blockDim.x;

		int index;
		int step;
		int count = GetCudaTaskCountAndIndexX(matrixWidth, combineCount, index, step);

		for(int i = 0; i < count; ++i) {
			float value = matrix[index];
			if(value > my) {
				my = value;
			}

			index += step;
		}
	}

	float maxVal = ReduceMaxXSharedBuffer( buffer );
	assert( isfinite( maxVal ) );

	if(yPos < matrixHeight && threadIdx.x == 0) {
		result[yPos] = maxVal;
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
		matrix += batchIndex * height * width + colIndex;
		int combineCount = ( height + blockDim.x - 1 ) / blockDim.x;

		int step;
		int count = GetCudaTaskCountAndIndexX( height, combineCount, rowIndex, step );

		matrix += rowIndex * width;
		for( int i = 0; i < count; ++i ) {
			if( *matrix > res.Value ) {
				res.Value = *matrix;
				res.Index = rowIndex;
			}

			rowIndex += step;
			matrix += step * width;
		}
	}

	CValueWithIndex maxVal = ReduceMaxWithIndexXSharedBuffer( threadBuffer );
	assert( isfinite( maxVal.Value ) );

	if( batchIndex < batchSize && colIndex < width && threadIdx.x == 0 ) {
		result[batchIndex * width + colIndex] = maxVal.Value;
		indices[batchIndex * width + colIndex] = maxVal.Index;
	}
}

static __global__ void FindMinValueInColumnsKernel( const float* matrixHandle, int matrixHeight, int matrixWidth,
	float* resultHandle, int* columnIndices )
{
	int index = 0;
	if( GetCudaTaskIndex( matrixWidth, index ) ) {
		matrixHandle += index;
		resultHandle += index;
		columnIndices += index;

		for( int i = 0; i < matrixHeight; ++i ) {
			if( *matrixHandle < *resultHandle ) {
				*resultHandle = *matrixHandle;
				*columnIndices = i;
			}
			matrixHandle += matrixWidth;
		}
	}
}

const int BatchVectorLookupAndCopyCombineBatch = 4;
template<class TInput, class TLookup>
__global__ void VectorChannelLookupAndCopyKernel(int batchSize, const TInput* __restrict__ input, int inputChannels,
	const TLookup* __restrict__ lookup, int vectorSize, TLookup* output, int outputChannels, int batchNorm)
{
	int b;
	int index;
	if(!GetCudaTaskIndex2D(batchNorm, vectorSize, b, index)) {
		return;
	}

	b *= BatchVectorLookupAndCopyCombineBatch;
	int bLast = b + BatchVectorLookupAndCopyCombineBatch;
	if(bLast > batchSize) {
		bLast = batchSize;
	}

	int count = bLast - b;

	input += b * inputChannels;
	output += b * outputChannels + index;
	lookup += index;
	for(int k = 0; k < count; ++k) {
		int tableIndex = (int)(*input);
		input += inputChannels;
		*output = lookup[tableIndex * vectorSize];
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

	b *= BatchVectorLookupAndCopyCombineBatch;
	int bLast = b + BatchVectorLookupAndCopyCombineBatch;
	if(bLast > batchSize) {
		bLast = batchSize;
	}

	int count = bLast - b;

	input += b * inputChannels;
	output += b * outputChannels + index;
	for(int k = 0; k < count; ++k) {
		*output = *input;
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
	b *= BatchVectorLookupAndAddToTableCombine;
	int bLast = b + BatchVectorLookupAndAddToTableCombine;
	if(bLast > batchSize) {
		bLast = batchSize;
	}

	int count = bLast - b;

	input += b * inputChannel;
	matrix += b * outputChannel + index;
	lookup += index;
	for(int k = 0; k < count; ++k) {
		int tableIndex = (int)(*input);
		input += inputChannel;
		atomicAdd(lookup + tableIndex * vectorSize, *matrix * mult);
		matrix += outputChannel;
	}
}

__global__ void LookupAndSumKernel( const int* __restrict__ indices, int batchSize, int indexCount,
	const float* __restrict__ table, int vectorSize, float* result )
{
	int batch;
	int elem;
	if( GetCudaTaskIndex2D( batchSize, vectorSize, batch, elem ) ) {
		result += batch * vectorSize + elem;
		indices += batch * indexCount;
		table += elem;
		if( *indices >= 0 ) {
			*result = table[*indices * vectorSize];
		} else {
			*result = 0.f;
		}
		for( int i = 1; i < indexCount; ++i ) {
			++indices;
			if( *indices >= 0 ) {
				*result += table[*indices * vectorSize];
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
		indices += batch * indexCount + indexCoord;
		if( *indices >= 0 ) {
			atomicAdd( table + *indices * vectorSize + vectorCoord,
				*( additions + batch * vectorSize + vectorCoord ) );
		}
	}
}

const int EnumBinarizationCombine = 16;
template<class T>
__global__ void EnumBinarizationKernel(int batchSize, const T* __restrict__ input, int enumSize, float* result)
{
	int index;
	int step;
	int count = GetCudaTaskCountAndIndex(batchSize * enumSize, EnumBinarizationCombine, index, step);

	for(int i = 0; i < count; ++i) {
		int batch = index / enumSize;
		int pos = index % enumSize;
		if(batch >= batchSize) {
			break;
		}
		result[index] = ((int)input[batch] == pos) ? 1 : 0;
		assert( isfinite( result[index] ) );
		index += step;
	}
}

__global__ void BitSetBinarizationKernel(int batchSize, int bitSetElementCount,
	const int* __restrict__ input, int outputVectorSize, float* result)
{
	const int BitsPerElement = sizeof(int) * CHAR_BIT;

	int index;
	int step;
	int count = GetCudaTaskCountAndIndex( batchSize * outputVectorSize, 1, index, step );

	for( int i = 0; i < count; ++i, index += step ) {
		int batchIndex = index / outputVectorSize;
		int inputBatchBegin = batchIndex * bitSetElementCount;
		int globalBitIndex = index % outputVectorSize;
		int elementIndex = globalBitIndex / BitsPerElement;

		int inputElement = input[inputBatchBegin + elementIndex];
		int bitIndex = globalBitIndex % 32;

		result[index] = inputElement & ( 1 << bitIndex ) ? 1.0f : 0.0f;
		assert( isfinite( result[index] ) );
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

	int totalY = batchSize * rowCount;
	int yPos;
	int xPos;
	if(GetCudaTaskIndex2D(totalY, widthNorm, yPos, xPos)) {
		int matrixBaseIndex = rows[yPos] * vectorSize;
		int vectorBaseIndex = vector[yPos / rowCount] * vectorSize;

		int index;
		int step;
		int count = GetCudaTaskCountAndIndexX(vectorSize, MultiplyLookupMatrixByLookupVectorCombine, index, step);

		for(int i = 0; i < count; ++i) {
			my += matrixTable[matrixBaseIndex + index] * vectorTable[vectorBaseIndex + index];
			index += step;
		}
		assert( isfinite( my ) );
	}

	float sum = ReduceSumXSharedBuffer(buffer);

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
	int totalX = batchSize * width;
	int yPos;
	int xPos;
	GetCudaTaskIndex2D(totalX, heightNorm, xPos, yPos);

	int batch = xPos / width;
	int resultIndex = xPos;
	xPos %= width;

	if(batch < batchSize && yPos < heightNorm) {
		// Calculate the needed part of the total
		int rowBaseIndex = batch * height;

		int index;
		int step;
		int count = GetCudaTaskCountAndIndexX(height, MultiplyTransposedLookupMatrixByVectorCombine, index, step);
		index += rowBaseIndex;
		for(int i = 0; i < count; ++i) {
			my += matrixTable[rows[index] * width + xPos] * vector[index];
			index += step;
		}
	}

	float sum = ReduceSumXSharedBuffer(buffer);

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
	if( yPos < batchSize * firstSize ) {
		int batch = yPos / firstSize;
		int tableIndex = tableIndices[yPos] * vectorSize;
		int secondIndex = secondIndices[batch] * vectorSize;

		int index;
		int step;
		int count = GetCudaTaskCountAndIndexX(vectorSize,
			MultiplyVectorByTransposedLookupVectorAndAddToTableCombine, index, step);

		float mul = first[yPos];

		for(int i = 0; i < count; ++i) {
			float val = secondTable[secondIndex + index] * mul;
			atomicAdd(table + tableIndex + index, val);
			index += step;
		}
	}
}

__global__ void MultiplyDiagMatrixByMatrixKernel(const float* __restrict__ first, int firstSize,
	const float* __restrict__ second, int secondWidth, float* result)
{
	int i;
	int j;
	if(GetCudaTaskIndex2D(firstSize, secondWidth, j, i)) {
		int index = j * secondWidth + i;
		result[index] = second[index] * first[j];
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

	b *= Multiply1DiagMatrixByMatrixCombine;
	int bLast = b + Multiply1DiagMatrixByMatrixCombine;
	if(bLast > batchSize) {
		bLast = batchSize;
	}

	int count = bLast - b;

	int j = index / secondWidth;
	index += b * matrixSize;
	result += index;
	second += index;
	float mult = first[j];

	for(int c = 0; c < count; ++c) {
		*result = mult * (*second);
		second += matrixSize;
		result += matrixSize;
	}
}

const int TransposeMatrixCombine = 8;
template<class T> __global__ void TransposeMatrixKernel(int batchSize,
	const T* __restrict__ first, int height, int medium, int width, int channels, T* result, int size)
{
	int index;
	int step;
	int count = GetCudaTaskCountAndIndex(size, TransposeMatrixCombine, index, step);

	for(int i = 0; i < count; ++i) {
		int resChannel = index % channels;
		int cur = index / channels;
		int resHeight = cur % width;
		cur = cur / width;
		int resMed = cur % medium;
		cur /= medium;
		int resWidth = cur % height;
		int resBatch = cur / height;

		result[(((resBatch * width + resHeight) * medium + resMed) * height + resWidth) * channels + resChannel] =
			first[index];

		index += step;
	}
}

const int MultiplyDiagMatrixByMatrixAndSumCombine = 16;
__global__ void MultiplyDiagMatrixByMatrixAndSumKernel( int batchSize, const float* __restrict__ first,
	int firstSize, const float* __restrict__ second, int secondWidth, float* result, int batchSizeNorm )
{
	extern __shared__ float buffer[];
	float& my = buffer[( threadIdx.z * blockDim.y + threadIdx.y ) * blockDim.x + threadIdx.x];
	my = 0.f; // NOTE: all threads are not used in the current task, should not interfere in the reduce max or sum

	int batch;
	int column;
	int row;
	GetCudaTaskIndex3D( firstSize, secondWidth, batchSizeNorm, row, column, batch );

	bool isValidZY = row < firstSize && column < secondWidth;

	if( isValidZY ) {
		int step;
		int count = GetCudaTaskCountAndIndex( batchSize, MultiplyDiagMatrixByMatrixAndSumCombine, batch, step );

		const float* __restrict__ currFirst = first + row + batch * firstSize;
		const float* __restrict__ currSecond = second + column + row * secondWidth + batch * secondWidth * firstSize;

		for( int i = 0; i < count; ++i ) {
			my += *currFirst * *currSecond;
			currFirst += step * firstSize;
			currSecond += step * secondWidth * firstSize;
		}
	}

	float sum = ReduceSumXSharedBuffer( buffer );

	if( isValidZY && threadIdx.x == 0 ) {
		if( !isfinite( sum ) ) {
			printf( "MultiplyDiagMatrixByMatrixAndSumKernel: ReduceSumXSharedBuffer=%f x=%d y=%d z=%d row=%d firstSize=%d column=%d secondWidth=%d batch=%d \n",
				sum, threadIdx.x, threadIdx.y, threadIdx.z, row, firstSize, column, secondWidth, batch );
		}
		assert( isfinite( sum ) );

		float* currResult = result + row * secondWidth + column;
		if( gridDim.x > 1 ) {
			atomicAdd( currResult, sum );
		} else {
			*currResult += sum;
		}
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

	if( row < height ) {
		first += row * width;
		second += row * width;

		int step;
		int count = GetCudaTaskCountAndIndex(width, RowMultiplyMatrixByMatrixCombine, column, step);
		first += column;
		second += column;
		for(int i = 0; i < count; ++i) {
			*acc += (*first) * (*second);
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
	}
}

const int MatrixSpreadRowsCombine = 16;
template<class T>
__global__ void MatrixSpreadRowsKernel(const T* __restrict__ source, int height, int width,
	T* result, const int* __restrict__ indices, int widthNorm)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if( j >= height || indices[j] < 0 ) {
		return;
	}

	int i;
	int step;
	int count = GetCudaTaskCountAndIndex(width, MatrixSpreadRowsCombine, i, step);
	source += j * width + i;
	result += indices[j] * width + i;
	for(int c = 0; c < count; ++c) {
		*result = *source;
		source += step;
		result += step;
	}
}

__global__ void MatrixSpreadRowsAddKernel(const float* __restrict__ source, int height, int width,
	float* result, const int* __restrict__ indices, int widthNorm)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if( j >= height || indices[j] < 0 ) {
		return;
	}

	int i;
	int step;
	int count = GetCudaTaskCountAndIndex(width, MatrixSpreadRowsCombine, i, step);
	int sourceIndex = j * width + i;
	result += indices[j] * width + i;
	for(int c = 0; c < count; ++c) {
		atomicAdd(result, source[sourceIndex]);
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

	col *= AddDiagMatrixToMatrixCombine;
	matrix += row * width + col;
	result += row * width + col;
	for( int i = col; i < min( width, col + AddDiagMatrixToMatrixCombine ); i++ ) {
		*result = *matrix;
		if( row == i ) {
			*result += diagMatrix[row];
		}
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

	col *= MatrixColumnsEltwiseDivideCombine;
	matrix += row * matrixWidth + col;
	result += row * matrixWidth + col;
	for( int i = col; i < min( matrixWidth, col + MatrixColumnsEltwiseDivideCombine ); i++ ) {
		*result++ = *matrix++ / vector[row];
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
	int count = GetCudaTaskCountAndIndex( resultSize, MultiplyMatrixByDiagMatrixCombine, index, step );

	for( int i = 0; i < count; ++i ) {
		const int b = index / matrixSize;
		const int row = ( index % matrixSize ) / width;
		const int col = ( index % matrixSize ) % width;
		result[index] = first[b * firstMatrixOffset + row * width + col] * second[b * secondMatrixOffset + col];
		assert( isfinite( result[index] ), "MultiplyMatrixByDiagMatrixKernel.result" );
		index += step;
	}
}

} // namespace NeoML
