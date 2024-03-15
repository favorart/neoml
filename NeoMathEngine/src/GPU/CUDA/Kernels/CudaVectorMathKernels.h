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
#include <Kernels/CudaRandom.h>
#include <CudaBlobDesc.h>
#include <stdio.h>

namespace NeoML {

const int VectorFillCombineCount = 8;

template<class T>
__global__ void VectorCopyKernel( T* result, const T* first, int count, int num, const char* name, size_t calls_counter, void* historyKernels )
{
	int index;
	int step;
	int actionCount = GetCudaTaskCountAndIndex( count, VectorFillCombineCount, index, step );
	PRINT_HEAD2_CNT_T( index, 0, 0, "VectorCopyKernel", first, result, count, calls_counter, historyKernels, VectorCopyKernelId );

	[[maybe_unused]] T* base_result = result;
	[[maybe_unused]] const T* base_first = first;

	result += index;
	first += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = *first;

		WARN2_CNT_SPEC_T( "VectorCopyKernel", *first, base_first, *result, base_result,
			count, i, index, num, name, calls_counter, historyKernels, VectorCopyKernelId ); //step, blockIdx.x, blockDim.x, threadIdx.x
		result += step;
		first += step;
	}
}

template<class T>
__global__ void VectorFillKernel(T* mem, T value, int count, int num, const char* name, size_t calls_counter, void* historyKernels)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorFillCombineCount, index, step);
	PRINT_HEAD1_CNT_T( index, 0, 0, "VectorFillKernel", mem, count, calls_counter, historyKernels, VectorFillKernelId );

	[[maybe_unused]] T* base_mem = mem;

	mem += index;

	for(int i = 0; i < actionCount; ++i) {
		*mem = value;

		WARN2_CNT_SPEC_T( "VectorFillKernel", 0.f, nullptr, *mem, base_mem,
			count, i, index, num, name, calls_counter, historyKernels, VectorFillKernelId ); //step, blockIdx.x, blockDim.x, threadIdx.x
		mem += step;
	}
}

template<class T>
__global__ void VectorFillSpecialKernel(T* mem, T value, int count)
{
	int index;
	int step;
	int actionCount = GetCudaTaskCountAndIndex( count, VectorFillCombineCount, index, step );

	mem += index;

	for( int i = 0; i < actionCount; ++i ) {
		*mem = value;
		mem += step;
	}
}

const int VectorFillHandleCombineCount = 8;
template<class T>
__global__ void VectorFillHandleKernel(T* mem, int count, const T* __restrict__ value, int num, const char* name, size_t calls_counter, void* historyKernels)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorFillHandleCombineCount, index, step);
	PRINT_HEAD1_CNT_T( index, 0, 0, "VectorFillHandleKernel", mem, count, calls_counter, historyKernels, VectorFillHandleKernelId );

	[[maybe_unused]] T* base_mem = mem;

	mem += index;

	for(int i = 0; i < actionCount; ++i) {
		*mem = *value;

		WARN2_CNT_SPEC_T( "VectorFillHandleKernel", 0.f, nullptr, *mem, base_mem,
			count, i, index, num, name, calls_counter, historyKernels, VectorFillHandleKernelId ); //step, blockIdx.x, blockDim.x, threadIdx.x
		mem += step;
	}
}

const int VectorConvertCombineCount = 8;
template<class From, class To>
__global__ void VectorConvertKernel( const From* from, To* to, int count )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorConvertCombineCount, index, step);

	using T = To;
	PRINT_HEAD2_T( index, 0, 0, "VectorConvertKernel", (T*)from, to, count );

	from += index;
	to += index;

	for( int i = 0; i < actionCount; ++i ) {
		*to = static_cast<To>( *from );
		if constexpr( std::is_same_v<To, float> ) {
			assert( isfinite( *to ) );
			assert( *to > -BigFloatNumber );
			assert( *to <  BigFloatNumber );
		}
		from += step;
		to += step;
	}
}

template<class T>
__global__ void VectorBroadcastCopyKernel( T* to, const T* from, CCudaBlobDesc toDesc, CCudaBlobDesc fromDesc,
	int additionalWidth, int resultSize )
{
	int toIndex = 0;
	int fromIndex = 0;
	int mul = additionalWidth;
	if( GetCudaTaskIndex( resultSize, toIndex ) ) {
		PRINT_HEAD2_T( toIndex, 0, 0, "VectorBroadcastCopyKernel", from, to, resultSize );

		to += toIndex * additionalWidth;
		for( int i = CCudaBlobDesc::MaxDimensions - 1; i >= 0; i-- ) {
			if( fromDesc.DimSize( i ) != 1 ) {
				fromIndex += ( toIndex % toDesc.DimSize( i ) ) * mul;
				mul *= fromDesc.DimSize( i );
			}
			toIndex /= toDesc.DimSize( i );
		}
		from += fromIndex;
		for( int i = 0; i < additionalWidth; i++ ) {
			*to = *from;
			if constexpr( std::is_same_v<T, float> ) {
				assert( isfinite( *to ) );
				assert( *to > -BigFloatNumber );
				assert( *to <  BigFloatNumber );
			}
			to++;
			from++;
		}
	}
}

const int VectorFillBernoulliCombine = 8;
__global__ void VectorFillBernoulliKernel( float* result, float p, int vectorSize, float value, int randomInit )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( ( vectorSize + 3 ) / 4, VectorFillBernoulliCombine, index, step );

	PRINT_HEAD1_F( index, 0, 0, "VectorFillBernoulliKernel", result, vectorSize );

	if( actionCount > 0 ) {
		CCudaRandom random( randomInit );
		random.Skip( index );

		index *= 4;
		result += index;

		const unsigned int threshold = p * UINT_MAX;

		for( int i = 0; i < actionCount; ++i ) {
			CIntArray<4> generated = random.Next();
			for( int j = 0; j < 4 && index + j < vectorSize; ++j ) {
				result[j] = generated[j] <= threshold ? value : 0;
				assert( isfinite( result[j] ) );
				assert( result[j] > -BigFloatNumber );
				assert( result[j] <  BigFloatNumber );
			}
			result += step * 4;
			index += step * 4;
			random.Skip( step - 1 );
		}
	}
}

__global__ void FilterSmallValuesKernel( float* data, float threshold, int count )
{
	int start;
	int stepSize;
	const int stepCount = GetCudaTaskCountAndIndex( count, VectorFillCombineCount, start, stepSize );

	PRINT_HEAD1_F( start, 0, 0, "FilterSmallValuesKernel", data, count );

	data += start;

	for( int i = 0; i < stepCount; ++i ) {
		if( *data < threshold && *data > -threshold ) {
			*data = 0.f;
		}
		data += stepSize;
	}
}

const int VectorSumCombineCount = 16;
__global__ void VectorSumKernel(const float* __restrict__ mem, int count, float* result, bool isNeg, bool setZero)
{
	assert( threadIdx.z == 0 );
	assert( threadIdx.y == 0 );

	extern __shared__ float sumData[];

	float sum = 0;

	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorSumCombineCount, index, step );

	PRINT_HEAD2_F( index, 0, 0, "VectorSumKernel", mem, result, count );

	mem += index;
	for(int i = 0; i < actionCount; ++i) {
		sum += *mem;
		mem += step;
	}

	sumData[threadIdx.x] = isNeg ? -sum : sum;

	__syncthreads();

	if(threadIdx.x != 0)
		return;

	sum = sumData[0];
	for(int i = 1; i < blockDim.x; ++i) {
		sum += sumData[i];
	}

	if(setZero) {
		*result = sum;
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
	} else if(gridDim.x == 1) {
		*result += sum;
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
	} else {
		atomicAdd(result, sum);
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
	}
}

__global__ void VectorSumAlongDimensionKernel( const float* __restrict__ input, int precedingDims, int dims,
	int followingDims, float* result )
{
	int x;
	int y;
	if( GetCudaTaskIndex2D( precedingDims, followingDims, x, y ) ) {
		PRINT_HEAD2_F( x, y, 0, "VectorSumAlongDimensionKernel1", input, result, dims );

		input += y * dims * precedingDims + x;
		result += y * precedingDims + x;
		*result = 0;
		for( int i = 0; i < dims; i++ ) {
			*result += *input;
			assert( isfinite( *result ) );
			assert( *result > -BigFloatNumber );
			assert( *result <  BigFloatNumber );
			input += precedingDims;
		}
	}
}

template<class T>
__global__ void VectorCumSumAlongDimensionKernel( const T* __restrict__ input, int precedingDims, int dims,
	int followingDims, T* result, bool reverse )
{
	int x;
	int y;
	if( GetCudaTaskIndex2D( precedingDims, followingDims, x, y ) ) {
		PRINT_HEAD2_T( x, y, 0, "VectorCumSumAlongDimensionKernel2", input, result, dims );

		const int firstElemOffset = reverse ? ( dims - 1 ) * precedingDims : 0;
		const int offset = y * dims * precedingDims + x + firstElemOffset;
		input += offset;
		result += offset;
		T curSum = *input;
		*result = curSum;
		const int step = reverse ? -precedingDims : precedingDims;
		for( int i = 1; i < dims; i++ ) {
			input += step;
			result += step;
			curSum += *input;
			*result = curSum;
			if constexpr( std::is_same_v<T, float> ) {
				assert( isfinite( *result ) );
				assert( *result > -BigFloatNumber );
				assert( *result <  BigFloatNumber );
			}
		}
	}
}

__global__ void VectorSumAlongDimensionDiagKernel( const float* __restrict__ input, int precedingDims, int dims,
	int followingDims, float* result )
{
	int x;
	int y;
	if( GetCudaTaskIndex2D( precedingDims, followingDims, x, y ) ) {
		PRINT_HEAD2_F( x, y, 0, "VectorSumAlongDimensionDiagKernel", input, result, dims );

		const int width = precedingDims * dims * followingDims;
		const int startOffset = y * dims * precedingDims + x;
		input += startOffset;
		result += ( y * precedingDims + x ) * width + startOffset;
		for( int i = 0; i < dims; i++ ) {
			*result += *input;
			assert( isfinite( *result ) );
			assert( *result > -BigFloatNumber );
			assert( *result <  BigFloatNumber );
			input += precedingDims;
			result += precedingDims;
		}
	}
}

__global__ void VectorCumSumAlongDimensionDiagKernel( const float* __restrict__ input, int precedingDims, int dims,
	int followingDims, float* result )
{
	int x;
	int y;
	if( GetCudaTaskIndex2D( precedingDims, dims * followingDims, x, y ) ) {
		PRINT_HEAD2_F( x, y, 0, "VectorCumSumAlongDimensionDiagKernel", input, result, dims );

		const int cumDim = y / followingDims;
		const int width = precedingDims * dims * followingDims;
		const int startOffset = ( y % followingDims ) * dims * precedingDims + x;
		input += startOffset;
		result += ( y * precedingDims + x ) * width + startOffset;
		for( int i = 0; i <= cumDim; i++ ) {
			*result += *input;
			assert( isfinite( *result ) );
			assert( *result > -BigFloatNumber );
			assert( *result <  BigFloatNumber );
			input += precedingDims;
			result += precedingDims;
		}
	}
}

const int VectorEqualCombineCount = 16;
__global__ void VectorEqualKernel( const int* first,
	const int* second, float* result, int count )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorEqualCombineCount, index, step );

	PRINT_HEAD1_F( index, 0, 0, "VectorEqualKernel", /*first, second,*/ result, count );

	first += index;
	second += index;
	result += index;

	for( int action = 0; action < actionCount; ++action ) {
		*result = (*first == *second) ? 1.0f : 0.0f;
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorEqualValueKernel( const int* first, 
	float* result, int count, const int* __restrict__ value )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorEqualCombineCount, index, step );

	PRINT_HEAD1_F( index, 0, 0, "VectorEqualValueKernel", /*first, value,*/ result, count );

	first += index;
	result += index;

	for( int action = 0; action < actionCount; ++action ) {
		*result = (*first == *value) ? 1.0f : 0.0f;
		first += step;
		result += step;
	}
}

const int VectorActivationCombineCount = 8;
__global__ void VectorELUKernel( const float* __restrict__ first, float* result, int count,
	const float* __restrict__ alpha )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorActivationCombineCount, index, step );

	PRINT_HEAD2_F( index, 0, 0, "VectorELUKernel", first, result, count );

	first += index;
	result += index;

	for( int action = 0; action < actionCount; ++action ) {
		*result = ( *first >= 0 ) ? *first : ( *alpha * ( ExponentFunc( *first ) - 1. ) );
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

__global__ void VectorELUDiffKernel( const float* __restrict__ first, const float* __restrict__ second,
	float* result, int count, const float* __restrict__ alpha )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorActivationCombineCount, index, step );

	PRINT_HEAD3_F( index, 0, 0, "VectorELUKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = ( *first >= 0 ) ? *second : ( *second * ExponentFunc( *first ) * *alpha );
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorELUDiffOpKernel( const float* __restrict__ first, const float* __restrict__ second,
	float* result, int count, const float* __restrict__ alpha )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorActivationCombineCount, index, step );

	PRINT_HEAD3_F( index, 0, 0, "VectorELUDiffOpKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = ( *first >= 0 ) ? *second : ( *second * ( *first + *alpha ) );
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorReLUKernel(const float* first, float* result,
	int count, const float* __restrict__ threshold)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorReLUKernel", first, threshold, result, count );

	first += index;
	result += index;
	if(*threshold > 0) {
		for(int i = 0; i < actionCount; ++i) {
			const float value = min(*first, *threshold);
			assert( isfinite( value ) );
			assert( *result > -BigFloatNumber );
			assert( *result <  BigFloatNumber );
			*result = value > 0 ? value : 0;
			first += step;
			result += step;
		}
	} else {
		for(int i = 0; i < actionCount; ++i) {
			const float value = *first;
			assert( isfinite( value ) );
			assert( *result > -BigFloatNumber );
			assert( *result <  BigFloatNumber );
			*result = value > 0 ? value : 0;
			first += step;
			result += step;
		}
	}
}

__global__ void VectorReLUDiffKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count, const float* __restrict__ threshold)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorReLUDiffKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	if(*threshold > 0) {
		for(int i = 0; i < actionCount; ++i) {
			*result = (*first > 0 && *first < *threshold) ? *second : 0;
			assert( isfinite( *result ) );
			assert( *result > -BigFloatNumber );
			assert( *result <  BigFloatNumber );
			first += step;
			second += step;
			result += step;
		}
	} else {
		for(int i = 0; i < actionCount; ++i) {
			*result = ( *first > 0 ) ? *second : 0;
			assert( isfinite( *result ) );
			assert( *result > -BigFloatNumber );
			assert( *result <  BigFloatNumber );
			first += step;
			second += step;
			result += step;
		}
	}
}

__global__ void VectorLeakyReLUKernel( const float* __restrict__ first, float* result,
	int count, const float* __restrict__ alpha )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorActivationCombineCount, index, step );

	PRINT_HEAD3_F( index, 0, 0, "VectorLeakyReLUKernel", first, alpha, result, count );

	first += index;
	result += index;
	for( int i = 0; i < actionCount; ++i ) {
		const float value = *first;
		*result = ( value > 0 ) ? value : ( *alpha * value );
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

__global__ void VectorLeakyReLUDiffKernel( const float* __restrict__ first, const float* __restrict__ second,
	float* result, int count, const float* __restrict__ alpha )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorActivationCombineCount, index, step );

	PRINT_HEAD3_F( index, 0, 0, "VectorLeakyReLUDiffKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = ( *first > 0 ) ? *second : ( *second * *alpha );
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorHSwishKernel( const float* first, float* result, int count )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorActivationCombineCount, index, step );

	PRINT_HEAD2_F( index, 0, 0, "VectorHSwishKernel", first, result, count );

	first += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		const float value = *first;
		if( value <= -3.f ) {
			*result = 0;
		} else if( value >= 3.f ) {
			*result = value;
		} else {
			*result = value * ( value + 3.f ) / 6.f;
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

__global__ void VectorHSwishDiffKernel( const float* __restrict__ first, const float* __restrict__ second,
	float* result, int count )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorActivationCombineCount, index, step );

	PRINT_HEAD3_F( index, 0, 0, "VectorHSwishDiffKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		const float value = *first;
		if( value <= -3.f ) {
			*result = 0;
		} else if( value >= 3.f ) {
			*result = *second;
		} else {
			*result = ( value / 3.f + 0.5f ) * *second;
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

const int VectorEltwiseMaxCombineCount = 8;
__global__ void VectorEltwiseMaxKernel(const float* first, const float* second,
	float* result, int count, size_t calls_counter, void* historyKernels)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorEltwiseMaxCombineCount, index, step);
	PRINT_HEAD3_CNT_F( index, 0, 0, "VectorEltwiseMaxKernel", first, second, result, count, calls_counter, historyKernels, VectorEltwiseMaxKernelId );

	[[maybe_unused]] float* base_result = result;
	[[maybe_unused]] const float* base_first = first;
	[[maybe_unused]] const float* base_second = second;

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		const float value1 = *first;
		const float value2 = *second;
		*result = ( value1 > value2 ) ? value1 : value2;

		WARN3_CNT_SPEC_F( "VectorEltwiseMaxKernel", value1, base_first, value2, base_second, *result, base_result,
			count, i, index, 0, (char*)0 /*, step, blockIdx.x, blockDim.x, threadIdx.x*/, calls_counter, historyKernels, VectorEltwiseMaxKernelId );
		first += step;
		second += step;
		result += step;
	}
}

const int VectorEltwiseMinCombineCount = 8;
__global__ void VectorEltwiseMinKernel(const float* first, const float* second,
	float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorEltwiseMinCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorEltwiseMinKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		const float value1 = *first;
		const float value2 = *second;
		*result = ( value1 < value2 ) ? value1 : value2;

		if( !isfinite( *result ) ||
			*result < -BigFloatNumber ||
			*result >  BigFloatNumber ) {
			printf( "VectorEltwiseMinKernel: first=%f second=%f result=%f i=%d index=%d step=%d blockIdx.x=%u blockDim.x=%u threadIdx.x=%u \n",
				*first, *second, *result, i, index, step, blockIdx.x, blockDim.x, threadIdx.x );
		}
		//assert( isfinite( *result ) );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorAbsKernel(const float* first, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD2_F( index, 0, 0, "VectorAbsKernel", first, result, count );

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		const float value = *first;
		*result = ( value > 0 ) ? value : -value;
		assert( isfinite( *result ) );
		first += step;
		result += step;
	}
}

__global__ void VectorAbsDiffKernel(const float* __restrict__ first, const float* __restrict__ second,
	float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorAbsDiffKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = ( *first > 0 ) ? *second : ( - *second );
		assert( isfinite( *result ) );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorHingeKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD2_F( index, 0, 0, "VectorHingeKernel", first, result, count );

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		const float value = 1 - *first;
		*result = ( value > 0 ) ? value : 0;
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

__global__ void VectorHingeDiffKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorHingeDiffKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = ( *first < 1 ) ? -*second : 0;
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorSquaredHingeKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD2_F( index, 0, 0, "VectorSquaredHingeKernel", first, result, count );

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		float value = *first;
		if(value < -1) {
			*result = -4 * value;
		} else {
			value = 1 - value;
			*result = ( value < 0 ) ? 0 : ( value * value );
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

__global__ void VectorSquaredHingeDiffKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorSquaredHingeDiffKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		float value = *first;
		if(value < -1) {
			*result = -4 * (*second);
		} else {
			value = 1 - value;
			*result = ( value < 0 ) ? 0 : ( -2 * value * ( *second ) );
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorHuberKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD2_F( index, 0, 0, "VectorHuberKernel", first, result, count );

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		if(*first < -1) {
			*result = -(*first) - 0.5f;
		} else if(*first > 1) {
			*result = *first - 0.5f;
		} else {
			*result = *first * (*first) * 0.5f;
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

__global__ void VectorHuberDiffKernel(const float* __restrict__ first,
	float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD2_F( index, 0, 0, "VectorHuberDiffKernel", first, result, count );

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		if(*first < -1) {
			*result = -1;
		} else if(*first > 1) {
			*result = 1;
		} else {
			*result = *first;
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

__global__ void VectorHardTanhKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD2_F( index, 0, 0, "VectorHardTanhKernel", first, result, count );

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		const float value = *first;
		if(value < -1) {
			*result = -1;
		} else if(value > 1) {
			*result = 1;
		} else {
			*result = value;
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

__global__ void VectorHardTanhDiffKernel(const float* __restrict__ first, const float* __restrict__ second,
	float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorHardTanhDiffKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		const float value = *first;
		if(value <= -1 || value >= 1) {
			*result = 0;
		} else {
			*result = *second;
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorHardSigmoidKernel(const float* __restrict__ first, float* result, int count, const float* slope, const float* bias)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorHardSigmoidKernel", first, slope, result, count );

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		const float value = *first * *slope + *bias;
		if(value < 0) {
			*result = 0;
		} else if(value > 1) {
			*result = 1;
		} else {
			*result = value;
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

__global__ void VectorHardSigmoidDiffKernel(const float* __restrict__ first, const float* __restrict__ second,
	float* result, int count, const float* slope, const float* bias)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorHardSigmoidDiffKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	const float minX = -*bias / *slope;
	const float maxX = ( 1.f - *bias ) / *slope;

	for(int i = 0; i < actionCount; ++i) {
		const float value = *first;
		if( ( value <= minX ) || ( value >= maxX ) ) {
			*result = 0;
		} else {
			*result = *second * *slope;
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorHardSigmoidDiffOpKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count, const float* slope)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorHardSigmoidDiffOpKernel", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		const float value = *first;
		if( value <= 0 || value >= 1 ) {
			*result = 0;
		} else {
			*result = *second * *slope;
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorExpKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD2_F( index, 0, 0, "VectorExpKernel", first,  result, count );

		result[index] = ExponentFunc(first[index]);
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

__global__ void VectorLogKernel( const float* __restrict__ first, float* result, int count )
{
	int index;
	if( GetCudaTaskIndex( count, index ) ) {
		PRINT_HEAD2_F( index, 0, 0, "VectorLogKernel", first, result, count );

		result[index] = logf(min(max(first[index], FLT_MIN), FLT_MAX));
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

__global__ void VectorNegLogKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD2_F( index, 0, 0, "VectorNegLogKernel", first, result, count );

		result[index] = -logf(min(max(first[index], FLT_MIN), FLT_MAX));
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

__global__ void VectorErfKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD2_F( index, 0, 0, "VectorErfKernel", first, result, count );

		result[index] = erff(first[index]);
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

__global__ void VectorBernulliKLDerivativeKernel(const float* __restrict__ first,
	float* result, int count, const float* __restrict__ target)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD3_F( index, 0, 0, "VectorBernulliKLDerivativeKernel", first, target, result, count );

		const float value = first[index];
		float klDer = -*target / value + (1 - *target) / (1 - value);
		if(klDer < -10) {
			klDer = -10;
		} else if(klDer > 10) {
			klDer = 10;
		}
		result[index] = klDer;
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

const int VectorAddCombineCount = 8;
template<class T>
__global__ void VectorAddKernel(const T* __restrict__ first, const T* __restrict__ second,
	T* result, int count, const char* name, int num, size_t calls_counter, void* historyKernels)
{
	int index = 0;
	int step = 0;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorAddCombineCount, index, step);
	PRINT_HEAD3_CNT_SPEC_T( index, 0, 0, "VectorAddKernel", first, second, result, count, num, name, calls_counter, historyKernels, VectorAddKernelId );

	[[maybe_unused]] const T* __restrict__ base_first = first;
	[[maybe_unused]] const T* __restrict__ base_second = second;
	[[maybe_unused]] T* base_result = result;

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = *first + *second;

		WARN3_CNT_SPEC_T( "VectorAddKernel", *first, base_first, *second, base_second, *result, base_result,
			count, i, index /*, step, blockIdx.x, blockDim.x, threadIdx.x*/, num, name, calls_counter, historyKernels, VectorAddKernelId );
		first += step;
		second += step;
		result += step;
	}
}

const int VectorAddValueCombineCount = 8;
template<class T>
__global__ void VectorAddValueKernel( const T* __restrict__ first,
	T* result, int count, const T* __restrict__ addition, size_t calls_counter, void* historyKernels )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorAddValueCombineCount, index, step);
	PRINT_HEAD3_CNT_T( index, 0, 0, "VectorAddValueKernel", first, addition, result, count, calls_counter, historyKernels, VectorAddValueKernelId );

	[[maybe_unused]] const T* __restrict__ base_first = first;
	[[maybe_unused]] T* base_result = result;

	first += index;
	result += index;

	if constexpr( std::is_same_v<T, float> ) {
		assert( isfinite( *addition ) );
	}

	for(int i = 0; i < actionCount; ++i) {
		*result = *first + *addition;

		WARN3_CNT_T( "VectorAddValueKernel", *first, base_first, *addition, addition, *result, base_result,
			count, i, index /*, step*/, calls_counter, historyKernels );
		first += step;
		result += step;
	}
}

const int VectorSubCombineCount = 8;
template<class T>
__global__ void VectorSubKernel( const T* __restrict__ first, const T* __restrict__ second, T* result, int count, size_t calls_counter, void* historyKernels )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorSubCombineCount, index, step );
	PRINT_HEAD3_CNT_T( index, 0, 0, "VectorSubKernel 1", first, second, result, count, calls_counter, historyKernels, VectorSub1KernelId );

	[[maybe_unused]] const T* __restrict__ base_first = first;
	[[maybe_unused]] const T* __restrict__ base_second = second;
	[[maybe_unused]] T* base_result = result;

	first += index;
	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = *first - *second;

		WARN3_CNT_T( "VectorSubKernel1", *first, base_first, *second, base_second, *result, base_result,
			count, i, index /*, step*/, calls_counter, historyKernels );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorSubKernel( const float* __restrict__ first,
	float second, float* result, int count, size_t calls_counter, void* historyKernels )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorSubCombineCount, index, step );
	PRINT_HEAD3_CNT_F( index, 0, 0, "VectorSubKernel2", first, &second, result, count, calls_counter, historyKernels, VectorSub2KernelId );

	[[maybe_unused]] const float* __restrict__ base_first = first;
	[[maybe_unused]] float* base_result = result;

	first += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = *first - second;

		WARN3_CNT_F( "VectorSubKernel2", *first, base_first, second, &second, *result, base_result,
			count, i, index /*, step*/, calls_counter, historyKernels );
		first += step;
		result += step;
	}
}

__global__ void VectorSubKernel( float first,
	const float* __restrict__ second, float* result, int count, size_t calls_counter, void* historyKernels )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorSubCombineCount, index, step );
	PRINT_HEAD3_CNT_F( index, 0, 0, "VectorSubKernel3", &first, second, result, count, calls_counter, historyKernels, VectorSub3KernelId );

	[[maybe_unused]] const float* __restrict__ base_second = second;
	[[maybe_unused]] float* base_result = result;

	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = first - *second;

		WARN3_CNT_F( "VectorSubKernel3", first, &first, *second, base_second, *result, base_result,
			count, i, index /*, step*/, calls_counter, historyKernels );
		second += step;
		result += step;
	}
}

// MultiplyAndSub
__global__ void VectorMultiplyAndSubKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count, const float* __restrict__ mult)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD3_F( index, 0, 0, "VectorMultiplyAndSubKernel", first, second, result, count );

		result[index] = first[index] - *mult * second[index];

		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

const int VectorMultiplyCombineCount = 8;
template<class T>
__global__ void VectorMultiplyKernel(const T* __restrict__ first,
	T* result, int count, const T* __restrict__ multiplier, size_t calls_counter, void* historyKernels)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorMultiplyCombineCount, index, step);
	PRINT_HEAD3_CNT_T( index, 0, 0, "VectorMultiplyKernel", first, multiplier, result, count, calls_counter, historyKernels, VectorMultiplyKernelId );

	[[maybe_unused]] const T* __restrict__ base_first = first;
	[[maybe_unused]] T* base_result = result;

	first += index;
	result += index;
	if constexpr( std::is_same_v<T, float> ) {
		assert( isfinite( *multiplier ) );
	}

	for(int i = 0; i < actionCount; ++i) {
		*result = *first * (*multiplier);

		WARN3_CNT_T( "VectorMultiplyKernel", *first, base_first, *multiplier, multiplier, *result, base_result,
			count, i, index /*, step, blockIdx.x, blockDim.x, threadIdx.x*/, calls_counter, historyKernels );
		first += step;
		result += step;
	}
}

__global__ void VectorNegMultiplyKernel(const float* __restrict__ first,
	float* result, int count, const float* __restrict__ multiplier)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorMultiplyCombineCount, index, step);
	PRINT_HEAD3_F( index, 0, 0, "VectorNegMultiplyKernel", first, multiplier, result, count );

	first += index;
	result += index;

	const float mul = -(*multiplier);
	assert( isfinite( mul ) );

	for(int i = 0; i < actionCount; ++i) {
		*result = *first * mul;
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

const int VectorEltwiseMultiplyCombineCount = 8;
template<class T>
__global__ void VectorEltwiseMultiplyKernel(const T* __restrict__ first,
	const T* __restrict__ second, T* result, int count, const char* name, int blockCount, int threadCount, size_t calls_counter, void* historyKernels)
{
	int index = 0;
	int step = 0;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorEltwiseMultiplyCombineCount, index, step);
	PRINT_HEAD3_CNT_SPEC_T( index, 0, 0, "VectorEltwiseMultiplyKernel", first, second, result, count, 1, name,
		calls_counter, historyKernels, VectorEltwiseMultiplyKernelId );

	[[maybe_unused]] const T* __restrict__ base_first = first;
	[[maybe_unused]] const T* __restrict__ base_second = second;
	[[maybe_unused]] T* base_result = result;

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = *first * (*second);

		WARN3_CNT_SPEC_T( "VectorEltwiseMultiplyKernel", *first, base_first, *second, base_second, *result, base_result,
			count, i, index /*, step, blockIdx.x, blockDim.x, threadIdx.x, blockCount, threadCount*/, 1, name,
			calls_counter, historyKernels, VectorEltwiseMultiplyKernelId );
		first += step;
		second += step;
		result += step;
	}
}

const int VectorEltwiseMultiplyAddCombineCount = 8;
__global__ void VectorEltwiseMultiplyAddKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorEltwiseMultiplyAddCombineCount, index, step);
	PRINT_HEAD3_F( index, 0, 0, "VectorEltwiseMultiplyAdd", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result += *first * (*second);
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

const int VectorEltwiseNegMultiplyCombineCount = 8;
__global__ void VectorEltwiseNegMultiplyKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorEltwiseNegMultiplyCombineCount, index, step);
	PRINT_HEAD3_F( index, 0, 0, "VectorEltwiseNegMultiply", first, second, result, count );

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = - *first * (*second);
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		second += step;
		result += step;
	}
}

const int VectorEltwiseDivideCombineCount = 8;
template<class T>
__global__ void VectorEltwiseDivideKernel(const T* __restrict__ first,
	const T* __restrict__ second, T* result, int count, size_t calls_counter, void* historyKernels)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorEltwiseDivideCombineCount, index, step);

	PRINT_HEAD3_CNT_T( index, 0, 0, "VectorEltwiseDivide", first, second, result, count, calls_counter, historyKernels, VectorEltwiseDivideKernelId );

	[[maybe_unused]] const T* __restrict__ base_first = first;
	[[maybe_unused]] const T* __restrict__ base_second = second;
	[[maybe_unused]] T* base_result = result;

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = *first / (*second);

		WARN3_CNT_SPEC_T( "VectorMinMaxKernel", *first, base_first, *second, base_second, *result, base_result,
			count, i, index, 0, (char*)0 /*, step*/, calls_counter, historyKernels, VectorEltwiseDivideKernelId );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorEltwisePowerKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD3_F( index, 0, 0, "VectorEltwisePowerKernel", first, second, result, count );
		result[index] = (second[index] == 1) ? first[index] : powf(first[index], second[index]);
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

__global__ void VectorSqrtKernel(const float* __restrict__ first, float* result, int count, size_t calls_counter, void* historyKernels)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD2_CNT_F( index, 0, 0, "VectorSqrtKernel", first, result, count, calls_counter, historyKernels, VectorSqrtKernelId );

		result[index] = sqrtf(first[index]);

		WARN2_CNT_SPEC_F( "VectorSqrtKernel", first[index], first, result[index], result,
			count, 0, index, 0, (char*)0, calls_counter, historyKernels, VectorSqrtKernelId );
	}
}

const int VectorInvCombineCount = 8;
__global__ void VectorInvKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorInvCombineCount, index, step);

	PRINT_HEAD2_F( index, 0, 0, "VectorInvKernel", first, result, count );

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		assert( isfinite( *first ) );
		if(-FLT_MIN <= *first && *first < 0) {
			*result = -FLT_MAX;
		} else if(0 <= *first && *first <= FLT_MIN) {
			*result = FLT_MAX;
		} else {
			*result = 1.f / (*first);
		}
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );
		first += step;
		result += step;
	}
}

const int VectorMinMaxCombineCount = 8;
__global__ void VectorMinMaxKernel(const float* __restrict__ first, float* result, int count,
	const float* __restrict__ minValue, const float* __restrict__ maxValue, size_t calls_counter, void* historyKernels)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorMinMaxCombineCount, index, step);

	PRINT_HEAD2_CNT_F( index, 0, 0, "VectorMinMaxKernel", first, result, count, calls_counter, historyKernels, VectorMinMaxKernelId );

	[[maybe_unused]] const float* __restrict__ base_first = first;
	[[maybe_unused]] float* base_result = result;

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = min(max(*first, *minValue), *maxValue);

		WARN2_CNT_F( "VectorMinMaxKernel", *first, base_first, *result, base_result,
			count, i, index /*, *minValue, *maxValue, blockIdx.x, blockDim.x, threadIdx.x*/, calls_counter, historyKernels );
		first += step;
		result += step;
	}
}

__global__ void VectorSigmoidKernel(const float* __restrict__ first, float* result, int count, size_t calls_counter, void* historyKernels)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD2_CNT_F( index, 0, 0, "VectorSigmoidKernel", first, result, count, calls_counter, historyKernels, VectorSigmoidKernelId );

		assert( isfinite( first[index] ) );
		result[index] = 1.f / (1.f + ExponentFunc(-first[index]));

		WARN2_CNT_F( "VectorSigmoidKernel", first[index], first, result[index], result,
			count, 0, index, calls_counter, historyKernels );
	}
}

__global__ void VectorSigmoidDiffKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count, size_t calls_counter, void* historyKernels)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD3_CNT_F( index, 0, 0, "VectorSigmoidDiffKernel", first, second, result, count, calls_counter, historyKernels, VectorSigmoidDiffKernelId );

		assert( isfinite( first[index] ) );
		const float expVal = ExponentFunc(-first[index]);
		const float expVal1 = expVal + 1.f;
		result[index] = expVal / expVal1 / expVal1;
		result[index] *= second[index];

		WARN3_CNT_F( "VectorSigmoidDiffKernel", first[index], first, second[index], second, result[index], result,
			count, 0, index, calls_counter, historyKernels );
	}
}

const int VectorSigmoidDiffOpCombineCount = 4;
__global__ void VectorSigmoidDiffOpKernel(const float* __restrict__ first, const float* __restrict__ second,
	float* result, int vectorSize, size_t calls_counter, void* historyKernels)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(vectorSize, VectorSigmoidDiffOpCombineCount, index, step);

	PRINT_HEAD3_CNT_F( index, 0, 0, "VectorSigmoidDiffOpKernel", first, second, result, vectorSize, calls_counter, historyKernels, VectorSigmoidDiffOpKernelId );

	[[maybe_unused]] const float* __restrict__ base_first = first;
	[[maybe_unused]] const float* __restrict__ base_second = second;
	[[maybe_unused]] float* base_result = result;

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = *first * (1.f - *first) * *second;

		WARN3_CNT_F( "VectorSigmoidDiffOpKernel", *first, base_first, *second, base_second, *result, base_result,
			vectorSize, i, index, /*step,*/ calls_counter, historyKernels );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorTanhKernel(const float* __restrict__ first, float* result, int count, size_t calls_counter, void* historyKernels)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD2_CNT_F( index, 0, 0, "VectorTanhKernel", first, result, count, calls_counter, historyKernels, VectorTanhKernelId );

		result[index] = -1.f  + 2 / (1.f + ExponentFunc(-2 * first[index]));

		WARN2_CNT_F( "VectorTanhKernel", first[index], first, result[index], result,
			count, 0, index, calls_counter, historyKernels );
	}
}

__global__ void VectorTanhDiffKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count, size_t calls_counter, void* historyKernels)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD3_CNT_F( index, 0, 0, "VectorTanhDiffKernel", first, second, result, count, calls_counter, historyKernels, VectorTanhDiffKernelId );

		float tanh = -1.f  + 2 / (1.f + ExponentFunc(-2 * first[index]));
		result[index] = second[index] * (1.f - tanh * tanh);

		WARN3_CNT_F( "VectorTanhDiffKernel", first[index], first, second[index], second, result[index], result,
			count, 0, index, calls_counter, historyKernels );
	}
}

const int VectorTanhDiffOpCombineCount = 4;
__global__ void VectorTanhDiffOpKernel(const float* __restrict__ first, const float* __restrict__ second,
	float* result, int vectorSize, size_t calls_counter, void* historyKernels)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(vectorSize, VectorTanhDiffOpCombineCount, index, step);

	PRINT_HEAD3_CNT_F( index, 0, 0, "VectorTanhDiffOpKernel", first, second, result, vectorSize, calls_counter, historyKernels, VectorTanhDiffOpKernelId );

	[[maybe_unused]] const float* __restrict__ base_first = first;
	[[maybe_unused]] const float* __restrict__ base_second = second;
	[[maybe_unused]] float* base_result = result;

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = (1.f - *first * *first) * *second;

		WARN3_CNT_F( "VectorTanhDiffOpKernel", *first, base_first, *second, base_second, *result, base_result,
			vectorSize, i, index, /*step,*/ calls_counter, historyKernels );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorPowerKernel(float exponent, const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD3_F( index, 0, 0, "VectorPowerKernel", first, &exponent, result, count );

		result[index] = powf(first[index], exponent);
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

__global__ void VectorPowerDiffKernel(float exponent, const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD3_F( index, 0, 0, "VectorPowerDiffKernel", first, second, result, count );

		result[index] = second[index] * exponent * powf(first[index], exponent - 1);
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

__global__ void VectorPowerDiffOpKernel(float exponent, const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		PRINT_HEAD3_F( index, 0, 0, "VectorPowerDiffOpKernel", first, second, result, count );

		result[index] = second[index] * exponent * powf(first[index], (exponent - 1.f) / exponent);
		assert( isfinite( result[index] ) );
		assert( result[index] > -BigFloatNumber );
		assert( result[index] <  BigFloatNumber );
	}
}

__global__ void VectorL1DiffAddKernel(const float* __restrict__ first, const float* __restrict__ second,
	float* result, int vectorSize, const float* __restrict__ threshold, const float* __restrict__ mult)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(vectorSize, VectorActivationCombineCount, index, step);

	PRINT_HEAD3_F( index, 0, 0, "VectorL1DiffAddKernel", first, second, result, vectorSize );

	first += index;
	second += index;
	result += index;

	const float negThres = -*threshold;
	const float thres = *threshold;
	const float mulVal = *mult;

	for(int i = 0; i < actionCount; ++i) {
		float x = *second;
		if(x < negThres) {
			x = negThres;
		} else if(x > thres) {
			x = thres;
		}

		*result = *first + mulVal * x;
		assert( isfinite( *result ) );
		assert( *result > -BigFloatNumber );
		assert( *result <  BigFloatNumber );

		first += step;
		second += step;
		result += step;
	}
}

__global__ void vectorNotKernel( const int* __restrict__ first,
	int* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		PRINT_HEAD2_I( index, 0, 0, "vectorNotKernel", first, result, vectorSize );

		result[index] = first[index] == 0 ? 1 : 0;
	}
}

__global__ void vectorGreaterEqualToZeroKernel( const int* __restrict__ first,
	float* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		PRINT_HEAD1_F( index, 0, 0, "vectorGreaterEqualToZeroKernel", result, vectorSize );

		result[index] = first[index] >= 0 ? 1.f : 0.f;
	}
}

template<class TSrc, class TDst>
__global__ void vectorLessKernel( const TSrc* __restrict__ first, const TSrc* __restrict__ second,
	TDst* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		using T = TDst;
		PRINT_HEAD3_T( index, 0, 0, "vectorLessKernel1", ( T* )first, ( T* )second, result, vectorSize );

		result[index] = static_cast<TDst>( first[index] < second[index] ? 1 : 0 );
	}
}

__global__ void vectorLessKernel( const float* __restrict__ first, float second,
	float* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		PRINT_HEAD3_F( index, 0, 0, "vectorLessKernel2", first, &second, result, vectorSize );

		result[index] = first[index] < second ? 1.f : 0.f;
	}
}

__global__ void vectorLessKernel( float first, const float* __restrict__ second,
	float* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		PRINT_HEAD3_F( index, 0, 0, "vectorLessKernel3", &first, second, result, vectorSize );

		result[index] = first < second[index] ? 1.f : 0.f;
	}
}

template<class T>
__global__ void vectorEqualKernel( const T* __restrict__ first, const T* __restrict__ second,
	int* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		PRINT_HEAD1_I( index, 0, 0, "vectorEqualKernel", /*first, second,*/ result, vectorSize );

		result[index] = first[index] == second[index] ? 1 : 0;
	}
}

template<class T>
__global__ void vectorWhereKernel( const int* __restrict__ first, const T* __restrict__ second,
	const T* __restrict__ third, T* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		PRINT_HEAD3_T( index, 0, 0, "vectorWhereKernel", /*first,*/ second, third, result, vectorSize );

		result[index] = first[index] != 0 ? second[index] : third[index];
	}
}

__global__ void VectorFindMaxValueInSetKernel( CCudaConstVectorArray vectors,
	float* result, int vectorSize)
{
	int index;
	if(GetCudaTaskIndex(vectorSize, index)) {
		PRINT_HEAD1_F( index, 0, 0, "VectorFindMaxValueInSetKernel", result, /*, vectors*/ vectorSize );

		float res = result[index];
		for(int j = 0; j < vectors.VectorCount; ++j) {
			const float value = vectors.Vectors[j][index];
			if(value > res) {
				res = value;
			}
		}
		result[index] = res;
		assert( isfinite( result[index] ) );
	}
}

__global__ void VectorFindMaxValueInSetWithIndicesKernel( CCudaConstVectorArray vectors,
	float* result, int* rowIndices, int vectorSize, int startVectorIndex)
{
	int index;
	if( GetCudaTaskIndex(vectorSize, index) ) {
		PRINT_HEAD1_F( index, 0, 0, "VectorFindMaxValueInSetWithIndicesKernel", result, /*, vectors, rowIndices*/ vectorSize /*,startVectorIndex*/ );

		float resIndex = rowIndices[index];
		float res = result[index];
		for( int j = 0; j < vectors.VectorCount; ++j ) {
			const float value = vectors.Vectors[j][index];
			if( value > res ) {
				res = value;
				resIndex = startVectorIndex + j;
			}
		}
		rowIndices[index] = resIndex;
		result[index] = res;
		assert( isfinite( result[index] ) );
	}
}

// VectorSpreadValues
__global__ void VectorSpreadValuesKernel(const float* __restrict__ source,
	CCudaVectorArray vectors, const int* __restrict__ rowIndices, int vectorSize, int startVectorIndex)
{
	int index;
	if(GetCudaTaskIndex(vectorSize, index)) {
		PRINT_HEAD1_F( index, 0, 0, "VectorSpreadValuesKernel", source, /*, vectors, rowIndices*/ vectorSize /*,startVectorIndex*/ );

		if( startVectorIndex <= rowIndices[index] && rowIndices[index] < startVectorIndex + vectors.VectorCount ) {
			*(vectors.Vectors[rowIndices[index] - startVectorIndex] + index ) = source[index];
			assert( isfinite( source[index] ) );
		}
	}
}

__global__ void VectorTopKDiffKernel( const float* __restrict__ source,
	const int* __restrict__ indices, float* result, int height, int width )
{
	int k;
	if( GetCudaTaskIndex( height, k ) ) {
		PRINT_HEAD2_F( k, 0, 0, "VectorTopKDiffKernel", source, /*indices,*/ result, height /*,width*/ );

		const int index = indices[k];
		result[k * width + index] = source[index];
		assert( isfinite( source[index] ) );
	}
}

__global__ void VectorNegKernel( const float* __restrict__ first,
	float* __restrict__ second, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		PRINT_HEAD2_F( index, 0, 0, "VectorNegKernel", first, second, vectorSize );
		second[index] = -first[index];
		assert( isfinite( second[index] ) );
	}
}

const int VectorLogDiffCombine = 16;
__global__ void VectorLogDiffKernel( const float* __restrict__ sourceGrad,
	int gradCount, int gradSize, int gradNorm,
	const float* __restrict__ first, float* resultGrad )
{
	int num;
	int index;
	if( !GetCudaTaskIndex2D( gradCount, gradNorm, num, index ) ) {
		return;
	}

	PRINT_HEAD3_F( index, num, 0, "VectorLogDiffKernel", first, sourceGrad, resultGrad, gradCount * gradNorm );

	const float div = first[num];
	const bool isCloseToZero = (-FLT_MIN <= div && div <= FLT_MIN);
	index *= VectorLogDiffCombine;
	sourceGrad += num * gradSize + index;
	resultGrad += num * gradSize + index;
	for( int i = index; i < min( gradSize, index + VectorLogDiffCombine ); i++ ) {
		if( isCloseToZero ) {
			*resultGrad = 0;
		} else {
			*resultGrad = *sourceGrad / div;
			assert( isfinite( *resultGrad ) );
			assert( *resultGrad > -BigFloatNumber );
			assert( *resultGrad <  BigFloatNumber );
		}
		resultGrad++;
		sourceGrad++;
	}
}

const int VectorAbsDiffCombine = 16;
__global__ void VectorAbsDiffKernel( const float* sourceGrad,
	int gradCount, int gradSize, int gradNorm,
	const float* __restrict__ first, float* resultGrad )
{
	int num;
	int index;
	if( !GetCudaTaskIndex2D( gradCount, gradNorm, num, index ) ) {
		return;
	}

	PRINT_HEAD3_F( index, num, 0, "VectorAbsDiffKernel", first, sourceGrad, resultGrad, gradCount * gradNorm );

	index *= VectorAbsDiffCombine;
	sourceGrad += num * gradSize + index;
	resultGrad += num * gradSize + index;
	for( int i = index; i < min( gradSize, index + VectorAbsDiffCombine ); i++ ) {
		if( first[num] > 0 ) {
			*resultGrad = *sourceGrad;
		} else {
			*resultGrad = -*sourceGrad;
		}
		resultGrad++;
		sourceGrad++;
	}
}

const int VectorMinMaxDiffCombine = 16;
__global__ void VectorMinMaxDiffKernel( const float* sourceGrad,
	int gradCount, int gradSize, int gradNorm,
	const float* __restrict__ first, float* resultGrad,
	const float* __restrict__ minPtr, const float* __restrict__ maxPtr )
{
	int num;
	int index;
	if( !GetCudaTaskIndex2D( gradCount, gradNorm, num, index ) ) {
		return;
	}

	PRINT_HEAD3_F( index, num, 0, "VectorMinMaxDiffKernel", first, sourceGrad, resultGrad, gradCount * gradNorm );

	const bool isOut = first[num] < *minPtr || first[num] > *maxPtr;
	index *= VectorMinMaxDiffCombine;
	sourceGrad += num * gradSize + index;
	resultGrad += num * gradSize + index;
	for( int i = index; i < min( gradSize, index + VectorMinMaxDiffCombine ); i++ ) {
		if( isOut ) {
			*resultGrad = 0;
		} else {
			*resultGrad = *sourceGrad;
			assert( isfinite( *resultGrad ) );
			assert( *resultGrad > -BigFloatNumber );
			assert( *resultGrad <  BigFloatNumber );
		}
		resultGrad++;
		sourceGrad++;
	}
}

const int VectorMaxCombineCount = 16;
__global__ void VectorMaxKernel( const float* __restrict__ first,
	float value, float* __restrict__ result, int count )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorMaxCombineCount, index, step );

	PRINT_HEAD3_F( index, 0, 0, "VectorMaxKernel", first, &value, result, count );

	first += index;
	result += index;

	for( int action = 0; action < actionCount; ++action ) {
		*result = ( *first >= value ) ? *first : value;

		if( !isfinite( *result ) ||
			*result < -BigFloatNumber ||
			*result >  BigFloatNumber )
		{
			printf( "VectorEltwiseMaxKernel: first=%f result=%f index=%d step=%d blockIdx.x=%u blockDim.x=%u threadIdx.x=%u \n",
				*first,*result, index, step, blockIdx.x, blockDim.x, threadIdx.x );
		}
		//assert( isfinite( *result ) );
		first += step;
		result += step;
	}
}

const int VectorMaxDiffCombineCount = 16;
__global__ void VectorMaxDiffKernel( float* grad, int gradCount, int gradSize, int gradNorm,
	const float* __restrict__ first, float secondValue )
{
	int num;
	int index;
	if( !GetCudaTaskIndex2D( gradCount, gradNorm, num, index ) || ( first[num] >= secondValue ) ) {
		return;
	}

	PRINT_HEAD3_F( index, num, 0, "VectorMaxDiffKernel", first, &secondValue, grad, gradCount * gradNorm );

	index *= VectorMinMaxDiffCombine;
	grad += num * gradSize + index;
	for( int i = index; i < min( gradSize, index + VectorMaxDiffCombineCount ); i++ ) {
		*grad++ = 0;
	}
}

} // namespace NeoML
