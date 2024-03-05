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
__global__ void VectorCopyKernel( T* result, const T* first, int count )
{
	int index;
	int step;
	int actionCount = GetCudaTaskCountAndIndex( count, VectorFillCombineCount, index, step );

	result += index;
	first += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = *first;
		if constexpr( std::is_same_v<T, float> ) {
			if( !isfinite( *result ) ||
				*result < -18002376725743890449408517795774411571.f ||
				*result > 18002376725743890449408517795774411571.f )
			{
				printf( "VectorCopyKernel: first=%f result=%f i=%d count=%d index=%d step=%d blockIdx.x=%u blockDim.x=%u threadIdx.x=%u \n",
					*first, *result, i, count, index, step, blockIdx.x, blockDim.x, threadIdx.x );
			}
			//assert( isfinite( *result ) );
			//assert( *result > -18002376725743890449408517795774411571.f );
			//assert( *result < 18002376725743890449408517795774411571.f );
		}
		result += step;
		first += step;
	}
}

template<class T>
__global__ void VectorFillKernel(T* mem, T value, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorFillCombineCount, index, step);

	mem += index;

	for(int i = 0; i < actionCount; ++i) {
		if constexpr( std::is_same_v<T, float> ) {
			if( !isfinite( *mem ) ||
				*mem < -18002376725743890449408517795774411571.f ||
				*mem >  18002376725743890449408517795774411571.f )
			{
				printf( "VectorFillKernel: mem=%f i=%d count=%d index=%d step=%d blockIdx.x=%u blockDim.x=%u threadIdx.x=%u \n",
					*mem, i, count, index, step, blockIdx.x, blockDim.x, threadIdx.x );
			}
			//------------
			*mem = value;
			//------------
			assert( isfinite( *mem ) );
			assert( *mem > -18002376725743890449408517795774411571.f );
			assert( *mem <  18002376725743890449408517795774411571.f );
		} else {
			*mem = value;
		}
		//*mem = value;
		mem += step;
	}
}

const int VectorFillHandleCombineCount = 8;
template<class T>
__global__ void VectorFillHandleKernel(T* mem, int count, const T* __restrict__ value)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorFillHandleCombineCount, index, step);

	mem += index;

	for(int i = 0; i < actionCount; ++i) {
		*mem = *value;
		if constexpr( std::is_same_v<T, float> ) {
			assert( isfinite( *mem ) );
			assert( *mem > -18002376725743890449408517795774411571.f );
			assert( *mem < 18002376725743890449408517795774411571.f );
		}
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

	from += index;
	to += index;

	for( int i = 0; i < actionCount; ++i ) {
		*to = static_cast<To>( *from );
		if constexpr( std::is_same_v<To, float> ) {
			assert( isfinite( *to ) );
			assert( *to > -18002376725743890449408517795774411571.f );
			assert( *to < 18002376725743890449408517795774411571.f );
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
				assert( *to > -18002376725743890449408517795774411571.f );
				assert( *to < 18002376725743890449408517795774411571.f );
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
				assert( result[j] > -18002376725743890449408517795774411571.f );
				assert( result[j] <  18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
	} else if(gridDim.x == 1) {
		*result += sum;
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
	} else {
		atomicAdd(result, sum);
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorSumAlongDimensionKernel( const float* __restrict__ input, int precedingDims, int dims,
	int followingDims, float* result )
{
	int x;
	int y;
	if( GetCudaTaskIndex2D( precedingDims, followingDims, x, y ) ) {
		input += y * dims * precedingDims + x;
		result += y * precedingDims + x;
		*result = 0;
		for( int i = 0; i < dims; i++ ) {
			*result += *input;
			assert( isfinite( *result ) );
			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
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
				assert( *result > -18002376725743890449408517795774411571.f );
				assert( *result < 18002376725743890449408517795774411571.f );
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
		const int width = precedingDims * dims * followingDims;
		const int startOffset = y * dims * precedingDims + x;
		input += startOffset;
		result += ( y * precedingDims + x ) * width + startOffset;
		for( int i = 0; i < dims; i++ ) {
			*result += *input;
			assert( isfinite( *result ) );
			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
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
		const int cumDim = y / followingDims;
		const int width = precedingDims * dims * followingDims;
		const int startOffset = ( y % followingDims ) * dims * precedingDims + x;
		input += startOffset;
		result += ( y * precedingDims + x ) * width + startOffset;
		for( int i = 0; i <= cumDim; i++ ) {
			*result += *input;
			assert( isfinite( *result ) );
			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
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

	first += index;
	result += index;

	for( int action = 0; action < actionCount; ++action ) {
		*result = ( *first >= 0 ) ? *first : ( *alpha * ( ExponentFunc( *first ) - 1. ) );
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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

	first += index;
	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = ( *first >= 0 ) ? *second : ( *second * ExponentFunc( *first ) * *alpha );
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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

	first += index;
	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = ( *first >= 0 ) ? *second : ( *second * ( *first + *alpha ) );
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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

	first += index;
	result += index;
	if(*threshold > 0) {
		for(int i = 0; i < actionCount; ++i) {
			const float value = min(*first, *threshold);
			assert( isfinite( value ) );
			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
			*result = value > 0 ? value : 0;
			first += step;
			result += step;
		}
	} else {
		for(int i = 0; i < actionCount; ++i) {
			const float value = *first;
			assert( isfinite( value ) );
			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
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

	first += index;
	second += index;
	result += index;

	if(*threshold > 0) {
		for(int i = 0; i < actionCount; ++i) {
			*result = (*first > 0 && *first < *threshold) ? *second : 0;
			assert( isfinite( *result ) );
			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
			first += step;
			second += step;
			result += step;
		}
	} else {
		for(int i = 0; i < actionCount; ++i) {
			*result = ( *first > 0 ) ? *second : 0;
			assert( isfinite( *result ) );
			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
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

	first += index;
	result += index;
	for( int i = 0; i < actionCount; ++i ) {
		const float value = *first;
		*result = ( value > 0 ) ? value : ( *alpha * value );
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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

	first += index;
	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = ( *first > 0 ) ? *second : ( *second * *alpha );
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
		first += step;
		second += step;
		result += step;
	}
}

const int VectorEltwiseMaxCombineCount = 8;
__global__ void VectorEltwiseMaxKernel(const float* first, const float* second,
	float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorEltwiseMaxCombineCount, index, step);

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		const float value1 = *first;
		const float value2 = *second;
		*result = ( value1 > value2 ) ? value1 : value2;

		if( !isfinite( *result ) ||
			*result < -18002376725743890449408517795774411571.f ||
			*result > 18002376725743890449408517795774411571.f ) {
			printf( "VectorEltwiseMaxKernel: first=%f second=%f result=%f i=%d index=%d step=%d blockIdx.x=%u blockDim.x=%u threadIdx.x=%u \n",
				*first, *second, *result, i, index, step, blockIdx.x, blockDim.x, threadIdx.x );
		}
		//assert( isfinite( *result ) );
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

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		const float value1 = *first;
		const float value2 = *second;
		*result = ( value1 < value2 ) ? value1 : value2;

		if( !isfinite( *result ) ||
			*result < -18002376725743890449408517795774411571.f ||
			*result > 18002376725743890449408517795774411571.f ) {
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

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		const float value = 1 - *first;
		*result = ( value > 0 ) ? value : 0;
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = ( *first < 1 ) ? -*second : 0;
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
		first += step;
		result += step;
	}
}

__global__ void VectorHardTanhKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorActivationCombineCount, index, step);

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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorExpKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		result[index] = ExponentFunc(first[index]);
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorLogKernel( const float* __restrict__ first, float* result, int count )
{
	int index;
	if( GetCudaTaskIndex( count, index ) ) {
		result[index] = logf(min(max(first[index], FLT_MIN), FLT_MAX));
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorNegLogKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		result[index] = -logf(min(max(first[index], FLT_MIN), FLT_MAX));
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorErfKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		result[index] = erff(first[index]);
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorBernulliKLDerivativeKernel(const float* __restrict__ first,
	float* result, int count, const float* __restrict__ target)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		const float value = first[index];
		float klDer = -*target / value + (1 - *target) / (1 - value);
		if(klDer < -10) {
			klDer = -10;
		} else if(klDer > 10) {
			klDer = 10;
		}
		result[index] = klDer;
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

const int VectorAddCombineCount = 8;
template<class T>
__global__ void VectorAddKernel(const T* __restrict__ first,
	const T* __restrict__ second, T* result, int count)
{
	int index = 0;
	int step = 0;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorAddCombineCount, index, step);

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = *first + *second;

		if constexpr( std::is_same_v<T, float> ) {
			if( !isfinite( *second ) || !isfinite( *first ) || !isfinite( *result ) ||
				*result < -18002376725743890449408517795774411571.f ||
				*result > 18002376725743890449408517795774411571.f ) {
				printf( "VectorAddKernel: first=%f second=%f result=%f i=%d count=%d index=%d step=%d blockIdx.x=%u blockDim.x=%u threadIdx.x=%u \n",
					*first, *second, *result, i, count, index, step, blockIdx.x, blockDim.x, threadIdx.x );
			}
			assert( !isnan( *first ) );
			assert( !isnan( *second ) );
			assert( !isnan( *result ) );
		}
		first += step;
		second += step;
		result += step;
	}
}

const int VectorAddValueCombineCount = 8;
template<class T>
__global__ void VectorAddValueKernel(
	const T* __restrict__ first, T* result, int count, const T* __restrict__ addition )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorAddValueCombineCount, index, step);

	first += index;
	result += index;
	if constexpr( std::is_same_v<T, float> ) {
		assert( isfinite( *addition ) );
	}

	for(int i = 0; i < actionCount; ++i) {
		*result = *first + *addition;

		if constexpr( std::is_same_v<T, float> ) {
			if( !isfinite( *first ) || !isfinite( *result ) ) {
				printf( "VectorAddValueKernel: first=%f result=%f i=%d index=%d step=%d\n", *first, *result, i, index, step );
			}
			assert( isfinite( *first ) );
			assert( isfinite( *result ) );
			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
		}
		first += step;
		result += step;
	}
}

const int VectorSubCombineCount = 8;
template<class T>
__global__ void VectorSubKernel( const T* __restrict__ first, const T* __restrict__ second, T* result, int count )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorSubCombineCount, index, step );

	first += index;
	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = *first - *second;
		if constexpr( std::is_same_v<T, float> ) {
			assert( isfinite( *result ) );
			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
		}
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorSubKernel( const float* __restrict__ first,
	float second, float* result, int count )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorSubCombineCount, index, step );

	first += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = *first - second;
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
		first += step;
		result += step;
	}
}

__global__ void VectorSubKernel( float first,
	const float* __restrict__ second, float* result, int count )
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex( count, VectorSubCombineCount, index, step );

	second += index;
	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = first - *second;
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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
		result[index] = first[index] - *mult * second[index];
		assert( isfinite( result[index] ) );

		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

const int VectorMultiplyCombineCount = 8;
template<class T>
__global__ void VectorMultiplyKernel(const T* __restrict__ first,
	T* result, int count, const T* __restrict__ multiplier)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorMultiplyCombineCount, index, step);

	first += index;
	result += index;
	if constexpr( std::is_same_v<T, float> ) {
		assert( isfinite( *multiplier ) );
	}

	for(int i = 0; i < actionCount; ++i) {
		*result = *first * (*multiplier);

		if constexpr( std::is_same_v<T, float> ) {
			if( !isfinite( *first ) || !isfinite( *result ) ) {
				printf( "VectorMultiplyKernel: first=%f result=%f i=%d index=%d step=%d\n", *first, *result, i, index, step );
			}
			assert( isfinite( *first ) );
			assert( isfinite( *result ) );

			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
		}
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

	first += index;
	result += index;

	const float mul = -(*multiplier);
	assert( isfinite( mul ) );

	for(int i = 0; i < actionCount; ++i) {
		*result = *first * mul;
		assert( isfinite( *result ) && isfinite( *first ) );

		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
		first += step;
		result += step;
	}
}

const int VectorEltwiseMultiplyCombineCount = 8;
template<class T>
__global__ void VectorEltwiseMultiplyKernel(const T* __restrict__ first,
	const T* __restrict__ second, T* result, int count)
{
	int index = 0;
	int step = 0;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorEltwiseMultiplyCombineCount, index, step);

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = *first * (*second);

		if constexpr( std::is_same_v<T, float> ) {
			if( !isfinite( *second ) || !isfinite( *first ) || !isfinite( *result ) ||
				*first < -18002376725743890449408517795774411571.f ||
				*first >  18002376725743890449408517795774411571.f ) {

				printf( "VectorEltwiseMultiplyKernel: first=%f second=%f result=%f i=%d index=%d count=%d step=%d blockIdx.x=%u blockDim.x=%u threadIdx.x=%u \n",
					   /* blockCount=%d threadCount=%d */
					*first, *second, *result, i, index, count, step, blockIdx.x, blockDim.x, threadIdx.x/*, blockCount, threadCount*/ );
			}
			assert( isfinite( *first ) );
			assert( isfinite( *second ) );
			assert( isfinite( *result ) );
		}
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

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result += *first * (*second);

		if( !isfinite( *second ) || !isfinite( *first ) || !isfinite( *result ) ) {
			printf( "VectorEltwiseMultiplyAddKernel: first=%f second=%f result=%f i=%d index=%d step=%d\n", *first, *second, *result, i, index, step );
		}
		assert( isfinite( *first ) );
		assert( isfinite( *second ) );
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
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

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = - *first * (*second);

		if( !isfinite( *second ) || !isfinite( *first ) || !isfinite( *result ) ) {
			printf( "VectorEltwiseNegMultiplyKernel: first=%f second=%f result=%f i=%d index=%d step=%d\n", *first, *second, *result, i, index, step );
		}
		assert( isfinite( *first ) );
		assert( isfinite( *second ) );
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
		first += step;
		second += step;
		result += step;
	}
}

const int VectorEltwiseDivideCombineCount = 8;
template<class T>
__global__ void VectorEltwiseDivideKernel(const T* __restrict__ first,
	const T* __restrict__ second, T* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorEltwiseDivideCombineCount, index, step);

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = *first / (*second);

		if constexpr( std::is_same_v<T, float> ) {
			if( !isfinite( *second ) || !isfinite( *first ) ) {
				printf( "VectorEltwiseDivideKernel: first=%f second=%f result=%f i=%d index=%d step=%d\n", *first, *second, *result, i, index, step );
			}
			//assert( isfinite( *first ) );
			//assert( isfinite( *second ) );
			assert( isfinite( *result ) );
			assert( *result > -18002376725743890449408517795774411571.f );
			assert( *result < 18002376725743890449408517795774411571.f );
		}
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
		result[index] = (second[index] == 1) ? first[index] : powf(first[index], second[index]);

		if( !isfinite( second[index] ) || !isfinite( first[index] ) ) {
			printf( "VectorEltwisePowerKernel: first=%f second=%f result=%f index=%d \n", first[index], second[index], result[index], index );
		}
		assert( isfinite( first[index] ) );
		assert( isfinite( second[index] ) );
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorSqrtKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		result[index] = sqrtf(first[index]);
		if( !isfinite( *result ) ) {
			printf( "VectorSqrtKernel: first=%f result=%f index=%d \n", *first, *result, index );
		}
	}
}

const int VectorInvCombineCount = 8;
__global__ void VectorInvKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorInvCombineCount, index, step);

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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
		first += step;
		result += step;
	}
}

const int VectorMinMaxCombineCount = 8;
__global__ void VectorMinMaxKernel(const float* __restrict__ first, float* result, int count,
	const float* __restrict__ minValue, const float* __restrict__ maxValue)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(count, VectorMinMaxCombineCount, index, step);

	first += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = min(max(*first, *minValue), *maxValue);
		if( !isfinite( *result ) ||
			*result < -18002376725743890449408517795774411571.f ||
			*result >  18002376725743890449408517795774411571.f ) {
			printf( "VectorMinMaxKernel: first=%f result=%f minValue=%f maxValue=%f i=%d index=%d step=%d blockIdx.x=%u blockDim.x=%u threadIdx.x=%u \n",
				*first, *result, *minValue, *maxValue, i, index, step, blockIdx.x, blockDim.x, threadIdx.x );
		}
		first += step;
		result += step;
	}
}

__global__ void VectorSigmoidKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		assert( isfinite( first[index] ) );
		result[index] = 1.f / (1.f + ExponentFunc(-first[index]));

		if( !isfinite( first[index] ) || !isfinite( result[index] ) ) {
			printf( "VectorSigmoidKernel: first=%f result=%f index=%d \n", first[index], result[index], index );
		}
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorSigmoidDiffKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		assert(isfinite(first[index]));
		const float expVal = ExponentFunc(-first[index]);
		const float expVal1 = expVal + 1.f;
		result[index] = expVal / expVal1 / expVal1;
		result[index] *= second[index];

		if( !isfinite( second[index] ) || !isfinite( first[index] ) || !isfinite( result[index] ) ) {
			printf( "VectorSigmoidDiffKernel: first=%f second=%f result=%f index=%d \n", first[index], second[index], result[index], index );
		}
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

const int VectorSigmoidDiffOpCombineCount = 4;
__global__ void VectorSigmoidDiffOpKernel(const float* __restrict__ first, const float* __restrict__ second,
	float* result, int vectorSize)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(vectorSize, VectorSigmoidDiffOpCombineCount, index, step);

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		*result = *first * (1.f - *first) * *second;

		if( !isfinite( *second ) || !isfinite( *first ) || !isfinite( *result ) ) {
			printf( "VectorSigmoidDiffOpKernel: first=%f second=%f result=%f i=%d index=%d step=%d\n", *first, *second, *result, i, index, step );
		}
		assert( isfinite( *first ) );
		assert( isfinite( *second ) );
		assert( isfinite( *result ) );
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorTanhKernel(const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		result[index] = -1.f  + 2 / (1.f + ExponentFunc(-2 * first[index]));
		if( !isfinite( *result ) || !isfinite( *first ) ) {
			printf( "VectorTanhKernel: first=%f result=%f index=%d \n", first[index], result[index], index );
		}
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorTanhDiffKernel(const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		float tanh = -1.f  + 2 / (1.f + ExponentFunc(-2 * first[index]));
		result[index] = second[index] * (1.f - tanh * tanh);
		if( !isfinite( *result ) || !isfinite( *first ) || !isfinite( *second ) ) {
			printf( "VectorTanhDiffKernel: first=%f second=%f result=%f index=%d \n", first[index], second[index], result[index], index );
		}
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

const int VectorTanhDiffOpCombineCount = 4;
__global__ void VectorTanhDiffOpKernel(const float* __restrict__ first, const float* __restrict__ second,
	float* result, int vectorSize)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(vectorSize, VectorTanhDiffOpCombineCount, index, step);

	first += index;
	second += index;
	result += index;

	for(int i = 0; i < actionCount; ++i) {
		assert( isfinite( *first ) );
		assert( isfinite( *second ) );
		*result = (1.f - *first * *first) * *second;

		if( !isfinite( *second ) || !isfinite( *first ) || !isfinite( *result ) 
			|| *result < -18002376725743890449408517795774411571.f
			|| *result > 18002376725743890449408517795774411571.f
			) {
			printf( "VectorTanhDiffOpKernel: first=%f second=%f result=%f i=%d index=%d step=%d\n", *first, *second, *result, i, index, step );
		}
		assert( isfinite( *result ) );
		first += step;
		second += step;
		result += step;
	}
}

__global__ void VectorPowerKernel(float exponent, const float* __restrict__ first, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		result[index] = powf(first[index], exponent);
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorPowerDiffKernel(float exponent, const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		result[index] = second[index] * exponent * powf(first[index], exponent - 1);
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorPowerDiffOpKernel(float exponent, const float* __restrict__ first,
	const float* __restrict__ second, float* result, int count)
{
	int index;
	if(GetCudaTaskIndex(count, index)) {
		result[index] = second[index] * exponent * powf(first[index], (exponent - 1.f) / exponent);
		assert( isfinite( result[index] ) );
		assert( result[index] > -18002376725743890449408517795774411571.f );
		assert( result[index] < 18002376725743890449408517795774411571.f );
	}
}

__global__ void VectorL1DiffAddKernel(const float* __restrict__ first, const float* __restrict__ second,
	float* result, int vectorSize, const float* __restrict__ threshold, const float* __restrict__ mult)
{
	int index;
	int step;
	const int actionCount = GetCudaTaskCountAndIndex(vectorSize, VectorActivationCombineCount, index, step);

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
		assert( *result > -18002376725743890449408517795774411571.f );
		assert( *result < 18002376725743890449408517795774411571.f );

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
		result[index] = first[index] == 0 ? 1 : 0;
	}
}

__global__ void vectorGreaterEqualToZeroKernel( const int* __restrict__ first,
	float* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		result[index] = first[index] >= 0 ? 1.f : 0.f;
	}
}

template<class TSrc, class TDst>
__global__ void vectorLessKernel( const TSrc* __restrict__ first, const TSrc* __restrict__ second,
	TDst* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		result[index] = static_cast<TDst>( first[index] < second[index] ? 1 : 0 );
	}
}

__global__ void vectorLessKernel( const float* __restrict__ first, float second,
	float* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		result[index] = first[index] < second ? 1.f : 0.f;
	}
}

__global__ void vectorLessKernel( float first, const float* __restrict__ second,
	float* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		result[index] = first < second[index] ? 1.f : 0.f;
	}
}

template<class T>
__global__ void vectorEqualKernel( const T* __restrict__ first, const T* __restrict__ second,
	int* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		result[index] = first[index] == second[index] ? 1 : 0;
	}
}

template<class T>
__global__ void vectorWhereKernel( const int* __restrict__ first, const T* __restrict__ second,
	const T* __restrict__ third, T* result, int vectorSize )
{
	int index;
	if( GetCudaTaskIndex( vectorSize, index ) ) {
		result[index] = first[index] != 0 ? second[index] : third[index];
	}
}

__global__ void VectorFindMaxValueInSetKernel( CCudaConstVectorArray vectors,
	float* result, int vectorSize)
{
	int index;
	if(GetCudaTaskIndex(vectorSize, index)) {
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

	const float div = first[num];
	const bool isCloseToZero = ( -FLT_MIN <= div && div <= FLT_MIN );
	index *= VectorLogDiffCombine;
	sourceGrad += num * gradSize + index;
	resultGrad += num * gradSize + index;
	for( int i = index; i < min( gradSize, index + VectorLogDiffCombine ); i++ ) {
		if( isCloseToZero ) {
			*resultGrad = 0;
		} else {
			*resultGrad = *sourceGrad / div;
			assert( isfinite( *resultGrad ) );
			assert( *resultGrad > -18002376725743890449408517795774411571.f );
			assert( *resultGrad < 18002376725743890449408517795774411571.f );
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
			assert( *resultGrad > -18002376725743890449408517795774411571.f );
			assert( *resultGrad <  18002376725743890449408517795774411571.f );
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

	first += index;
	result += index;

	for( int action = 0; action < actionCount; ++action ) {
		*result = ( *first >= value ) ? *first : value;

		if( !isfinite( *result ) ||
			*result < -18002376725743890449408517795774411571.f ||
			*result > 18002376725743890449408517795774411571.f )
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

	index *= VectorMinMaxDiffCombine;
	grad += num * gradSize + index;
	for( int i = index; i < min( gradSize, index + VectorMaxDiffCombineCount ); i++ ) {
		*grad++ = 0;
	}
}

} // namespace NeoML
