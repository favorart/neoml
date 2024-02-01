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

#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <float.h>

namespace NeoML {

// Reduce for parallel calculations
// Adds up the common buffer over the X coordinate for each pair of Y, Z, puts the result into threadIdx.x == 0
// The function should be called on all threads
// When calling, the X size should first be aligned using alignXSizeForWarp
inline __device__ float ReduceSumXSharedBuffer(float* buffer)
{
	// The part of the warp that belongs to the same convolution
	int xWarp = warpSize;
	while(blockDim.x < xWarp) {
		xWarp >>= 1;
	}

	if(blockDim.x > xWarp) {
		// Take into account other threads and warps when calculating the final total
		__syncthreads();
	}

	// Calculate the partial sum over each warp
	float sum = 0;
	int indexInWarp = threadIdx.x % xWarp;
	int baseIndex = (threadIdx.z * blockDim.y +  threadIdx.y) * blockDim.x + indexInWarp;
	for(int i = 0; (indexInWarp + i) < blockDim.x; i += xWarp) {
		sum += buffer[baseIndex + i];
		assert( isfinite( buffer[baseIndex + i] ) );
	}
	assert( isfinite( sum ) );

	// Add up inside the warp (butterfly reduction)
	for(int laneMask = xWarp >> 1; laneMask >= 1; laneMask >>= 1) {
		const float res = __shfl_xor_sync( 0xffffffff, sum, laneMask );
		if( !isfinite( sum ) || !isfinite( res ) /*|| !isfinite( sum + res )*/ ) {
			printf( "ReduceSumXSharedBuffer: sum=%f res=%f laneMask=%d xWarp=%d threadIdx.x=%d threadIdx.y=%d threadIdx.z=%d blockDim.x=%d blockDim.y=%d\n",
				sum, res, laneMask, xWarp, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y );
			assert( !isnan( sum ) && !isnan( res ) );
		}
		sum += res;
	}
	assert( isfinite( sum ) );
	assert( sum > -18002376725743890449408517795774411571.f );
	assert( sum < 18002376725743890449408517795774411571.f );

	return sum;
}

inline __device__ float ReduceMaxXSharedBuffer(float* buffer)
{
	// The part of the warp that belongs to the same convolution
	int xWarp = warpSize;
	while(blockDim.x < xWarp) {
		xWarp >>= 1;
	}

	if(blockDim.x > xWarp) {
		// Take into account other threads and warps when calculating the final result
		__syncthreads();
	}

	// Calculate the maximum over each warp
	int indexInWarp = threadIdx.x % xWarp;
	int baseIndex = (threadIdx.z * blockDim.y +  threadIdx.y) * blockDim.x + indexInWarp;
	float maxVal = buffer[baseIndex];
	assert( isfinite( buffer[baseIndex] ) );
	for(int i = xWarp; (indexInWarp + i) < blockDim.x; i += xWarp) {
		assert( isfinite( buffer[baseIndex + i] ) );
		if(buffer[baseIndex + i] > maxVal) {
			maxVal = buffer[baseIndex + i];
		}
	}
	assert( isfinite( maxVal ) );

	// Find maximum inside the warp (butterfly reduction)
	for(int laneMask = xWarp >> 1; laneMask >= 1; laneMask >>= 1) {
		float otherVal = __shfl_xor_sync(0xffffffff, maxVal, laneMask);
		assert( isfinite( otherVal ) );
		if(otherVal > maxVal) {
			maxVal = otherVal;
		}
	}
	assert( isfinite( maxVal ) );

	return maxVal;
}

struct CValueWithIndex {
	float Value;
	int Index;
};

inline __device__ CValueWithIndex ReduceMaxWithIndexXSharedBuffer(CValueWithIndex* buffer)
{
	// The part of the warp that belongs to the same convolution
	int xWarp = warpSize;
	while(blockDim.x < xWarp) {
		xWarp >>= 1;
	}

	if(blockDim.x > xWarp) {
		// Take into account other threads and warps when calculating the final result
		__syncthreads();
	}

	// Calculate the maximum over each warp
	int indexInWarp = threadIdx.x % xWarp;
	int baseIndex = (threadIdx.z * blockDim.y +  threadIdx.y) * blockDim.x + indexInWarp;
	CValueWithIndex maxVal = buffer[baseIndex];
	assert( isfinite( buffer[baseIndex].Value ) );
	for(int i = xWarp; (indexInWarp + i) < blockDim.x; i += xWarp) {
		assert( isfinite( buffer[baseIndex + i].Value ) );
		if(buffer[baseIndex + i].Value > maxVal.Value) {
			maxVal = buffer[baseIndex + i];
		}
	}
	assert( isfinite( maxVal.Value ) );

	// Find maximum inside the warp (butterfly reduction)
	for(int laneMask = xWarp >> 1; laneMask >= 1; laneMask >>= 1) {
		long long maxValWarp = reinterpret_cast<const long long&>(maxVal);
		long long otherValWarp = __shfl_xor_sync(0xffffffff, maxValWarp, laneMask);
		const CValueWithIndex& otherVal = reinterpret_cast<const CValueWithIndex&>(otherValWarp);
		assert( isfinite( otherVal.Value ) );
		if(otherVal.Value > maxVal.Value) {
			maxVal = otherVal;
		}
	}
	assert( isfinite( maxVal.Value ) );

	return maxVal;
}

} // namespace NeoML
