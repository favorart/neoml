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
#include <Kernels/CudaRandom.h>

namespace NeoML {

__global__ void RandomMatrixDropout( const float* first, int firstHeight,
	int firstWidth, float* res, int seed, float forwardRate, size_t calls_counter, void* historyKernels )
{
	const unsigned int threshold = forwardRate * UINT_MAX;
	int row;
	int col;
	if( GetCudaTaskIndex2D( firstHeight, ( firstWidth + 3 ) / 4, row, col ) ) {
		PRINT_HEAD2_CNT_F( row, col, 0, "RandomMatrixDropout", first, res, firstHeight, calls_counter, historyKernels, RandomMatrixDropoutKernelId );

		CCudaRandom random(seed);
		random.Skip(col);
		col *= 4;
		assert( col < firstWidth );
		const int index = row * firstWidth + col;

		CIntArray<4> generated = random.Next();
		for(int j = 0; j < 4 && ( col + j ) < firstWidth; ++j) {
			float result = res[index + j] = (generated[j] <= threshold) ? (first[index + j] / forwardRate) : 0.f;

			WARN2_CNT_F( "RandomMatrixDropout", first[index + j], first, result, res, firstWidth, firstHeight, (index + j), calls_counter, historyKernels );
		}
	}
}

__global__ void RandomSpatialDropout( const float* input, float* res, int inputObjectCount,
	int inputObjectSize, int maskObjectCount, int maskObjectSize, int seed, float forwardRate )
{
	const unsigned int threshold = forwardRate * UINT_MAX;
	int obj;
	int row;
	int col;
	if( GetCudaTaskIndex3D( inputObjectCount, inputObjectSize / maskObjectSize, maskObjectSize, obj, row, col ) ) {
		PRINT_HEAD2_F( obj, row, col, "RandomSpatialDropout", input, res, inputObjectCount );

		const int pack = obj % maskObjectCount;
		const int index = obj * inputObjectSize + row * maskObjectSize + col;
		const int numBlock = ( pack * maskObjectSize + col ) / 4;
		const int numLeft = ( pack * maskObjectSize + col ) % 4;
		CCudaRandom random(seed);
		random.Skip(numBlock);

		CIntArray<4> generated = random.Next();
		res[index] = (generated[numLeft] <= threshold) ? (input[index] / forwardRate) : 0.f;
	}
}

} // namespace NeoML
