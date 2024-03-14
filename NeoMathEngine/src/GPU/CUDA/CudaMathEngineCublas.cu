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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <CudaMathEngine.h>
#include <CudaCommon.h>
#include <CudaAssert.h>
#include <CudaDevice.h>
#include <CublasFunctions.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>
#include <Kernels/CudaGrid.h>
#include <cmath>

#include <cuda_runtime_api.h>

namespace NeoML {

const int VectorSDotCombineCount = 8;

//---------------------------------------------------------------------------------------------------------------------

__global__ void VectorRoundKernel( float* result, int count )
{
	assert( threadIdx.y == 0 );
	assert( threadIdx.z == 0 );

	int index;
	int step;
	int actionCount = GetCudaTaskCountAndIndex( count, VectorSDotCombineCount, index, step );

	result += index;

	for( int i = 0; i < actionCount; ++i ) {
		*result = roundf( *result );
		result += step;
	}
}

void CCudaMathEngine::vectorRound( const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorSDotCombineCount );

	VectorRoundKernel<<<blockCount, threadCount>>>( GetRaw( resultHandle ), vectorSize );
}

//---------------------------------------------------------------------------------------------------------------------

__global__ void VectorNumerateKernel( const float* first, const float* second, float* result,
	int firstSize, int secondSize, int resultSize, int num, size_t calls_counter, void* historyKernels, int id )
{
	assert( threadIdx.y == 0 );
	assert( threadIdx.z == 0 );

	int index;
	int step;
	int actionCount = GetCudaTaskCountAndIndex( firstSize + secondSize + resultSize, VectorSDotCombineCount, index, step );
	PRINT_HEAD3_CNT_F( index, 0, 0, "VectorNumerateKernel", first, second, result, resultSize, calls_counter, historyKernels, id );

	int shift = index;
	for( int i = 0; i < actionCount; ++i ) {
		if( shift < firstSize ) {
			WARN3_CNT_NORES_F( "VectorNumerateKernel first", first[shift], first, firstSize, id, shift, num, calls_counter, historyKernels );
		} else if( shift < ( firstSize + secondSize ) ) {
			WARN3_CNT_NORES_F( "VectorNumerateKernel second", second[shift - firstSize], second, secondSize, id, shift, num, calls_counter, historyKernels );
		} else {
			WARN3_CNT_SPEC_F( "VectorNumerateKernel", /*1*/0.f, 0, /*2*/0.f, 0, result[shift - ( firstSize + secondSize )], result, resultSize,
				id, shift, num, /*name*/( char* )0, calls_counter, historyKernels, id );
		}
		shift += step;
	}
}

void CCudaMathEngine::vectorNumerate( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int firstSize, int secondSize, int resultSize, int num, size_t calls_counter, void* historyKernels, int id )
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, firstSize + secondSize + resultSize, VectorSDotCombineCount );

	VectorNumerateKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ),
		firstSize, secondSize, resultSize, num, calls_counter, historyKernels, id );
}

//---------------------------------------------------------------------------------------------------------------------

__global__ void VectorSDotKernel( const float* first, const float* second, float* result, int count, size_t calls_counter, void* historyKernels )
{
	assert( threadIdx.y == 0 );
	assert( threadIdx.z == 0 );

	int index;
	int step;
	int actionCount = GetCudaTaskCountAndIndex( count, VectorSDotCombineCount, index, step );
	PRINT_HEAD3_CNT_F( index, 0, 0, "VectorSDotKernel", first, second, result, count, calls_counter, historyKernels, VectorSDotKernelId );

	unsigned tid = threadIdx.x;
	extern __shared__ double buffer[];

	first += index;
	second += index;

	if( actionCount > 0 ) {
		double sum = 0;
		for( int i = 0; i < actionCount; ++i ) {
			sum = std::fma( ( double )*first, ( double )*second, sum );

			assert( isfinite( sum ) );
			assert( sum > -18002376725743890449408517795774411571.f );
			assert( sum < 18002376725743890449408517795774411571.f );

			first += step;
			second += step;
		}
		buffer[tid] = sum;
	} else {
		buffer[tid] = 0;
	}

	__syncthreads();
	if( tid == 0 ) {
		double sum = 0;
		for( int i = 0; i < blockDim.x; ++i ) {
			sum = std::fma( buffer[i], 1., sum );
		}
		atomicAdd( result, ( float )sum );
		WARN3_CNT_F( "VectorSDotKernel", *first, first, *second, second, *result, result, count, tid, index, calls_counter, historyKernels );
	}
}

//---------------------------------------------------------------------------------------------------------------------

extern size_t calls_counter;

void CCudaMathEngine::VectorDotProduct(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	int vectorSize, const CFloatHandle& resultHandle)
{
	// printf( "VectorDotProduct \n" ); // !!! HAVE !!!
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	//VectorFill( resultHandle, 0.f, 1, 81 );

	int blockCount;
	int threadCount;
	getCudaTaskGrid( blockCount, threadCount, vectorSize, VectorSDotCombineCount );
	
	//VectorSDotKernel<<<blockCount, threadCount>>>( GetRaw( firstHandle ), GetRaw( secondHandle ),
	//	GetRaw( resultHandle ), vectorSize, ++calls_counter, GetRaw(historyKernels) );

	ASSERT_CUBLAS( cublas->Sdot( cublasHandle, vectorSize, GetRaw( firstHandle ), 1,
		GetRaw( secondHandle ), 1, GetRaw( resultHandle ) ) );

	//vectorRound( resultHandle, 1 );

	vectorNumerate( firstHandle, secondHandle, resultHandle,
		vectorSize, vectorSize, 1, /*num*/0, ++calls_counter, GetRaw( historyKernels ), VectorSDotKernelId );
}

void CCudaMathEngine::VectorMultiplyAndAdd( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle, int num )
{
	// printf( "VectorMultiplyAndAdd \n" ); // !!! HAVE !!!
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const float* const first = GetRaw( firstHandle );
	const float* const second = GetRaw( secondHandle );
	float* const result = GetRaw( resultHandle );
	const float* const mult = GetRaw( multHandle );

	if( result != first ) {
		ASSERT_CUDA( cudaMemcpy( result, first, vectorSize * sizeof( float ), cudaMemcpyDeviceToDevice ) );
	}
	ASSERT_CUBLAS( cublas->Saxpy( cublasHandle, vectorSize, mult, second, 1, result, 1 ) );

	assert( num > 0 );
	vectorNumerate( firstHandle, secondHandle, resultHandle,
		vectorSize, vectorSize, vectorSize, num, ++calls_counter, GetRaw( historyKernels ), VectorMultiplyAndAddKernelId );
}

void CCudaMathEngine::MultiplyMatrixByTransposedMatrix( const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize, int, int num )
{
	// printf( "MultiplyMatrixByTransposedMatrix1 \n" ); // !!! HAVE !!!
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	ASSERT_CUBLAS( cublas->Sgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, secondHeight, firstHeight, firstWidth,
		cudaConstOne, GetRaw( secondHandle ), secondRowSize, GetRaw( firstHandle ), firstRowSize, cudaConstZero,
		GetRaw( resultHandle ), resultRowSize ) );

	assert( num > 0 );
	assert( resultRowSize == secondHeight );
	assert( secondRowSize == firstWidth );
	vectorNumerate( firstHandle, secondHandle, resultHandle,
		firstHeight * firstWidth, firstWidth * secondHeight, firstHeight * secondHeight,
		num, ++calls_counter, GetRaw(historyKernels), MultiplyMatrixByTransposedMatrix1KernelId );
}

void CCudaMathEngine::MultiplyMatrixByTransposedMatrix( int batchSize, const CConstFloatHandle& firstHandle,
	int firstHeight, int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight,
	const CFloatHandle& resultHandle, int )
{
	printf( "MultiplyMatrixByTransposedMatrix2 \n" );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	ASSERT_CUBLAS( cublas->SgemmStridedBatched( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, secondHeight,
		firstHeight, firstWidth, cudaConstOne, GetRaw( secondHandle ), firstWidth, firstWidth * secondHeight,
		GetRaw( firstHandle ), firstWidth, firstHeight * firstWidth, cudaConstZero, GetRaw( resultHandle ),
		secondHeight, secondHeight * firstHeight, batchSize ) );
}

void CCudaMathEngine::MultiplyTransposedMatrixByMatrixAndAdd( const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize, int )
{
	//printf( "MultiplyTransposedMatrixByMatrixAndAdd \n" ); // !!! HAVE !!!
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	ASSERT_CUBLAS( cublas->Sgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, secondWidth, firstWidth, firstHeight,
		cudaConstOne, GetRaw( secondHandle ), secondRowSize, GetRaw( firstHandle ), firstRowSize, cudaConstOne,
		GetRaw( resultHandle ), resultRowSize ) );

	assert( secondRowSize == secondWidth );
	assert( resultRowSize == secondWidth );
	vectorNumerate( firstHandle, secondHandle, resultHandle,
		firstHeight * firstWidth, firstHeight * secondWidth, firstWidth * secondWidth,
		/*num*/0, ++calls_counter, GetRaw( historyKernels ), MultiplyTransposedMatrixByMatrixAndAddKernelId );
}

void CCudaMathEngine::MultiplyTransposedMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int )
{
	printf( "MultiplyTransposedMatrixByMatrix \n" );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	ASSERT_CUBLAS( cublas->SgemmStridedBatched( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, secondWidth, firstWidth,
		firstHeight, cudaConstOne, GetRaw(secondHandle), secondWidth, firstHeight * secondWidth, GetRaw(firstHandle),
		firstWidth, firstHeight * firstWidth, cudaConstZero, GetRaw(resultHandle), secondWidth, firstWidth * secondWidth,
		batchSize ) );
}

void CCudaMathEngine::MultiplyMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int )
{
	// printf( "MultiplyMatrixByMatrix \n" ); // !!! HAVE !!!
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	if( batchSize == 1 ) {
		ASSERT_CUBLAS( cublas->Sgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, secondWidth, firstHeight, firstWidth,
			cudaConstOne, GetRaw( secondHandle ), secondWidth, GetRaw( firstHandle ), firstWidth, cudaConstZero,
			GetRaw( resultHandle ), secondWidth ) );
	} else {
		ASSERT_CUBLAS( cublas->SgemmStridedBatched( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, secondWidth, firstHeight, firstWidth,
			cudaConstOne, GetRaw( secondHandle ), secondWidth, firstWidth * secondWidth, GetRaw( firstHandle ), firstWidth,
			firstHeight * firstWidth, cudaConstZero, GetRaw( resultHandle ), secondWidth, secondWidth * firstHeight, batchSize ) );
	}

	vectorNumerate( firstHandle, secondHandle, resultHandle,
		batchSize * firstHeight * firstWidth, batchSize * firstWidth * secondWidth, batchSize * firstHeight * secondWidth,
		/*num*/0, ++calls_counter, GetRaw( historyKernels ), MultiplyMatrixByMatrixKernelId );
}

void CCudaMathEngine::multiplyMatrixByTransposedMatrixAndAdd(const CConstFloatHandle& firstHandle,
	int firstHeight, int firstWidth, int firstRowSize,
	const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize)
{
	printf( "multiplyMatrixByTransposedMatrixAndAdd \n" );
	SetCudaDevice( device->DeviceNumber );
	ASSERT_CUBLAS( cublas->Sgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, secondHeight, firstHeight, firstWidth,
		cudaConstOne, GetRaw( secondHandle ), secondRowSize, GetRaw( firstHandle ), firstRowSize, cudaConstOne,
		GetRaw( resultHandle ), resultRowSize ) );
}

void CCudaMathEngine::BatchMultiplyMatrixByDiagMatrix( int batchSize, const CConstFloatHandle& firstHandle, int height,
	int width, int firstMatrixOffset, const CConstFloatHandle& secondHandle, int secondMatrixOffset,
	const CFloatHandle& resultHandle, int )
{
	printf( "BatchMultiplyMatrixByDiagMatrix \n" );
	if( height == 1 && batchSize == 1 ) {
		VectorEltwiseMultiply( firstHandle, secondHandle, resultHandle, width );
		return;
	} else if( width == 1 && batchSize == 1 ) {
		VectorMultiply( firstHandle, resultHandle, height, secondHandle );
		return;
	}

	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	if( batchSize == 1 ) {
		SetCudaDevice( device->DeviceNumber );
		ASSERT_CUBLAS( cublas->Sdgmm( cublasHandle, CUBLAS_SIDE_LEFT, width, height, GetRaw( firstHandle ), width,
			GetRaw( secondHandle ), 1, GetRaw( resultHandle ), width ) );
		return;
	}

	multiplyMatrixByDiagMatrix( batchSize, firstHandle, height, width, firstMatrixOffset, secondHandle,
		secondMatrixOffset, resultHandle );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
