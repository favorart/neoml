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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace NeoML {

static constexpr int CudaHistoryKernelsSize = 1000;
struct CCudaHistoryKernel {
	size_t counter;
	int kernel;
	unsigned long long res_addr;
	unsigned long long fst_addr;
	unsigned long long snd_addr;
	unsigned long long cnt;
};

struct CCudaVectorArray {
	static const int MaxSize = 16;
	float* Vectors[MaxSize];
	int VectorCount;
};

struct CCudaConstVectorArray {
	static const int MaxSize = 16;
	const float* Vectors[MaxSize];
	int VectorCount;
};

//------------------------------------------------------------------------------------------------------------

// define for logarithms FLT_MIN/MAX. define used to avoid problems with CUDA
#define FLT_MIN_LOG -87.33654474f
#define FLT_MAX_LOG 88.f

// The exponent with limitations to avoid NaN
inline __device__ float ExponentFunc(float f)
{
	if(f < FLT_MIN_LOG) {
		return 0;
	} else if(f > FLT_MAX_LOG) {
		return FLT_MAX;
	} else {
		return expf(f);
	}
}

// LogSumExp for two numbers
inline __device__ float LogSumExpFunc(float f, float s)
{
	if(f >= s) {
		return f + log1pf( ExponentFunc( s - f ) );
	} else {
		return s + log1pf( ExponentFunc( f - s ) );
	}
}

//------------------------------------------------------------------------------------------------------------

// RLE image
struct CCudaRleStroke {
	short Start;	// stroke start
	short End;		// stroke end (first position after it ends)
};

struct CCudaRleImage {
	int StrokesCount;
	int Height;
	int Width;
	CCudaRleStroke Stub;
	CCudaRleStroke Lines[1];
};

//------------------------------------------------------------------------------------------------------------

// Setting device
void SetCudaDevice( int deviceNum );

//------------------------------------------------------------------------------------------------------------

constexpr float BigFloatNumber = 18002376725743890449408517795774411571.f; // 18002376725743890.f; // 

//------------------------------------------------------------------------------------------------------------

constexpr int AddVectorToMatrixElementsKernel1Id = 1;
constexpr int AddVectorToMatrixElementsKernel2Id = 2;
constexpr int AddMatrixElementsToVectorKernel1Id = 3;
constexpr int AddMatrixElementsToVectorKernel2Id = 4;
constexpr int AddVectorToMatrixRowsKernelId = 5;
constexpr int SumMatrixRowsAddKernelId = 6;
constexpr int MatrixLogSumExpByRowsKernelId = 7;
constexpr int MatrixSoftmaxByRowsKernelId = 8;
constexpr int FindMaxValueWithIndicesInRowsKernelId = 9;
constexpr int VectorChannelLookupAndCopyKernelId = 10;
constexpr int MultiplyDiagMatrixByMatrixKernelId = 11;
constexpr int MultiplyDiagMatrixByMatrixAndSumKernelId = 12;
constexpr int MatrixSpreadRowsKernelId = 13;

constexpr int RandomMatrixDropoutKernelId = 14;

constexpr int BlobMergeByDimKernelId = 15;
constexpr int BlobSplitByDimKernelId = 16;
constexpr int BlobGetSubSequenceKernelId = 17;

constexpr int VectorAddKernelId = 18;
constexpr int VectorAddValueKernelId = 19;
// 20?
constexpr int VectorCopyKernelId = 21;
constexpr int VectorEltwiseDivideKernelId = 22;
constexpr int VectorEltwiseMaxKernelId = 23;
constexpr int VectorEltwiseMulKernelId = 24;
constexpr int VectorFillKernelId = 25;
constexpr int VectorMinMaxKernelId = 26;
constexpr int VectorMultiplyKernelId = 27;
constexpr int VectorSigmoidKernelId = 28;
constexpr int VectorSigmoidDiffKernelId = 29;
constexpr int VectorSigmoidDiffOpKernelId = 30;
constexpr int VectorSqrtKernelId = 31;
constexpr int VectorSub1KernelId = 32;
constexpr int VectorSub2KernelId = 33;
constexpr int VectorSub3KernelId = 34;
constexpr int VectorTanhKernelId = 35;
constexpr int VectorTanhDiffKernelId = 36;
constexpr int VectorTanhDiffOpKernelId = 37;
constexpr int VectorFillHandleKernelId = 38;
constexpr int VectorEltwiseMultiplyKernelId = 39;

constexpr int VectorSDotKernelId = 40;
constexpr int VectorMultiplyAndAddKernelId = 41;
constexpr int MultiplyMatrixByTransposedMatrix1KernelId = 42;
constexpr int MultiplyTransposedMatrixByMatrixAndAddKernelId = 43;
constexpr int MultiplyMatrixByMatrixKernelId = 44;

//------------------------------------------------------------------------------------------------------------

__device__ constexpr const char* const StringKernelId[45] = { "", /*0*/
	"AddVectorToMatrixElementsKernel(1)", /*1*/
	"AddVectorToMatrixElementsKernel(2)", /*2*/
	"AddMatrixElementsToVectorKernel(1)", /*3*/
	"AddMatrixElementsToVectorKernel(2)", /*4*/
	"AddVectorToMatrixRowsKernel", /*5*/
	"SumMatrixRowsAddKernel", /*6*/
	"MatrixLogSumExpByRowsKernel", /*7*/
	"MatrixSoftmaxByRowsKernel", /*8*/
	"FindMaxValueWithIndicesInRowsKernel", /*9*/
	"VectorChannelLookupAndCopyKernel", /*10*/
	"MultiplyDiagMatrixByMatrixKernel", /*11*/
	"MultiplyDiagMatrixByMatrixAndSumKernel", /*12*/
	"MatrixSpreadRowsKernel", /*13*/
	"RandomMatrixDropoutKernel", /*14*/
	"BlobMergeByDimKernel", /*15*/
	"BlobSplitByDimKernel", /*16*/
	"BlobGetSubSequenceKernel", /*17*/
	"VectorAddKernel", /*18*/
	"VectorAddValueKernel", /*19*/
	"", /*20?*/
	"VectorCopyKernel", /*21*/
	"VectorEltwiseDivideKernel", /*22*/
	"VectorEltwiseMaxKernel", /*23*/
	"VectorEltwiseMulKernel", /*24*/
	"VectorFillKernel", /*25*/
	"VectorMinMaxKernel", /*26*/
	"VectorMultiplyKernel", /*27*/
	"VectorSigmoidKernel", /*28*/
	"VectorSigmoidDiffKernel", /*29*/
	"VectorSigmoidDiffOpKernel", /*30*/
	"VectorSqrtKernel", /*31*/
	"VectorSubKernel(1)", /*32*/
	"VectorSubKernel(2)", /*33*/
	"VectorSubKernel(3)", /*34*/
	"VectorTanhKernel", /*35*/
	"VectorTanhDiffKernel", /*36*/
	"VectorTanhDiffOpKernel", /*37*/
	"VectorFillHandleKernel", /*38*/
	"VectorEltwiseMultiplyKernel", /*39*/
	"VectorSDotKernel", /*40*/
	"VectorMultiplyAndAddKernel", /*41*/
	"MultiplyMatrixByTransposedMatrix1Kernel", /*42*/
	"MultiplyTransposedMatrixByMatrixAndAddKernel", /*43*/
	"MultiplyMatrixByMatrixKernel" /*44*/
};

//------------------------------------------------------------------------------------------------------------

#define CUDA_PRINT_ADDR_WARN_F( first, base_first, second, base_second, result, base_result )   { \
		if (first) printf( "first=%f (%llx) ", (first), ( unsigned long long )(base_first) ); \
		if (second) printf( "second=%f (%llx) ", (second), ( unsigned long long )(base_second) ); \
		printf( "result=%f (%llx) ", (result), ( unsigned long long )(base_result) ); \
	}

#define CUDA_PRINT_ADDR_F( first, second, result )   { \
		if (first) printf( "first=%f (%llx) ", *(first), ( unsigned long long )(first) ); \
		if (second) printf( "second=%f (%llx) ", *(second), ( unsigned long long )(second) ); \
		printf( "result=(%llx) ", /*(result),*/ ( unsigned long long )(result) ); \
	}

#define CUDA_PRINT_ADDR_I( first, second, result )   { \
		if (first) printf( "first=%d (%llx) ", *(first), ( unsigned long long )(first) ); \
		if (second) printf( "second=%d (%llx) ", *(second), ( unsigned long long )(second) ); \
		printf( "result=(%llx) ", /*(result),*/ ( unsigned long long )(result) ); \
	}

#define CUDA_PRINT_REST( count, num, name, calls_counter, h, w, i, index )   { \
		if (count) printf( " count=%d ", (count) ); \
		if (i) printf( "i=%d ", (i) ); \
		if (h) printf( "h=%d ", (h) ); \
		if (w) printf( "w=%d ", (w) ); \
		if (index) printf( "index=%d ", (index) ); \
		if (num) printf( " !%d! ", (num) ); \
		if (name) printf( " '%s' ", (name) ); \
		if (calls_counter) printf( " call=%llu ", ( unsigned long long )(calls_counter) ); \
		printf( "\n" ); \
	}

#define CUDA_PRINT_WARN( kernelName, first, base_first, second, base_second, result, base_result, count, num, name, calls_counter, h, w, i, index )   { \
		printf( "%s:  ", (kernelName) ); \
		CUDA_PRINT_ADDR_WARN_F( first, base_first, second, base_second, result, base_result ); \
		CUDA_PRINT_REST( count, num, name, calls_counter, h, w, i, index ); \
	}

#define CUDA_PRINT_F( kernelName, first, second, result, count, num, name, calls_counter )   { \
		printf( "%s(flt)  ", (kernelName) ); \
		CUDA_PRINT_ADDR_F( first, second, result ); \
		CUDA_PRINT_REST( count, num, name, calls_counter, /*h*/0, /*w*/0, /*i*/0, /*index*/0 ); \
	}

#define CUDA_PRINT_I( kernelName, first, second, result, count, num, name, calls_counter )   { \
		printf( "%s(int)  ", (kernelName) ); \
		CUDA_PRINT_ADDR_I( first, second, result ); \
		CUDA_PRINT_REST( count, num, name, calls_counter, /*h*/0, /*w*/0, /*i*/0, /*index*/0 ); \
	}

//------------------------------------------------------------------------------------------------------------

static __device__ size_t LastPrintCounter = 0;

#define CUDA_PRINT_HISTORY( calls_counter, historyKernels ) { \
	if ( LastPrintCounter < calls_counter ) { \
		CCudaHistoryKernel* history = ( CCudaHistoryKernel* )historyKernels; \
		const int last = ( calls_counter % CudaHistoryKernelsSize ); \
		const int first = ( ( LastPrintCounter + 1 ) % CudaHistoryKernelsSize ); \
		printf( "history: {\n\t %10s \t %10s \t %10s \t %10s \t %8s \t %s \n", "counter", "result", "first", "second", "count", "kernel" ); \
		for( int i = first; i != last; i = ( i + 1 ) % CudaHistoryKernelsSize ) { \
			if( history[i].counter > 0 ) { \
				printf( "\t %10llu \t %10llx \t %10llx \t %10llx \t %8llu \t %s \n", \
					history[i].counter, history[i].res_addr, history[i].fst_addr, history[i].snd_addr, history[i].cnt, StringKernelId[history[i].kernel] ); \
			} \
		} \
		printf( "\t %10llu \t %10llx \t %10llx \t %10llx \t %8llu \t %s \n", \
			history[last].counter, history[last].res_addr, history[last].fst_addr, history[last].snd_addr, history[last].cnt, StringKernelId[history[last].kernel] ); \
		printf( "}\n\n" ); \
		LastPrintCounter = calls_counter; \
	} \
}

#define CUDA_INIT_HISTORY( first, second, result, count, calls_counter, historyKernels, id ) { \
		CCudaHistoryKernel& history = ( ( CCudaHistoryKernel* )historyKernels )[calls_counter % CudaHistoryKernelsSize]; \
		history.kernel = id; \
		history.counter = calls_counter; \
		history.res_addr = ( unsigned long long )result; \
		history.fst_addr = ( unsigned long long )first; \
		history.snd_addr = ( unsigned long long )second; \
		history.cnt = ( unsigned long long )count; \
	}

//------------------------------------------------------------------------------------------------------------

constexpr int min_calls_counter = 1;
constexpr int MAX_calls_counter = 2;

#define PRINT_HEAD3_CNT_SPEC_T( i, j, k, kernelName, first, second, result, count, num, name, calls_counter, historyKernels, id )   { \
		if( i == 0 && j == 0 && k == 0 ) { \
			CUDA_INIT_HISTORY( first, second, result, count, calls_counter, historyKernels, id ); \
			if( calls_counter > min_calls_counter && calls_counter <= MAX_calls_counter ) { \
				if constexpr( std::is_same_v<T, float> ) { \
					CUDA_PRINT_F( kernelName, first, second, result, count, num, name, calls_counter ); \
				} \
				CUDA_PRINT_HISTORY( calls_counter, historyKernels ); \
			} \
		} \
	}

#define PRINT_HEAD3_CNT_T( i, j, k, kernelName, first, second, result, count, calls_counter, historyKernels, id )   { \
		if( i == 0 && j == 0 && k == 0 ) { \
			CUDA_INIT_HISTORY( first, second, result, count, calls_counter, historyKernels, id ); \
			if( calls_counter > min_calls_counter && calls_counter <= MAX_calls_counter ) { \
				if constexpr( std::is_same_v<T, float> ) { \
					CUDA_PRINT_F( kernelName, first, second, result, count, /*num*/0, /*name*/(char*)0, calls_counter ); \
				} else if constexpr( std::is_same_v<T, int> ) { \
					CUDA_PRINT_I( kernelName, first, second, result, count, /*num*/0, /*name*/(char*)0, calls_counter ); \
				} else { \
					printf( "%s(ERR_TYPE) \n", (kernelName) ); \
				} \
				CUDA_PRINT_HISTORY( calls_counter, historyKernels ); \
			} \
		} \
	}

#define PRINT_HEAD3_CNT_F( i, j, k, kernelName, first, second, result, count, calls_counter, historyKernels, id )   { \
		if( i == 0 && j == 0 && k == 0 ) { \
			CUDA_INIT_HISTORY( first, second, result, count, calls_counter, historyKernels, id ); \
			if( calls_counter > min_calls_counter && calls_counter <= MAX_calls_counter ) { \
				CUDA_PRINT_F( kernelName, first, second, result, count, /*num*/0, /*name*/(char*)0, calls_counter ); \
				CUDA_PRINT_HISTORY( calls_counter, historyKernels ); \
			} \
		} \
	}

//------------------------------------------------------------------------------------------------------------

#define PRINT_HEAD2_CNT_T( i, j, k, kernelName, first, result, count, calls_counter, historyKernels, id )     \
		PRINT_HEAD3_CNT_T( i, j, k, kernelName, first, /*2*/(T*)0, result, count, calls_counter, historyKernels, id )

#define PRINT_HEAD2_CNT_F( i, j, k, kernelName, first, result, count, calls_counter, historyKernels, id )     \
		PRINT_HEAD3_CNT_F( i, j, k, kernelName, first, /*2*/(float*)0, result, count, calls_counter, historyKernels, id )

#define PRINT_HEAD1_CNT_T( i, j, k, kernelName, result, count, calls_counter, historyKernels, id )     \
		PRINT_HEAD3_CNT_T( i, j, k, kernelName, /*1*/(T*)0, /*2*/(T*)0, result, count, calls_counter, historyKernels, id )

//------------------------------------------------------------------------------------------------------------

#define PRINT_HEAD3_T( i, j, k, kernelName, first, second, result, count )       { \
		if( i == 0 && j == 0 && k == 0 ) { \
			if constexpr( std::is_same_v<T, float> ) { \
				CUDA_PRINT_F( kernelName, first, second, result, count, /*num*/0, /*name*/(char*)0, /*calls_counter*/0 ); \
			} else if constexpr( std::is_same_v<T, int> ) { \
				CUDA_PRINT_I( kernelName, first, second, result, count, /*num*/0, /*name*/(char*)0, /*calls_counter*/0 ); \
			} else { \
				printf( "%s(ERR_TYPE) \n", (kernelName) ); \
			} \
		} \
    }

#define PRINT_HEAD3_F(  i, j, k, kernelName, first, second, result, count )      { \
		if( i == 0 && j == 0 && k == 0 ) { \
			CUDA_PRINT_F( kernelName, first, second, result, count, /*num*/0, /*name*/(char*)0, /*calls_counter*/0 ); \
		} \
	}

#define PRINT_HEAD3_I(  i, j, k, kernelName, first, second, result, count )      { \
		if( i == 0 && j == 0 && k == 0 ) { \
			CUDA_PRINT_I( kernelName, first, second, result, count, /*num*/0, /*name*/(char*)0, /*calls_counter*/0 ); \
		} \
	}

//------------------------------------------------------------------------------------------------------------

#define PRINT_HEAD2_T( i, j, k, kernelName, first, result, count )         \
		PRINT_HEAD3_T( i, j, k, kernelName, first, /*2*/(T*)0, result, count )

#define PRINT_HEAD2_F( i, j, k, kernelName, first, result, count )         \
		PRINT_HEAD3_F( i, j, k, kernelName, first, /*2*/(float*)0, result, count )

#define PRINT_HEAD2_I( i, j, k, kernelName, first, result, count )         \
		PRINT_HEAD3_I( i, j, k, kernelName, first, /*2*/(int*)0, result, count )


#define PRINT_HEAD1_F( i, j, k, kernelName, result, count )                \
		PRINT_HEAD3_F( i, j, k, kernelName, /*1*/(float*)0, /*2*/(float*)0, result, count )

#define PRINT_HEAD1_I( i, j, k, kernelName, result, count )                \
		PRINT_HEAD3_I( i, j, k, kernelName, /*1*/(int*)0, /*2*/(int*)0, result, count )

//------------------------------------------------------------------------------------------------------------

static __device__ int OnceCounterFlag = 0;
static __device__ int SectionFlag = 0;

#define WARN3_CNT_SPEC_F( kernelName, first, base_first, second, base_second, result, base_result, count, i, index, num, name, calls_counter, historyKernels, id )   { \
		if( !isfinite( (result) ) || \
			(result) < -BigFloatNumber || \
			(result) >  BigFloatNumber )  \
		{ \
			while ( true ) { \
			if( atomicExch( &SectionFlag, 1 ) == 0 ) {   \
				CUDA_PRINT_WARN( kernelName, first, base_first, second, base_second, result, base_result, count, num, name, calls_counter, /*h*/0, /*w*/0, i, index ); \
				if ( !(id == VectorAddKernelId && num == 27) \
					&& id != VectorEltwiseDivideKernelId     \
					&& id != VectorSqrtKernelId              \
					&& id != VectorSDotKernelId              \
					&& id != VectorEltwiseMaxKernelId        \
					&& OnceCounterFlag < calls_counter )     \
				{ \
					CUDA_PRINT_HISTORY( calls_counter, historyKernels ); \
					OnceCounterFlag = calls_counter; \
					atomicExch( &SectionFlag, 0 ); \
					/*while( true );*/ \
				} else { \
					atomicExch( &SectionFlag, 0 ); \
				} \
				break; \
			} \
			} \
		} \
		/*assert( isfinite( (result) ) );*/ \
		/*assert( (result) > -BigFloatNumber );*/ \
		/*assert( (result) <  BigFloatNumber );*/ \
	}

#define WARN3_CNT_F( kernelName, first, base_first, second, base_second, result, base_result, h, w, index, calls_counter, historyKernels )   { \
		if( !isfinite( (result) ) || \
			(result) < -BigFloatNumber || \
			(result) >  BigFloatNumber )  \
		{ \
			while ( true ) { \
			if( atomicExch( &SectionFlag, 1 ) == 0 ) { \
				CUDA_PRINT_WARN( kernelName, first, base_first, second, base_second, result, base_result, /*count*/0, /*num*/0, /*name*/(char*)0, calls_counter, h, w, /*i*/0, index ); \
				if ( OnceCounterFlag < calls_counter ) { \
					CUDA_PRINT_HISTORY( calls_counter, historyKernels ); \
					OnceCounterFlag = calls_counter; \
				} \
				atomicExch( &SectionFlag, 0 ); \
				break; \
				/*while( true );*/ \
			} \
			} \
		} \
		/*assert( isfinite( (result) ) );*/ \
		/*assert( (result) > -BigFloatNumber );*/ \
		/*assert( (result) <  BigFloatNumber );*/ \
	}

#define WARN3_CNT_NORES_F( kernelName, result, base_result, count, id, index, num, calls_counter, historyKernels )   { \
		if( !isfinite( (result) ) || \
			(result) < -BigFloatNumber || \
			(result) >  BigFloatNumber )  \
		{ \
			if ( id != VectorSDotKernelId ) { \
				while ( true ) { \
					if( atomicExch( &SectionFlag, 1 ) == 0 ) { \
						CUDA_PRINT_WARN( kernelName, 0.f, (float*)0, 0.f, (float*)0, result, base_result, count, num, /*name*/(char*)0, calls_counter, /*h*/0, /*w*/0, id, index ); \
						atomicExch( &SectionFlag, 0 ); \
						break; \
					} \
				} \
			} \
		} \
		/*assert( isfinite( (result) ) );*/ \
		/*assert( (result) > -BigFloatNumber );*/ \
		/*assert( (result) <  BigFloatNumber );*/ \
	}


#define WARN3_CNT_SPEC_T( kernelName, first, base_first, second, base_second, result, base_result, count, i, index, num, name, calls_counter, historyKernels, id )   { \
		if constexpr( std::is_same_v<T, float> ) { \
			WARN3_CNT_SPEC_F( kernelName, first, base_first, float(second), base_second, result, base_result, count, i, index, num, name, calls_counter, historyKernels, id ); \
		} \
	}

#define WARN3_CNT_T( kernelName, first, base_first, second, base_second, result, base_result, h, w, index, calls_counter, historyKernels )   { \
		if constexpr( std::is_same_v<T, float> ) { \
			WARN3_CNT_F( kernelName, first, base_first, float(second), base_second, result, base_result, h, w, index, calls_counter, historyKernels ); \
		} \
	}

//------------------------------------------------------------------------------------------------------------

#define WARN2_CNT_SPEC_F( kernelName, first, base_first, result, base_result, count, i, index, num, name, calls_counter, historyKernels, id )     \
		WARN3_CNT_SPEC_F( kernelName, first, base_first, 0.f, (float*)0, result, base_result, count, i, index, num, name, calls_counter, historyKernels, id )

#define WARN2_CNT_F( kernelName, first, base_first, result, base_result, h, w, index, calls_counter, historyKernels )      \
		WARN3_CNT_F( kernelName, first, base_first, 0.f, (float*)0, result, base_result, h, w, index, calls_counter, historyKernels )


#define WARN2_CNT_SPEC_T( kernelName, first, base_first, result, base_result, count, i, index, num, name, calls_counter, historyKernels, id )     \
		WARN3_CNT_SPEC_T( kernelName, first, base_first, 0, (T*)0, result, base_result, count, i, index, num, name, calls_counter, historyKernels, id );

#define WARN2_CNT_T( kernelName, first, base_first, result, base_result, h, w, index, calls_counter, historyKernels )       \
		WARN3_CNT_T( kernelName, first, base_first, 0, (T*)0, result, base_result, h, w, index, calls_counter, historyKernels );

} // namespace NeoML

#endif // NEOML_USE_CUDA
