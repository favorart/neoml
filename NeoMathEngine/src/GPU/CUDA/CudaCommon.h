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

constexpr int min_calls_counter = 0;
constexpr int MAX_calls_counter = 1000000;

#define PRINT_HEAD2_T( i, j, k, kernelName, first, result, count )       { \
		if( i == 0 && j == 0 && k == 0 ) { \
			if constexpr( std::is_same_v<T, float> ) { \
				printf( "%s(flt)  first=%f (%llx) result=(%llx)  count=%d \n", (kernelName), \
					*(first), ( unsigned long long )(first), /**(result),*/ ( unsigned long long )(result), (count) ); \
			} else if constexpr( std::is_same_v<T, int> ) { \
				printf( "%s(int)  first=%d (%llx) result=(%llx)  count=%d \n", (kernelName), \
					*(first), ( unsigned long long )(first), /**(result),*/ ( unsigned long long )(result), (count) ); \
			} else { \
				printf( "%s(ERR_TYPE) \n", (kernelName) ); \
			} \
		} \
    }

#define PRINT_HEAD2_F(  i, j, k, kernelName, first, result, count )      { \
		if( i == 0 && j == 0 && k == 0 ) { \
			printf( "%s(flt)  first=%f (%llx) result=(%llx)  count=%d \n", (kernelName), \
				*(first), ( unsigned long long )(first), /**(result),*/ ( unsigned long long )(result), (count) ); \
		} \
	}

#define PRINT_HEAD2_I(  i, j, k, kernelName, first, result, count )      { \
		if( i == 0 && j == 0 && k == 0 ) { \
			printf( "%s(int)  first=%d (%llx) result=(%llx)  count=%d \n", (kernelName), \
				*(first), ( unsigned long long )(first), /**(result),*/ ( unsigned long long )(result), (count) ); \
		} \
	}

//------------------------------------------------------------------------------------------------------------

#define PRINT_HEAD3_T( i, j, k, kernelName, first, second, result, count )       { \
		if( i == 0 && j == 0 && k == 0 ) { \
			if constexpr( std::is_same_v<T, float> ) { \
				printf( "%s(flt)  first=%f (%llx) second=%f (%llx) result=(%llx)  count=%d \n", (kernelName), \
					*(first), ( unsigned long long )(first), *(second), ( unsigned long long )(second), /**(result),*/ ( unsigned long long )(result), (count) ); \
			} else if constexpr( std::is_same_v<T, int> ) { \
				printf( "%s(int)  first=%d (%llx) second=%d (%llx) result=(%llx)  count=%d \n", (kernelName), \
					*(first), ( unsigned long long )(first), *(second), ( unsigned long long )(second), /**(result),*/ ( unsigned long long )(result), (count) ); \
			} else { \
				printf( "%s(ERR_TYPE) \n", (kernelName) ); \
			} \
		} \
    }

#define PRINT_HEAD3_F(  i, j, k, kernelName, first, second, result, count )      { \
		if( i == 0 && j == 0 && k == 0 ) { \
			printf( "%s(flt)  first=%f (%llx) second=%f (%llx) result=(%llx)  count=%d \n", (kernelName), \
				*(first), ( unsigned long long )(first), *(second), ( unsigned long long )(second), /**(result),*/ ( unsigned long long )(result), (count) ); \
		} \
	}

//------------------------------------------------------------------------------------------------------------

#define PRINT_HEAD1_CNT_T( i, j, k, kernelName, result, count, calls_counter )   { \
		if( i == 0 && j == 0 && k == 0 && calls_counter > min_calls_counter && calls_counter <= MAX_calls_counter ) { \
			if constexpr( std::is_same_v<T, float> ) { \
				printf( "%s(flt)  result=(%llx)  count=%d  call=%llu \n", (kernelName), \
					/**(result),*/ ( unsigned long long )(result), \
					(count), ( unsigned long long )(calls_counter) ); \
			} else /*if constexpr( std::is_same_v<T, int> )*/ { \
				printf( "%s(int)  result=(%llx)  count=%d  call=%llu \n", (kernelName), \
					/**(result),*/ ( unsigned long long )(result), \
					(count), ( unsigned long long )(calls_counter) ); \
			/*} else { printf( "%s(ERR_TYPE) \n", (kernelName) );*/ } \
		} \
	}

#define PRINT_HEAD1_F( i, j, k, kernelName, result, count )   { \
		if( i == 0 && j == 0 && k == 0 ) { \
			printf( "%s(flt)  result=(%llx)  count=%d \n", (kernelName), \
				/**(result),*/ ( unsigned long long )(result), (count) ); \
		} \
	}

#define PRINT_HEAD1_I( i, j, k, kernelName, result, count )   { \
		if( i == 0 && j == 0 && k == 0 ) { \
			printf( "%s(int)  result=(%llx)  count=%d \n", (kernelName), \
				/**(result),*/ ( unsigned long long )(result), (count) ); \
		} \
	}

//------------------------------------------------------------------------------------------------------------

#define PRINT_HEAD2_CNT_T( i, j, k, kernelName, first, result, count, calls_counter )   { \
		if( i == 0 && j == 0 && k == 0 && calls_counter > min_calls_counter && calls_counter <= MAX_calls_counter ) { \
			if constexpr( std::is_same_v<T, float> ) { \
				printf( "%s(flt)  first=%f (%llx) result=(%llx)  count=%d  call=%llu \n", (kernelName), \
					*(first), ( unsigned long long )(first), /**(result),*/ ( unsigned long long )(result), \
					(count), ( unsigned long long )(calls_counter) ); \
			} else /*if constexpr( std::is_same_v<T, int> )*/ { \
				printf( "%s(int)  first=%d (%llx) result=(%llx)  count=%d  call=%llu \n", (kernelName), \
					*(first), ( unsigned long long )(first), /**(result),*/ ( unsigned long long )(result), \
					(count), ( unsigned long long )(calls_counter) ); \
			/*} else { printf( "%s(ERR_TYPE) \n", (kernelName) );*/ } \
		} \
	}

#define PRINT_HEAD2_CNT_F( i, j, k, kernelName, first, result, count, calls_counter )   { \
		if( i == 0 && j == 0 && k == 0 && calls_counter > min_calls_counter && calls_counter <= MAX_calls_counter ) { \
			printf( "%s(flt)  first=%f (%llx) result=(%llx)  count=%d  call=%llu \n", (kernelName), \
				*(first), ( unsigned long long )(first), /**(result),*/ ( unsigned long long )(result), \
					(count), ( unsigned long long )(calls_counter) ); \
		} \
	}

//------------------------------------------------------------------------------------------------------------

#define PRINT_HEAD3_CNT_T( i, j, k, kernelName, first, second, result, count, calls_counter )   { \
		if( i == 0 && j == 0 && k == 0 && calls_counter > min_calls_counter && calls_counter <= MAX_calls_counter ) { \
			if constexpr( std::is_same_v<T, float> ) { \
				printf( "%s(flt)  first=%f (%llx) second=%f (%llx) result=(%llx)  count=%d  call=%llu \n", (kernelName), \
					*(first), ( unsigned long long )(first), *(second), ( unsigned long long )(second), /**(result),*/ ( unsigned long long )(result), \
					(count), ( unsigned long long )(calls_counter) ); \
			} else /*if constexpr( std::is_same_v<T, int> )*/ { \
				printf( "%s(int)  first=%d (%llx) second=%d (%llx) result=(%llx)  count=%d  call=%llu \n", (kernelName), \
					*(first), ( unsigned long long )(first), *(second), ( unsigned long long )(second), /**(result),*/ ( unsigned long long )(result), \
					(count), ( unsigned long long )(calls_counter) ); \
			/*} else { printf( "%s(ERR_TYPE) \n", (kernelName) );*/ } \
		} \
	}

#define PRINT_HEAD3_CNT_F( i, j, k, kernelName, first, second, result, count, calls_counter )   { \
		if( i == 0 && j == 0 && k == 0 && calls_counter > min_calls_counter && calls_counter <= MAX_calls_counter ) { \
			printf( "%s(flt)  first=%f (%llx) second=%f (%llx) result=(%llx)  count=%d  call=%llu \n", (kernelName), \
				*(first), ( unsigned long long )(first), *(second), ( unsigned long long )(second), /**(result),*/ ( unsigned long long )(result), \
				(count), ( unsigned long long )(calls_counter) ); \
		} \
	}

#define PRINT_HEAD3_CNT_SPEC_T( i, j, k, kernelName, first, second, result, count, num, name, calls_counter )   { \
		if constexpr( std::is_same_v<T, float> ) { \
			if( i == 0 && j == 0 && k == 0 && calls_counter > min_calls_counter && calls_counter <= MAX_calls_counter ) { \
				printf( "%s(flt)  first=%f (%llx) second=%f (%llx) result=(%llx)  count=%d  !%d!  '%s'  call=%llu \n", (kernelName), \
					*(first), ( unsigned long long )(first), *(second), ( unsigned long long )(second), /**(result),*/ ( unsigned long long )(result), \
					(count), (num), (name), ( unsigned long long )(calls_counter) ); \
			} \
		} \
	}

//------------------------------------------------------------------------------------------------------------

#define WARN2_F( kernelName, first, base_first, result, base_result, h, w, index )   { \
		if( !isfinite( (result) ) || \
			(result) < -18002376725743890449408517795774411571.f || \
			(result) >  18002376725743890449408517795774411571.f ) \
		{ \
			printf( "%s: first=%f (%llx) result=%f (%llx)  h=%d w=%d index=%d \n", (kernelName), \
				(first), ( unsigned long long )(base_first), (result), ( unsigned long long )(base_result), \
				(h), (w), (index) ); \
		} \
		assert( isfinite( (result) ) ); \
		assert( (result) > -18002376725743890449408517795774411571.f ); \
		assert( (result) <  18002376725743890449408517795774411571.f ); \
	}


#define WARN3_F( kernelName, first, base_first, second, base_second, result, base_result, h, w, index )   { \
		if( !isfinite( (result) ) || \
			(result) < -18002376725743890449408517795774411571.f || \
			(result) >  18002376725743890449408517795774411571.f ) \
		{ \
			printf( "%s: first=%f (%llx) first=%f (%llx) result=%f (%llx)  h=%d w=%d index=%d \n", (kernelName), \
				(first), ( unsigned long long )(base_first), (second), ( unsigned long long )(base_second), (result), ( unsigned long long )(base_result), \
				(h), (w), (index) ); \
		} \
		assert( isfinite( (result) ) ); \
		assert( (result) > -18002376725743890449408517795774411571.f ); \
		assert( (result) <  18002376725743890449408517795774411571.f ); \
	}

//------------------------------------------------------------------------------------------------------------

#define WARN3_CNT_SPEC_F( kernelName, first, base_first, second, base_second, result, base_result, count, i, index, num, name, calls_counter )   { \
		if( !isfinite( (result) ) || \
			(result) < -18002376725743890449408517795774411571.f || \
			(result) >  18002376725743890449408517795774411571.f ) \
		{ \
			assert( num && name ); \
			if ( second && base_second ) { \
				printf( "%s: first=%f (%llx) second=%f (%llx)  result=%f (%llx)  count=%d i=%d index=%d  !%d!  '%s'  call=%llu \n", (kernelName), \
					(first), ( unsigned long long )(base_first), (second), ( unsigned long long )(base_second), (result), ( unsigned long long )(base_result), \
					(count), (i), (index), (num), (name), ( unsigned long long )(calls_counter) ); \
			} else { \
				printf( "%s: first=%f (%llx) result=%f (%llx)  count=%d i=%d index=%d  !%d!  '%s'  call=%llu \n", (kernelName), \
					(first), ( unsigned long long )(base_first), (result), ( unsigned long long )(base_result), \
					(count), (i), (index), (num), (name), ( unsigned long long )(calls_counter) ); \
			} \
		} \
	}

#define WARN3_CNT_F( kernelName, first, base_first, second, base_second, result, base_result, h, w, index, calls_counter )   { \
		if( !isfinite( (result) ) || \
			(result) < -18002376725743890449408517795774411571.f || \
			(result) >  18002376725743890449408517795774411571.f ) \
		{ \
			if ( second && base_second ) { \
				printf( "%s: first=%f (%llx) second=%f (%llx)  result=%f (%llx)  h=%d w=%d index=%d  call=%llu \n", (kernelName), \
					(first), ( unsigned long long )(base_first), (second), ( unsigned long long )(base_second), (result), ( unsigned long long )(base_result), \
					(h), (w), (index), ( unsigned long long )(calls_counter) ); \
			} else { \
				printf( "%s: first=%f (%llx) result=%f (%llx)  h=%d w=%d index=%d  call=%llu \n", (kernelName), \
					(first), ( unsigned long long )(base_first), (result), ( unsigned long long )(base_result), \
					(h), (w), (index), ( unsigned long long )(calls_counter) ); \
			} \
		} \
		assert( isfinite( (result) ) ); \
		assert( (result) > -18002376725743890449408517795774411571.f ); \
		assert( (result) <  18002376725743890449408517795774411571.f ); \
	}


#define WARN3_CNT_SPEC_T( kernelName, first, base_first, second, base_second, result, base_result, count, i, index, num, name, calls_counter )   { \
		if constexpr( std::is_same_v<T, float> ) { \
			WARN3_CNT_SPEC_F( kernelName, first, base_first, float(second), base_second, result, base_result, count, i, index, num, name, calls_counter ); \
		} \
	}

#define WARN3_CNT_T( kernelName, first, base_first, second, base_second, result, base_result, h, w, index, calls_counter )   { \
		if constexpr( std::is_same_v<T, float> ) { \
			WARN3_CNT_F( kernelName, first, base_first, float(second), base_second, result, base_result, h, w, index, calls_counter ); \
		} \
	}

//------------------------------------------------------------------------------------------------------------

#define WARN2_CNT_SPEC_F( kernelName, first, base_first, result, base_result, count, i, index, num, name, calls_counter )     \
		WARN3_CNT_SPEC_F( kernelName, first, base_first, 0.f, nullptr, result, base_result, count, i, index, num, name, calls_counter )

#define WARN2_CNT_F( kernelName, first, base_first, result, base_result, h, w, index, calls_counter )      \
		WARN3_CNT_F( kernelName, first, base_first, 0.f, nullptr, result, base_result, h, w, index, calls_counter )


#define WARN2_CNT_SPEC_T( kernelName, first, base_first, result, base_result, count, i, index, num, name, calls_counter )     \
		WARN3_CNT_SPEC_T( kernelName, first, base_first, 0, nullptr, result, base_result, count, i, index, num, name, calls_counter );

#define WARN2_CNT_T( kernelName, first, base_first, result, base_result, h, w, index, calls_counter )       \
		WARN3_CNT_T( kernelName, first, base_first, 0, nullptr, result, base_result, h, w, index, calls_counter );

} // namespace NeoML

#endif // NEOML_USE_CUDA
