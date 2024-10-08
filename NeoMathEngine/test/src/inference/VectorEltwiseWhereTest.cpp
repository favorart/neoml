/* Copyright © 2017-2024 ABBYY

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

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

template<class T>
static void vectorEltwiseWhereImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );
	const CInterval vectorSizeInterval = params.GetInterval( "VectorSize" );
	const CInterval valuesInterval = params.GetInterval( "Values" );

	const int vectorSize = random.UniformInt( vectorSizeInterval.Begin, vectorSizeInterval.End );
	CREATE_FILL_ARRAY( int, first, 0, 3, vectorSize, random )
	CREATE_FILL_ARRAY( T, second, valuesInterval.Begin, valuesInterval.End, vectorSize, random )
	CREATE_FILL_ARRAY( T, third, valuesInterval.Begin, valuesInterval.End, vectorSize, random )

	CREATE_FILL_ARRAY( T, result, valuesInterval.Begin, valuesInterval.End, vectorSize, random )

	MathEngine().VectorEltwiseWhere( CARRAY_INT_WRAPPER( first ), CARRAY_WRAPPER( T, second ), CARRAY_WRAPPER( T, third ),
		CARRAY_WRAPPER( T, result ), vectorSize );

	for( int i = 0; i < vectorSize; i++ ) {
		const T expected = first[i] != 0 ? second[i] : third[i];
		EXPECT_EQ( expected, result[i] );
	}
}

//------------------------------------------------------------------------------------------------------------

class CVectorEltwiseWhereTest : public CTestFixtureWithParams {
};

INSTANTIATE_TEST_CASE_P( CVectorEltwiseWhereTestInstantiation, CVectorEltwiseWhereTest,
	::testing::Values(
		CTestParams(
			"VectorSize = (10..100);"
			"Values = (-50..50);"
			"TestCount = 100;"
		)
	)
);

TEST_P( CVectorEltwiseWhereTest, Random )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	RUN_TEST_IMPL( vectorEltwiseWhereImpl<float> );
	RUN_TEST_IMPL( vectorEltwiseWhereImpl<int> );
}
