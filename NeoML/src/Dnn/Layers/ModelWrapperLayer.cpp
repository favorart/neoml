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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/ModelWrapperLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/Layers/SourceLayer.h>
#include <NeoML/Dnn/Layers/SinkLayer.h>

namespace NeoML {

void CProblemSourceLayer::SetBatchSize(int _batchSize)
{
	if(_batchSize == batchSize) {
		return;
	}
	batchSize = _batchSize;
	ForceReshape();
}

void CProblemSourceLayer::SetProblem(const CPtr<const IProblem>& _problem)
{
	NeoAssert( _problem != nullptr );
	NeoAssert( GetDnn() == nullptr || problem == nullptr
		|| ( problem->GetFeatureCount() == _problem->GetFeatureCount()
			&& problem->GetClassCount() == _problem->GetClassCount() ) );

	problem = _problem;
	nextProblemIndex = 0;
}

void CProblemSourceLayer::SetLabelType( TBlobType newLabelType )
{
	NeoAssert( newLabelType == CT_Float || newLabelType == CT_Int );

	if( labelType == newLabelType ) {
		return;
	}

	labelType = newLabelType;

	ForceReshape();
}

void CProblemSourceLayer::Reshape()
{
	NeoAssert(!GetDnn()->IsRecurrentMode());

	CheckLayerArchitecture( problem.Ptr() != 0, "source problem is null" );
	CheckOutputs();
	CheckLayerArchitecture( GetOutputCount() >= 2, "problem source layer has less than 2 outputs" );

	// The data
	outputDescs[0] = CBlobDesc( CT_Float );
	outputDescs[0].SetDimSize( BD_BatchWidth, batchSize );
	outputDescs[0].SetDimSize( BD_Channels, problem->GetFeatureCount() );
	exchangeBufs[0].SetSize(outputDescs[0].BlobSize());

	// The labels
	int labelSize = problem->GetClassCount();
	if(labelSize == 2) {
		labelSize = 1;
	}
	outputDescs[1] = CBlobDesc( labelType );
	outputDescs[1].SetDimSize( BD_BatchWidth, batchSize );
	if( labelType != CT_Int ) {
		outputDescs[1].SetDimSize( BD_Channels, labelSize );
	}
	exchangeBufs[1].SetSize(outputDescs[1].BlobSize());

	// The weights
	outputDescs[2] = CBlobDesc( CT_Float );
	outputDescs[2].SetDimSize( BD_BatchWidth, batchSize );
	exchangeBufs[2].SetSize(outputDescs[2].BlobSize());
}

void CProblemSourceLayer::RunOnce()
{
	NeoAssert( problem != nullptr );

	::memset( exchangeBufs[1].GetPtr(), 0, exchangeBufs[1].Size() * sizeof( float ) );
	if( emptyFill == 0.f ) {
		::memset( exchangeBufs[0].GetPtr(), 0, exchangeBufs[0].Size() * sizeof( float ) );
	} else {
		for( int i = 0; i < exchangeBufs[0].Size(); ++i ) {
			exchangeBufs[0][i] = emptyFill;
		}
	}

	float* data = exchangeBufs[0].GetPtr();
	float* labels = exchangeBufs[1].GetPtr();
	float* weights = exchangeBufs[2].GetPtr();

	int vectorCount = problem->GetVectorCount();
	CFloatMatrixDesc matrix = problem->GetMatrix();
	CFloatVectorDesc vector;

	for(int i = 0; i < batchSize; ++i) {
		// The data
		matrix.GetRow( nextProblemIndex, vector );
		for(int j = 0; j < vector.Size; ++j) {
			data[vector.Indexes == nullptr ? j : vector.Indexes[j]] = static_cast<float>( vector.Values[j] );
		}

		// The labels
		// Update labels
		if( labelType == CT_Float ) {
			if( outputBlobs[1]->GetChannelsCount() == 1 ) {
				*labels = static_cast< float >( problem->GetBinaryClass( nextProblemIndex ) );
			} else {
				int classLabel = problem->GetClass( nextProblemIndex );
				NeoAssert( 0 <= classLabel && classLabel < outputBlobs[1]->GetChannelsCount() );
				::memset( labels, 0, outputBlobs[1]->GetChannelsCount() * sizeof( float ) );
				labels[classLabel] = 1;
			}
		} else {
			static_assert( sizeof( float ) == sizeof( int ), "sizeof( float ) != sizeof( int )" );
			NeoAssert( outputBlobs[1]->GetChannelsCount() == 1 );
			*reinterpret_cast<int*>( labels ) = problem->GetClass( nextProblemIndex );
		}

		// The weights
		*weights = static_cast<float>(problem->GetVectorWeight(nextProblemIndex));

		++nextProblemIndex;
		nextProblemIndex %= vectorCount;

		data += outputBlobs[0]->GetObjectSize();
		labels += outputBlobs[1]->GetObjectSize();
		weights += outputBlobs[2]->GetObjectSize();
	}

	outputBlobs[0]->CopyFrom(exchangeBufs[0].GetPtr());
	if( labelType == CT_Float ) {
		outputBlobs[1]->CopyFrom( exchangeBufs[1].GetPtr() );
	} else {
		outputBlobs[1]->CopyFrom( reinterpret_cast<int*>( exchangeBufs[1].GetPtr() ) );
	}
	outputBlobs[2]->CopyFrom(exchangeBufs[2].GetPtr());
}

void CProblemSourceLayer::BackwardOnce()
{
	NeoAssert(0);
}

constexpr int ProblemSourceLayerVersion = 2001;

void CProblemSourceLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( ProblemSourceLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	if( version >= 2001 ) {
		archive.Serialize( emptyFill );
	} else { // loading
		emptyFill = 0;
	}
	archive.Serialize( batchSize );
	int labelTypeInt = static_cast<int>( labelType );
	archive.Serialize( labelTypeInt );

	if( archive.IsLoading() ) {
		nextProblemIndex = NotFound;
		labelType = static_cast<TBlobType>( labelTypeInt );
		problem = nullptr;
	}
}

// Creates CProblemSourceLayer with the name
CProblemSourceLayer* ProblemSource( CDnn& dnn, const char* name,
	TBlobType labelType, int batchSize, const CPtr<const IProblem>& problem )
{
	CPtr<CProblemSourceLayer> result = new CProblemSourceLayer( dnn.GetMathEngine() );
	result->SetProblem( problem );
	result->SetLabelType( labelType );
	result->SetBatchSize( batchSize );
	result->SetName( name );
	dnn.AddLayer( *result );
	return result;
}

//---------------------------------------------------------------------------------------------------------------------

const char* const CDnnModelWrapper::SourceLayerName = "CCnnModelWrapper::SourceLayer";
const char* const CDnnModelWrapper::SinkLayerName = "CCnnModelWrapper::SinkLayer";

CDnnModelWrapper::CDnnModelWrapper(IMathEngine& _mathEngine, unsigned int seed) :
	ClassCount(0),
	SourceEmptyFill(0),	
	Random(seed),
	Dnn(Random, _mathEngine),
	mathEngine( _mathEngine )
{
	SourceLayer = FINE_DEBUG_NEW CSourceLayer( mathEngine );
	SourceLayer->SetName(SourceLayerName);

	SinkLayer = FINE_DEBUG_NEW CSinkLayer( mathEngine );
	SinkLayer->SetName(SinkLayerName);
}

bool CDnnModelWrapper::Classify(const CFloatVectorDesc& desc, CClassificationResult& result) const
{
	NeoAssert( SourceBlob != nullptr );
	NeoPresume( SourceBlob == SourceLayer->GetBlob() );

	exchangeBuffer.SetSize( SourceBlob->GetDataSize() );
	if( SourceEmptyFill == 0.f ) {
		::memset( exchangeBuffer.GetPtr(), 0, exchangeBuffer.Size() * sizeof( float ) );
	} else {
		for( int i = 0; i < exchangeBuffer.Size(); ++i ) {
			exchangeBuffer[i] = SourceEmptyFill;
		}
	}

	for(int i = 0; i < desc.Size; ++i) {
		exchangeBuffer[( desc.Indexes == nullptr ) ? i : desc.Indexes[i]] = desc.Values[i];
	}
	SourceBlob->CopyFrom( exchangeBuffer.GetPtr() );

	return classify( result );
}

constexpr int DnnModelWrapperVersion = 2001;

void CDnnModelWrapper::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( DnnModelWrapperVersion, CDnn::ArchiveMinSupportedVersion );

	archive.Serialize( ClassCount );
	if( version >= 2001 ) {
		archive.Serialize( SourceEmptyFill );
	} else { // loading
		SourceEmptyFill = 0;
	}
	archive.Serialize( Random );
	archive.Serialize( Dnn );

	CString sourceName = SourceLayer->GetName();
	archive.Serialize( sourceName );
	CString sinkName = SinkLayer->GetName();
	archive.Serialize( sinkName );

	CBlobDesc sourceDesc( CT_Float );
	sourceDesc.SetDimSize( BD_BatchWidth, 0 ); // set zero
	if( SourceBlob != nullptr ) {
		sourceDesc = SourceBlob->GetDesc();
	}
	for( int i = 0; i < CBlobDesc::MaxDimensions; ++i ) {
		int size = sourceDesc.DimSize( i );
		archive.Serialize( size );
		sourceDesc.SetDimSize( i, size );
	}

	if( archive.IsLoading() ) {
		if( Dnn.HasLayer( sourceName ) ) {
			SourceLayer = CheckCast<CSourceLayer>( Dnn.GetLayer( sourceName ).Ptr() );
		} else {
			SourceLayer->SetName( sourceName );
		}
		if( Dnn.HasLayer( sinkName ) ) {
			SinkLayer = CheckCast<CSinkLayer>( Dnn.GetLayer( sinkName ).Ptr() );
		} else {
			SinkLayer->SetName( sinkName );
		}
		if( sourceDesc.BlobSize() == 0 ) { // is zero
			SourceBlob = nullptr;
		} else {
			SourceBlob = CDnnBlob::CreateBlob(mathEngine, CT_Float, sourceDesc);
			SourceLayer->SetBlob(SourceBlob);
		}
		exchangeBuffer.SetSize(0);
		tempExp.SetSize(0);
	}
}

bool CDnnModelWrapper::classify( CClassificationResult& result ) const
{
	Dnn.RunOnce();

	const CPtr<CDnnBlob>& resultBlob = SinkLayer->GetBlob();
	NeoAssert(resultBlob->GetObjectCount() == 1);

	result.ExceptionProbability = CClassificationProbability(0);
	result.Probabilities.SetSize(ClassCount);

	if(ClassCount == 2) {
		// Binary classification is a special case
		NeoAssert(resultBlob->GetObjectSize() == 1);

		// Use a sigmoid function to estimate probabilities
		const double zeroClassProb = 1 / (1 + exp(resultBlob->GetData().GetValue()));
		result.Probabilities[0] = CClassificationProbability(zeroClassProb);
		result.Probabilities[1] = CClassificationProbability(1 - zeroClassProb);
		result.PreferredClass = zeroClassProb >= 0.5 ? 0 : 1;
	} else {
		NeoAssert(resultBlob->GetObjectSize() == ClassCount);

		tempExp.SetSize(ClassCount);
		resultBlob->CopyTo(tempExp.GetPtr(), tempExp.Size());

		// Normalize the weights before taking expf
		float maxWeight = tempExp[0];
		result.PreferredClass = 0;
		for( int i = 1; i < ClassCount; ++i ) {
			if(tempExp[i] > maxWeight) {
				maxWeight = tempExp[i];
				result.PreferredClass = i;
			}
		}

		// Use softmax to estimate probabilities
		float expSum = 0;

		for(int i = 0; i < ClassCount; ++i) {
			// To divide all exponent terms by the maximum, subtract it beforehand
			tempExp[i] = expf(tempExp[i] - maxWeight);
			expSum += tempExp[i];
		}

		for(int i = 0; i < ClassCount; ++i) {
			result.Probabilities[i] = CClassificationProbability(tempExp[i] / expSum);
		}
	}

	return true;
}

//---------------------------------------------------------------------------------------------------------------------

CPtr<IModel> CDnnTrainingModelWrapper::Train(const IProblem& trainingClassificationData)
{
	CPtr<CDnnModelWrapper> model = FINE_DEBUG_NEW CDnnModelWrapper( mathEngine );
	CPtr<CProblemSourceLayer> problem = FINE_DEBUG_NEW CProblemSourceLayer( mathEngine );

	problem->SetName(model->SourceLayer->GetName());
	problem->SetProblem(&trainingClassificationData);

	model->ClassCount = trainingClassificationData.GetClassCount();

	BuildAndTrainDnn(model->Dnn, problem, model->SourceLayer, model->SinkLayer);

	model->SourceBlob = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 1, trainingClassificationData.GetFeatureCount() );
	model->SourceLayer->SetBlob(model->SourceBlob);

	return model.Ptr();
}

} //namespace NeoML
