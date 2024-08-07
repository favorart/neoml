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

#include "common.h"
#pragma hdrstop

#include "onnx.pb.h"

#include "GraphOutput.h"
#include "TensorUtils.h"

using namespace NeoML;

namespace NeoOnnx {

CGraphOutput::CGraphOutput( const onnx::ValueInfoProto& output ) :
	name( output.name().c_str() )
{
}

CPtr<const CSinkLayer> CGraphOutput::AddSinkLayer( const CUserTensor& input, CDnn& dnn ) const
{
	CPtr<CSinkLayer> sink = new CSinkLayer( dnn.GetMathEngine() );
	sink->SetName( Name() );
	sink->Connect( 0, *input.Layer(), input.OutputIndex() );
	dnn.AddLayer( *sink );
	return sink.Ptr();
}

} // namespace NeoOnnx

