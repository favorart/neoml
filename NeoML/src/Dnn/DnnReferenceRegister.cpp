/* Copyright @ 2024 ABBYY

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

namespace NeoML {

CDnnReferenceRegister::CDnnReferenceRegister() = default;

CDnnReferenceRegister::CDnnReferenceRegister(CDnn* _originalDnn) :
	learningState(false),
	referenceCounter(-1),
	originalDnn(_originalDnn),
	originalRandom(new CRandom(_originalDnn->Random()))
{
	NeoAssert(_originalDnn != nullptr);
	if(originalDnn->referenceDnnRegister.referenceCounter++ == 0) {
		originalDnn->referenceDnnRegister.learningState = originalDnn->IsLearningEnabled();
	}
}

CDnnReferenceRegister::~CDnnReferenceRegister()
{
	if(originalDnn == nullptr) {
		NeoAssertMsg(referenceCounter == 0,
			"delete reference dnns before original dnn");
	}

	if(referenceCounter == -1 && --(originalDnn->referenceDnnRegister.referenceCounter) == 0
		&& originalDnn->referenceDnnRegister.learningState)
	{
		originalDnn->EnableLearning();
	}

	if(originalRandom != nullptr) {
		delete originalRandom;
	}
}

CDnnReferenceRegister& CDnnReferenceRegister::operator=(CDnnReferenceRegister&& other) {
	if(this != &other) {
		learningState = other.learningState;
		referenceCounter = other.referenceCounter;
		originalDnn = other.originalDnn;
		originalRandom = other.originalRandom;

		other.originalDnn = nullptr;
		other.referenceCounter = 0;
		other.learningState = false;
		other.originalRandom = nullptr;
	}
	return *this;
}

} // namespace NeoML
