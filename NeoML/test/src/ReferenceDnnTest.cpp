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
#include <mutex>
#pragma hdrstop

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

struct CReferenceDnnTestParam {
	CDnn* net;
};

static void runDnn(int, void* params)
{
	CReferenceDnnTestParam* taskParams = static_cast<CReferenceDnnTestParam*>(params);
	taskParams->net->RunOnce();
}

static CDnn* createDnn(CRandom& random)
{
	CDnn* net = new CDnn(random, MathEngine());

	CBaseLayer* layer = Source(*net, "in");
	layer = FullyConnected(50)("fc1", layer);
	layer = Dropout(0.1f)("dp1", layer);
	layer = FullyConnected(200)("fc2", layer);
	layer = Dropout(0.1f)("dp2", layer);
	layer = FullyConnected(10)("fc3", layer);
	layer = Sink(layer, "sink");

	return net;
}

static void initializeBlob(CDnnBlob* blob, CRandom& random, int min, int max) {
	for(int j = 0; j < blob->GetDataSize(); ++j) {
		blob->GetData().SetValueAt(j, static_cast<float>(random.Uniform(min, max)));
	}
}

static void getTestDnns(CArray<CDnn*>& dnns, CArray<CRandom>& randoms, bool useReference, const int& numOfThreads)
{
	for(int i = 0; i < numOfThreads; ++i) {
		bool ifCreateNewDnn = (i == 0 || !useReference);
		if(ifCreateNewDnn) {
			dnns.Add(createDnn(randoms[i]));
		} else {
			dnns.Add( dnns[i-1]->CreateReferenceDnn());
		}

		CPtr<CDnnBlob> srcBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 8, 20, 30, 10 });
		static_cast<CSourceLayer*>(dnns[i]->GetLayer("in").Ptr())->SetBlob(srcBlob);
	}

	for(int i = 0; i < numOfThreads; ++i) {
		auto srcLayer = static_cast<CSourceLayer*>(dnns[i]->GetLayer("in").Ptr());
		initializeBlob(srcLayer->GetBlob(), dnns[i]->Random(), 0, 1);
	}
}

static void runMultithreadInference(CArray<CDnn*>& dnns, const int numOfThreads)
{
	CArray<CReferenceDnnTestParam> taskParams;
	for(int i = 0; i < numOfThreads; ++i) {
		taskParams.Add({ dnns[i] });
	}

	IThreadPool* pool = CreateThreadPool(numOfThreads);
	for(int i = 0; i < numOfThreads; ++i) {
		pool->AddTask(i, runDnn, &(taskParams[i]));
	}

	pool->WaitAllTask();
	delete pool;
}

static void perfomanceTest(bool useReference, const int numOfThreads=4)
{
	CArray<CDnn*> dnns;
	CArray<CRandom> randoms;

	const int numOfOriginalDnns = useReference ? 1 : numOfThreads;
	for(int i = 0; i < numOfOriginalDnns; ++i) {
		randoms.Add(CRandom(0));
	}

	IPerformanceCounters* counters(MathEngine().CreatePerformanceCounters());
	counters->Synchronise();

	getTestDnns(dnns, randoms, useReference, numOfThreads);
	runMultithreadInference(dnns, numOfThreads);

	counters->Synchronise();
	std::cerr
		<< '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) << " ms."
		<< '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";


	delete counters;
	// delete references first
	for(int i = 1; i < numOfThreads; ++i) {
		delete dnns[i];
	}
	delete dnns[0];
}

} // namespace NeoMLTest

//------------------------------------------------------------------------------------------------

TEST(ReferenceDnnTest, ReferenceDnnInferenceTest)
{
	const int numOfThreads = 4;
	CArray<CDnn*> dnns;
	CArray<CRandom> randoms = { CRandom(0x123) };

	getTestDnns(dnns, randoms, true, numOfThreads);
	runMultithreadInference(dnns, numOfThreads);

	for(int i = 0; i < numOfThreads - 1; ++i) {
		EXPECT_TRUE(CompareBlobs(
			*(static_cast<CSourceLayer*>(dnns[i]->GetLayer("in").Ptr())->GetBlob()),
			*(static_cast<CSourceLayer*>(dnns[i + 1]->GetLayer("in").Ptr())->GetBlob()))
		);

		EXPECT_TRUE(CompareBlobs(
			*(static_cast<CSinkLayer*>(dnns[i]->GetLayer("sink").Ptr())->GetBlob()),
			*(static_cast<CSinkLayer*>(dnns[i + 1]->GetLayer("sink").Ptr())->GetBlob()))
		);
	}

	// delete references first
	for(int i = 1; i < numOfThreads; ++i) {
		delete dnns[i];
	}
	delete dnns[0];
}

TEST(ReferenceDnnTest, CDnnReferenceRegisterTest)
{
	// Implement scenario - learn dnn, use multihtread inference, learn again
	const int numOfThreads = 4;

	CArray<CReferenceDnnTestParam> taskParams;
	CArray<CDnn*> dnns;

	// 1.Create and learn dnn
	CRandom random(0x123);
	dnns.Add(createDnn(random));

	CPtr<CDnnBlob> srcBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 8, 20, 30, 10 });
	initializeBlob(srcBlob.Ptr(), random, 0, 1);
	static_cast<CSourceLayer*>(dnns[0]->GetLayer("in").Ptr())->SetBlob(srcBlob);

	CPtr<CSourceLayer> labels = Source(*dnns[0], "labels");
	CPtr<CDnnBlob> labelBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, 10 });
	initializeBlob(labelBlob.Ptr(), random, 0, 1);
	labels->SetBlob(labelBlob);

	CPtr<CL1LossLayer> loss = L1Loss()("loss", dnns[0]->GetLayer("fc3").Ptr(), labels.Ptr());
	CPtr<CDnnSimpleGradientSolver> solver = new CDnnSimpleGradientSolver(MathEngine());
	dnns[0]->SetSolver(solver);

	for(int i = 0; i < 10; ++i) {
		dnns[0]->RunAndLearnOnce();
	}

	// 2. Run mulithread inference
	dnns[0]->DeleteLayer("labels");
	dnns[0]->DeleteLayer("loss");

	for(int i = 1; i < numOfThreads; ++i) {
		dnns.Add(dnns[0]->CreateReferenceDnn());

		CPtr<CDnnBlob> srcBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 8, 20, 30, 10 });
		initializeBlob(srcBlob.Ptr(), random, 0, 1);
		static_cast<CSourceLayer*>(dnns[i]->GetLayer("in").Ptr())->SetBlob(srcBlob);
	}

	EXPECT_TRUE(!dnns[0]->IsLearningEnabled());
	runMultithreadInference(dnns, numOfThreads);

	// 3. Learn again
	for(int i = 1; i < numOfThreads; ++i) {
		delete dnns[i];
	}

	dnns[0]->AddLayer(*labels);
	dnns[0]->AddLayer(*loss);

	EXPECT_TRUE(dnns[0]->IsLearningEnabled());
	for(int i = 0; i < 10; ++i) {
		dnns[0]->RunAndLearnOnce();
	}
	delete dnns[0];
}

TEST(ReferenceDnnTest, DISABLED_PerfomanceReferenceDnnsThreads)
{
	perfomanceTest(true);
}

TEST(ReferenceDnnTest, DISABLED_PerfomanceDnnsThreads)
{
	perfomanceTest(false);
}
