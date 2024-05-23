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

static void initializeBertInput(CRandom& random, CPtr<CDnnBlob> blob, int lower, int higher)
{
	int numOfThreads = 4;
	CPtr<CDnnUniformInitializer> uniformInitializer = new CDnnUniformInitializer(CRandom(0x123), 0, 50000);
	CArray<int> tempData;
	tempData.SetSize(blob->GetDataSize());

	int* data = tempData.GetPtr();
	for (int i = 0; i < tempData.Size(); ++i) {
		int num = (int)(uniformInitializer->Random()).Uniform(lower, higher);
		*data++ = num;
	}

	blob->CopyFrom(tempData.GetPtr());
}

} // namespace NeoMLTest

//------------------------------------------------------------------------------------------------

static const int paragraphs_cnt = 100;
static const int batch_size = 4;
static const int numOfThreads = 4;
static const int numOfMeasures = 1;

TEST(ThreadTest, BertTest)
{
    CRandom random(0x123);
    CDnn bert(random, MathEngine());
    CArchiveFile file(".\\RobertaTextSeqmentationInference.dnn", CArchive::load);
    CArchive archive(&file, CArchive::SD_Loading);

    bert.Serialize(archive);

    CPtr<CDnnBlob> input_ids = CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, batch_size, 512, 1, 1, 1, 1 });
	initializeBertInput(CRandom(0x123), input_ids, 0, 50000);
    static_cast<CSourceLayer*>(bert.GetLayer("input_ids").Ptr())->SetBlob(input_ids);

    CPtr<CDnnBlob> clsPositions = CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, 1, 1, 1, 1, 1, paragraphs_cnt });
	initializeBertInput(CRandom(0x123), clsPositions, 0, 512);
    static_cast<CSourceLayer*>(bert.GetLayer("cls_positions").Ptr())->SetBlob(clsPositions);

    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();

    for (int i = 0; i < numOfMeasures; ++i)
        bert.RunOnce();

    counters->Synchronise();
    std::cerr
        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) / numOfMeasures << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";
}

TEST(ThreadTest, BertThreadTest)
{
    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();

    CRandom random(0x123);
    CDnn* bert = new CDnn(random, MathEngine());
    auto t = FileSystem::GetCurrentDir();
    CArchiveFile file(".\\RobertaTextSeqmentationInference.dnn", CArchive::load);
    CArchive archive(&file, CArchive::SD_Loading);

    bert->Serialize(archive);

    CArray<CDnn*> dnns;
    CObjectArray<CDnnBlob> input_ids;
    CObjectArray<CDnnBlob> clsPositions;
    CArray<CReferenceDnnTestParam> taskParams;

    CPtr<CDnnBlob> input_ids0 = CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, batch_size / numOfThreads, 512, 1, 1, 1, 1 });
    initializeBertInput(CRandom(0x123), input_ids0, 0, 50000);
    static_cast<CSourceLayer*>(bert->GetLayer("input_ids").Ptr())->SetBlob(input_ids0);


    CPtr<CDnnBlob> clsPositions0 = CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, 1, 1, 1, 1, 1, paragraphs_cnt });
	initializeBertInput(CRandom(0x123), clsPositions0, 0, 512);
    static_cast<CSourceLayer*>(bert->GetLayer("cls_positions").Ptr())->SetBlob(clsPositions0);
    taskParams.Add({ bert });

    for (int i = 0; i < numOfThreads - 1; ++i) {
		dnns.Add(bert->CreateReferenceDnn());
        input_ids.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, batch_size / numOfThreads, 512, 1, 1, 1, 1 }));
        clsPositions.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, 1, 1, 1, 1, 1, paragraphs_cnt }));

        taskParams.Add({ dnns[i] });

		initializeBertInput(CRandom(0x123), input_ids[i], 0, 50000);
		initializeBertInput(CRandom(0x123), clsPositions[i], 0, 512);

        static_cast<CSourceLayer*>(dnns[i]->GetLayer("cls_positions").Ptr())->SetBlob(clsPositions[i]);
        static_cast<CSourceLayer*>(dnns[i]->GetLayer("input_ids").Ptr())->SetBlob(input_ids[i]);
    }

    IThreadPool* pool = CreateThreadPool(numOfThreads);
    counters->Synchronise();

    for (int j = 0; j < numOfMeasures; ++j) {
        for (int i = 0; i < numOfThreads; ++i) {
            pool->AddTask(i, runDnn, &(taskParams[i]));
        }
        pool->WaitAllTask();
    }

    counters->Synchronise();
    std::cerr
        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) / numOfMeasures << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";

    /*for (int i = 0; i < numOfThreads - 1; ++i) {
        EXPECT_TRUE( CompareBlobs( *( static_cast<CSinkLayer*>(bert->GetLayer("output").Ptr())->GetBlob() ),
            *( static_cast<CSinkLayer*>(dnns[i]->GetLayer("output").Ptr())->GetBlob() ), 1e-5f ) );
    }*/

    for (int i = 0; i < numOfThreads - 1; ++i) {
        delete dnns[i];
    }
    delete bert;
}

TEST(ThreadTest, DummyBertThread)
{
    CRandom random(0x123);
    CArray<CDnn*> dnns;
    CObjectArray<CDnnBlob> input_ids;
    CObjectArray<CDnnBlob> clsPositions;
    CArray<CReferenceDnnTestParam> taskParams;

    for (int i = 0; i < numOfThreads; ++i) {
        CDnn* dnn = new CDnn(random, MathEngine());
        CArchiveFile file(".\\RobertaTextSeqmentationInference.dnn", CArchive::load);
        CArchive archive(&file, CArchive::SD_Loading);
        dnn->Serialize(archive);
        dnns.Add(dnn);
        input_ids.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, batch_size / numOfThreads, 512, 1, 1, 1, 1 }));
        clsPositions.Add(CDnnBlob::CreateTensor(MathEngine(), CT_Int, { 1, 1, 1, 1, 1, 1, paragraphs_cnt }));

        taskParams.Add({ dnns[i] });

		initializeBertInput(CRandom(0x123), clsPositions[i], 0, 512);
		initializeBertInput(CRandom(0x123), input_ids[i], 0, 50000);

        static_cast<CSourceLayer*>(dnns[i]->GetLayer("cls_positions").Ptr())->SetBlob(clsPositions[i]);
        static_cast<CSourceLayer*>(dnns[i]->GetLayer("input_ids").Ptr())->SetBlob(input_ids[i]);
    }


    std::unique_ptr<IPerformanceCounters> counters(MathEngine().CreatePerformanceCounters());
    counters->Synchronise();
    IThreadPool* pool = CreateThreadPool(numOfThreads);

    for (int j = 0; j < numOfMeasures; ++j) {
        for (int i = 0; i < numOfThreads; ++i) {
            pool->AddTask(i, runDnn, &(taskParams[i]));
        }
        pool->WaitAllTask();
    }

    counters->Synchronise();
    std::cerr
        << '\n' << "Time: " << (double((*counters)[0].Value) / 1000000) / numOfMeasures << " ms."
        << '\t' << "Peak.Mem: " << (double(MathEngine().GetPeakMemoryUsage()) / 1024 / 1024) << " MB \n";

    for (int i = 0; i < numOfThreads; ++i) {
        delete dnns[i];
    }
}
