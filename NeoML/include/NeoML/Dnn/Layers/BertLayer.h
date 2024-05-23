// © 2023 ABBYY
// Author: Pavel Voropaev
// Description: NeoML-layer implementing BERT architecture

#pragma once


#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/MultichannelLookupLayer.h>
#include <NeoML/Dnn/Layers/PositionalEmbeddingLayer.h>
#include <NeoML/Dnn/Layers/ObjectNormalizationLayer.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/TransformerSourceMaskLayer.h>
#include <NeoML/Dnn/Layers/TransformerLayer.h>
#include <NeoML/Dnn/Layers/TransformLayer.h>
#include <NeoML/Dnn/Layers/TiedEmbeddingsLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/MultiheadAttentionLayer.h>
#include <NeoML/Dnn/Layers/SubSequenceLayer.h>

namespace NeoML {

// Слой BERT (может загружать веса обычного BERT, RoBERTa и, возможно, некоторых других вариаций)
//
// Вход #0: батч токенизированных последовательностей
// [1 x batch_size x seq_len x 1 x 1 x 1 x 1]
// Вход #1: маска, список длин последовательностей
// [1 x batch_size x 1 x 1 x 1 x 1 x 1]
//
// Выход 0: потокенные эмбеддинги
// [1 x batch_size x seq_len x 1 x 1 x 1 x HiddenSize]
// Выход 1: выход линейного слоя THead
// None: (отсутствует)
// Pooler: [1 x batch_size x 1 x 1 x 1 x 1 x HiddenSize]
// LMHead(Tied): [1 x batch_size x seq_len x 1 x 1 x 1 x VocabSize]
class NEOML_API CBertLayer : public NeoML::CCompositeLayer {
	NEOML_DNN_LAYER(CBertLayer)
public:
	// Опциональный линейный слой после трансформера.
	enum class THead {
		None,  // Без линейного слоя
		Pooler,  // Выход линейного слоя (HiddenSize) на CLS (первом) токене
		LMHead,  // Линейный слой (HiddenSize) + gelu + LayerNorm + линейный слой-декодер (HiddenSize x VocabSize)
		LMHeadTied, // То же, но веса декодера связаны с весами входного словарного эмбеддинга

		Count
	};

	// Настройки слоя. Выставлены по умолчанию для BERT-base / RoBERTa-base.
	struct CParam {
		// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88165
		CParam() {}

		// Размерность словаря входного текста
		int VocabSize = 50265;
		// Количество слоев TransformerEncoder
		int EncoderLayerCount = 12;
		// Внутренняя размерность, в которой работают большинство слоев
		int HiddenSize = 768;
		// Количество "голов" attention
		int AttentionHeadCount = 12;
		// Размерность полносвязного слоя FeedForward в attention
		int AttentionFeedForwardSize = 3072;
		// Использовать ли маску в attention
		bool UseMask = false;
		// Максимальная длина обрабатываемой последовательности
		int MaxSeqLen = 512;
		// Линейный слой после трансформера
		THead Head = THead::None;
	};

	explicit CBertLayer(NeoML::IMathEngine& mathEngine, const CParam& param = CParam{});

	void RebuildLayer(const CParam& param);
	CParam GetParam() const { return param; }

	void Serialize(CArchive& archive) override;

	// Выключить обучение N первых слоев трансформера.
	// 0 -- заморожены только эмбеддинги. NotFound -- разморожено всё.
	void SetFrozenLayersCount(int frozenLayersCount);
	int GetFrozenLayersCount() const { return frozenLayersCount; }

	// Включить дропаут или выключить (слои dropout удаляются). По умолчанию выключен.
	void SetDropoutRate(float rate);
	float GetDropoutRate() const { return dropoutRate; }

	// Удаляет предобученную голову, загружаемую из архива с весами
	void RemovePretrainedHead();

	NeoML::IMathEngine& GetMathEngine() const { return GetInternalDnn()->GetMathEngine(); }

	// Имена запчастей для внешнего доступа
	// Входные эмбеддинги
	static const char* const WordEmbeddingsLayerName;
	static const char* const PositionEmbeddingsLayerName;
	static const char* const LayerNormEmbeddingsLayerName;
	static const char* const EmbeddingsDropoutLayerName;

	// Слои TransformerEncoder с именами f"{EncoderLayerNamePrefix}{i}"
	static const char* const EncoderLayerNamePrefix;
	static const char* const MaskLayerName;

	// Головы (см. THead)
	// Только для Pooler, LMHead, LMHeadTied
	static const char* const HeadDenseLayerName;
	// Только для LMHead, LMHeadTied
	static const char* const HeadLayerNormLayerName;
	// Только для LMHead, LMHeadTied
	static const char* const HeadDecoderLayerName;
	// Только для LMHeadTied (bias отдельно от HeadDecoder, т.к. у TiedEmbeddings его нет)
	// Это слой PositionalEmbedding
	static const char* const HeadDecoderBiasLayerName;

protected:
	~CBertLayer() override = default;
	void Reshape() override;

private:
	static const int bertLayerArchiveVersion = 1;

	CParam param;
	int frozenLayersCount = NotFound;
	float dropoutRate = 0.;

	// Для быстрого доступа при заморозке-разморозке
	// RobertaEmbeddings
	NeoML::CMultichannelLookupLayer* wordEmbeddings{};
	NeoML::CPositionalEmbeddingLayer* positionEmbeddings{};
	NeoML::CObjectNormalizationLayer* layerNormEmbeddings{};
	NeoML::CDropoutLayer* dropoutEmbeddings{};
	// RobertaEncoder
	NeoML::CTransformerSourceMaskLayer* encoderMask{};
	CArray<NeoML::CTransformerEncoderLayer*> encoder;

	void fillLayerPointers();
	void buildLayer();
	void addEncoderLayer();
	void addDropoutLayer(float rate);
	void removeDropoutLayer();
	void addPooler();
	void addLmHead(bool tied);
};

} // namespace NeoML
