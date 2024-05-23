// © 2023 ABBYY
// Author: Pavel Voropaev
// System: NLC

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/BertLayer.h>

namespace NeoML {

// Имена слоев соответствуют pytorch
const char* const CBertLayer::WordEmbeddingsLayerName = "embeddings.word_embeddings";
const char* const CBertLayer::PositionEmbeddingsLayerName = "embeddings.position_embeddings";
const char* const CBertLayer::LayerNormEmbeddingsLayerName = "embeddings.LayerNorm";
const char* const CBertLayer::EmbeddingsDropoutLayerName = "embeddings.dropout";

const char* const CBertLayer::EncoderLayerNamePrefix = "encoder.layer.";
const char* const CBertLayer::MaskLayerName = "mask";

// Только у головы
const char* const CBertLayer::HeadDenseLayerName = "lm_head.dense";
const char* const CBertLayer::HeadLayerNormLayerName = "lm_head.layer_norm";
const char* const CBertLayer::HeadDecoderLayerName = "lm_head.decoder";
const char* const CBertLayer::HeadDecoderBiasLayerName = "lm_head.decoder.bias";
// Слои без весов и настроек, вряд ли понадобятся снаружи
const char* const headGeluLayerName = "lm_head.gelu";
const char* const headTransformLayerName = "lm_head.kostil";
const char* const headTransformBackLayerName = "lm_head.kostil-obratno";

REGISTER_NEOML_LAYER(CBertLayer, "RobertaLayer")

	CBertLayer::CBertLayer(IMathEngine& mathEngine, const CParam& param) :
	CCompositeLayer(mathEngine, "RobertaLayer")
{
	RebuildLayer(param);
}

void CBertLayer::RebuildLayer(const CParam& _param)
{
	assert(min(_param.AttentionFeedForwardSize,
		_param.AttentionHeadCount,
		_param.EncoderLayerCount,
		_param.HiddenSize,
		_param.VocabSize) > 0);
	assert(_param.HiddenSize % _param.AttentionHeadCount == 0);
	param = _param;

	encoder.DeleteAll();
	DeleteAllLayers();
	buildLayer();
}

void CBertLayer::Serialize(CArchive& archive)
{
	const int version = archive.SerializeVersion(bertLayerArchiveVersion);
	CCompositeLayer::Serialize(archive);
	archive.Serialize(dropoutRate);
	archive.Serialize(frozenLayersCount);
	archive.Serialize(param.AttentionFeedForwardSize);
	archive.Serialize(param.AttentionHeadCount);
	archive.Serialize(param.VocabSize);
	archive.Serialize(param.HiddenSize);
	archive.Serialize(param.EncoderLayerCount);
	archive.Serialize(param.UseMask);
	if (version >= 1) {
		archive.Serialize(param.Head);
	}
	else {
		param.Head = THead::None;
	}

	if (archive.IsLoading()) {
		fillLayerPointers();
	}
}

void CBertLayer::SetFrozenLayersCount(int _frozenLayersCount)
{
	assert(NotFound <= _frozenLayersCount && _frozenLayersCount <= param.EncoderLayerCount);
	frozenLayersCount = _frozenLayersCount;

	if (frozenLayersCount > NotFound) {
		wordEmbeddings->DisableLearning();
		positionEmbeddings->DisableLearning();
		layerNormEmbeddings->DisableLearning();
	}
	else {
		wordEmbeddings->EnableLearning();
		positionEmbeddings->EnableLearning();
		layerNormEmbeddings->EnableLearning();
	}

	for (int i = 0; i < frozenLayersCount; ++i) {
		encoder[i]->DisableLearning();
	}
	for (int i = frozenLayersCount; i < param.EncoderLayerCount; ++i) {
		encoder[i]->EnableLearning();
	}
}

void CBertLayer::SetDropoutRate(float rate)
{
	assert(0.f <= rate && rate < 1.f);

	if (rate == dropoutRate) {
		return;
	}
	dropoutRate = rate;

	for (auto* layer : encoder) {
		layer->SetDropoutRate(rate);
	}

	if (rate > 0.f) {
		addDropoutLayer(rate);
		dropoutEmbeddings->SetDropoutRate(rate);
	}
	else {
		removeDropoutLayer();
	}
}

void CBertLayer::RemovePretrainedHead()
{
	assert(param.Head == THead::LMHeadTied);
	param.Head = THead::None;

	SetOutputMapping(1, *encoder.Last(), 0);

	DeleteLayer(HeadDenseLayerName);
	DeleteLayer(headGeluLayerName);
	DeleteLayer(HeadLayerNormLayerName);
	DeleteLayer(HeadDecoderLayerName);
	DeleteLayer(headTransformLayerName);
	DeleteLayer(HeadDecoderBiasLayerName);
	DeleteLayer(headTransformBackLayerName);
}

void CBertLayer::Reshape()
{
	if (param.Head == THead::LMHeadTied) {
		const int batchSize = inputDescs[0].BatchWidth();
		auto* layer = CheckCast<CTransformLayer>(GetLayer(headTransformBackLayerName));
		layer->SetDimensionRule(BD_BatchWidth, CTransformLayer::O_SetSize, batchSize);
	}
	CCompositeLayer::Reshape();
}

void CBertLayer::fillLayerPointers()
{
	wordEmbeddings = CheckCast<CMultichannelLookupLayer>(GetLayer(WordEmbeddingsLayerName));
	positionEmbeddings = CheckCast<CPositionalEmbeddingLayer>(GetLayer(PositionEmbeddingsLayerName));
	layerNormEmbeddings = CheckCast<CObjectNormalizationLayer>(GetLayer(LayerNormEmbeddingsLayerName));
	dropoutEmbeddings = nullptr;
	if (HasLayer(EmbeddingsDropoutLayerName)) {
		dropoutEmbeddings = CheckCast<CDropoutLayer>(GetLayer(EmbeddingsDropoutLayerName));
	}
	if (param.UseMask) {
		encoderMask = CheckCast<CTransformerSourceMaskLayer>(GetLayer(MaskLayerName));
	}
	encoder.DeleteAll();
	encoder.SetBufferSize(param.EncoderLayerCount);
	for (int i = 0; i < param.EncoderLayerCount; ++i) {
		encoder.Add(CheckCast<CTransformerEncoderLayer>(GetLayer(EncoderLayerNamePrefix + Str(i))));
	}
}

void CBertLayer::buildLayer()
{
	// Эмбеддинг по словарю bpe (VocabSize x HiddenSize)
	// Внешний вход #0: батч токенизированных последовательностей
	// Вход:  [1 x batch_size x seq_len x 1 x 1 x 1 x 1]
	// Выход: [1 x batch_size x seq_len x 1 x 1 x 1 x HiddenSize]
	auto wordEmbs = MakeCPtr<CMultichannelLookupLayer>(MathEngine());
	wordEmbs->SetName(WordEmbeddingsLayerName);
	wordEmbs->SetDimensions({ CLookupDimension{ param.VocabSize, param.HiddenSize } });
	wordEmbs->SetUseFrameworkLearning(true);
	AddLayer(*wordEmbs);
	wordEmbeddings = wordEmbs;
	SetInputMapping(0, *wordEmbeddings, 0);

	// Поэлементное суммирование словарных эмбеддингов с позиционными
	// Вход=Выход: [1 x batch_size x seq_len x 1 x 1 x 1 x HiddenSize]
	auto posEmbs = MakeCPtr<CPositionalEmbeddingLayer>(MathEngine());
	posEmbs->SetName(PositionEmbeddingsLayerName);
	posEmbs->SetType(CPositionalEmbeddingLayer::PET_LearnableAddition);
	posEmbs->Connect(*wordEmbs);
	AddLayer(*posEmbs);
	positionEmbeddings = posEmbs;

	// Поэлементная нормализация
	// Вход=Выход: [1 x batch_size x seq_len x 1 x 1 x 1 x HiddenSize]
	auto layerNorm = MakeCPtr<CObjectNormalizationLayer>(MathEngine());
	layerNorm->SetName(LayerNormEmbeddingsLayerName);
	layerNorm->Connect(*posEmbs);
	AddLayer(*layerNorm);
	layerNormEmbeddings = layerNorm;

	if (param.UseMask) {
		// Внешний вход #1: маска, список длин последовательностей
		// Вход 0 - длины последовательностей [1 x batch_size x 1 x 1 x 1 x 1 x 1]
		// Вход 1 - текст после layerNorm     [1 x batch_size x seq_len x 1 x 1 x 1 x HiddenSize]
		// Выход: [1 x batch_size x AttentionHeadCount x 1 x seq_len x 1 x seq_len]
		auto mask = MakeCPtr<CTransformerSourceMaskLayer>(MathEngine());
		mask->SetName(MaskLayerName);
		mask->SetHeadCount(param.AttentionHeadCount);
		AddLayer(*mask);
		encoderMask = mask;
		SetInputMapping(1, *encoderMask, 0);
		mask->Connect(1, *layerNormEmbeddings, 0);
	}

	// Слои трансформера
	// Вход 0 - текст после layerNorm [1 x batch_size x seq_len x 1 x 1 x 1 x HiddenSize]
	// Вход 1 - TransformerSourceMask [1 x batch_size x AttentionHeadCount x 1 x seq_len x 1 x seq_len]
	// Выход: [1 x batch_size x seq_len x 1 x 1 x 1 x HiddenSize]
	for (int i = 0; i < param.EncoderLayerCount; ++i) {
		addEncoderLayer();
	}
	// Внешний выход 0: [1 x batch_size x seq_len x 1 x 1 x 1 x HiddenSize]
	SetOutputMapping(0, *encoder.Last());

	// Опциональный линейный слой
	switch (param.Head) {
	case THead::None:
		break;
	case THead::Pooler:
		addPooler();
		break;
	case THead::LMHead:
		addLmHead(false);
		break;
	case THead::LMHeadTied:
		addLmHead(true);
		break;
	case THead::Count:
		break;
	default:
		assert(false);
	}
	compileTimeAssert(THead::Count == static_cast<THead>(4));
}

void CBertLayer::addDropoutLayer(float rate)
{
	if (dropoutEmbeddings != nullptr) {
		return;
	}
	auto dropout = MakeCPtr<CDropoutLayer>(MathEngine());
	dropout->SetName(EmbeddingsDropoutLayerName);
	dropout->Connect(*layerNormEmbeddings);
	dropout->SetDropoutRate(rate);
	encoder.First()->Connect(*dropout);
	AddLayer(*dropout);
	dropoutEmbeddings = dropout;
}

void CBertLayer::removeDropoutLayer()
{
	if (dropoutEmbeddings == nullptr) {
		return;
	}
	DeleteLayer(*dropoutEmbeddings);
	encoder.First()->Connect(*layerNormEmbeddings);
	dropoutEmbeddings = nullptr;
}

void CBertLayer::addEncoderLayer()
{
	int layerId = encoder.Size();

	auto layer = MakeCPtr<CTransformerEncoderLayer>(MathEngine());
	layer->SetName(EncoderLayerNamePrefix + Str(layerId));
	layer->SetActivation({ AF_GELU, CGELULayer::CParam{ CGELULayer::CM_Precise } });
	layer->SetFeedForwardSize(param.AttentionFeedForwardSize);
	layer->SetHeadCount(param.AttentionHeadCount);
	layer->SetHiddenSize(param.HiddenSize);

	auto* attention = CheckCast<CMultiheadAttentionLayer>(layer->GetLayer("SelfAttention"));
	attention->SetUseMask(param.UseMask);
	attention->SetMaskType(CMultiheadAttentionLayer::MT_Eltwise);

	if (layerId == 0) {
		layer->Connect(*layerNormEmbeddings);
	}
	else {
		layer->Connect(*encoder[layerId - 1]);
	}
	if (param.UseMask) {
		layer->Connect(1, *encoderMask, 0);
	}

	AddLayer(*layer);
	encoder.Add(layer);
}

// Пулинг в виде выбора эмбеддинга CLS (первого) токена
void CBertLayer::addPooler()
{
	// x = SubSequence( 0, 1 )( encoder.Last() );
	// Выход: [1 x batch_size x 1 x 1 x 1 x 1 x HiddenSize]
	auto firstEl = MakeCPtr<CSubSequenceLayer>(MathEngine());
	firstEl->SetStartPos(0);
	firstEl->SetLength(1);
	firstEl->Connect(*encoder.Last());
	AddLayer(*firstEl);

	// x = FullyConnected( param.HiddenSize )( HeadDenseLayerName, x );
	// Внешний выход #1: [1 x batch_size x 1 x 1 x 1 x 1 x HiddenSize]
	auto dense = MakeCPtr<CFullyConnectedLayer>(MathEngine(), HeadDenseLayerName);
	dense->SetNumberOfElements(param.HiddenSize);
	dense->Connect(*firstEl);
	AddLayer(*dense);
	SetOutputMapping(1, *dense, 0);

	// В оригинале здесь еще активация:
	// x = Tanh()( x );
	// Тангенс - спорная активация, выдаем без нее, можете сами сделать так, как нужно вам.
}

// Линейный слой + "декодер" в размерность словаря
void CBertLayer::addLmHead(bool tied)
{
	// x = FullyConnected( param.HiddenSize )( HeadDenseLayerName, x );
	auto dense = MakeCPtr<CFullyConnectedLayer>(MathEngine(), HeadDenseLayerName);
	dense->SetNumberOfElements(param.HiddenSize);
	dense->Connect(*encoder.Last());
	AddLayer(*dense);

	// x = Gelu()( x );
	auto gelu = MakeCPtr<CGELULayer>(MathEngine());
	gelu->SetName(headGeluLayerName);
	gelu->SetCalculationMode(CGELUActivationParam::TCalculationMode::CM_Precise);
	gelu->Connect(*dense);
	AddLayer(*gelu);

	// x = ObjectNormalization()( HeadLayerNormLayerName, x );
	auto layerNorm = MakeCPtr<CObjectNormalizationLayer>(MathEngine());
	layerNorm->SetName(HeadLayerNormLayerName);
	layerNorm->Connect(*gelu);
	AddLayer(*layerNorm);

	if (tied) {
		// x = TiedEmbeddings( WordEmbeddingsLayerName, 0 )( HeadDecoderLayerName, x );
		auto decoder = MakeCPtr<CTiedEmbeddingsLayer>(MathEngine());
		decoder->SetName(HeadDecoderLayerName);
		decoder->SetEmbeddingsLayerName(WordEmbeddingsLayerName);
		decoder->SetChannelIndex(0);
		decoder->Connect(*layerNorm);
		AddLayer(*decoder);

		// Далее нам нужен обучаемый bias, которого нет в оригинальных эмбеддингах. Начинаются танцы на костылях:
		// https://jira.abbyy.com/browse/TL-295
		// x = Transform( 1, batch_size * seq_len, 1, 1, 1, 1, param.VocabSize )( x );
		auto transform = MakeCPtr<CTransformLayer>(MathEngine());
		transform->SetName(headTransformLayerName);
		transform->SetDimensionRule(BD_BatchWidth, CTransformLayer::O_Remainder, 0);
		transform->SetDimensionRule(BD_ListSize, CTransformLayer::O_SetSize, 1);
		transform->Connect(*decoder);
		AddLayer(*transform);

		// x = PositionalEmbedding( PET_LearnableAddition )( HeadDecoderBiasLayerName, x );
		// x->SetMaxListSize( param.VocabSize )
		auto posEmbs = MakeCPtr<CPositionalEmbeddingLayer>(MathEngine());
		posEmbs->SetName(HeadDecoderBiasLayerName);
		posEmbs->SetType(CPositionalEmbeddingLayer::PET_LearnableAddition);
		posEmbs->SetMaxListSize(1);
		posEmbs->Connect(*transform);
		AddLayer(*posEmbs);

		// x = Transform( 1, batch_size, seq_len, 1, 1, 1, param.VocabSize )( x );
		transform = MakeCPtr<CTransformLayer>(MathEngine());
		transform->SetName(headTransformBackLayerName);
		// BD_BatchWidth to be reshaped in Reshape()
		transform->SetDimensionRule(BD_BatchWidth, CTransformLayer::O_SetSize, 1 /* batch_size */);
		transform->SetDimensionRule(BD_ListSize, CTransformLayer::O_Remainder, 0);
		transform->SetDimensionRule(BD_Channels, CTransformLayer::O_SetSize, param.VocabSize);
		transform->Connect(*posEmbs);
		AddLayer(*transform);

		SetOutputMapping(1, *transform, 0);
	}
	else {
		// x = FullyConnected( param.VocabSize )( HeadDecoderLayerName, x );
		auto decoder = MakeCPtr<CFullyConnectedLayer>(MathEngine(), HeadDecoderLayerName);
		decoder->SetNumberOfElements(param.VocabSize);
		decoder->Connect(*layerNorm);
		AddLayer(*decoder);
		SetOutputMapping(1, *decoder, 0);
	}
}

} // namespace NeoML
