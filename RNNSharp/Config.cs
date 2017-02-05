using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class LayerConfig
    {
        public int LayerSize;
        public LayerType LayerType;
    }

    public class LSTMLayerConfig : LayerConfig
    {
        public LSTMLayerConfig()
        {
            LayerType = LayerType.LSTM;
        }
    }

    public class DropoutLayerConfig : LayerConfig
    {
        public float DropoutRatio;

        public DropoutLayerConfig()
        {
            LayerType = LayerType.DropOut;
        }
    }

    public class SampledSoftmaxLayerConfig : SoftmaxLayerConfig
    {
        public int NegativeSampleSize;

        public SampledSoftmaxLayerConfig()
        {
            LayerType = LayerType.SampledSoftmax;
        }
    }

    public class SoftmaxLayerConfig : LayerConfig
    {
        public SoftmaxLayerConfig()
        {
            LayerType = LayerType.Softmax;
        }
    }

    public class SimpleLayerConfig : LayerConfig
    {
        public SimpleLayerConfig()
        {
            LayerType = LayerType.Simple;
        }
    }

    public class Config
    {
        //Settings for model type
        private static readonly string MODEL_TYPE = "MODEL_TYPE";

        //Settings for current directory
        private static readonly string CURRENT_DIRECTORY = "CURRENT_DIRECTORY";

        //Settings for hidden layers
        //Format: [Type:Size], example:
        //HIDDEN_LAYER = 300:LSTM,200:BPTT
        private static readonly string HIDDEN_LAYER = "HIDDEN_LAYER";

        //Settings for output layer
        //Example: OUTPUT_LAYER = Softmax
        private static readonly string OUTPUT_LAYER = "OUTPUT_LAYER";

        private static readonly string CRF_LAYER = "CRF_LAYER";

        //Settings for model file path
        private static readonly string MODEL_FILEPATH = "MODEL_FILEPATH";

        //Settings for sparse feature
        private static readonly string TFEATURE_CONTEXT = "TFEATURE_CONTEXT";

        private static readonly string TFEATURE_FILENAME = "TFEATURE_FILENAME";

        private static readonly string RT_FEATURE_CONTEXT = "RTFEATURE_CONTEXT";

        //Settings for sparse feature weight type
        private static readonly string TFEATURE_WEIGHT_TYPE = "TFEATURE_WEIGHT_TYPE";

        //Settings for pretrained model type
        private static readonly string PRETRAIN_TYPE = "PRETRAIN_TYPE";

        //Settings for word embedding model
        private static readonly string WORDEMBEDDING_CONTEXT = "WORDEMBEDDING_CONTEXT";

        private static readonly string PRETRAINEDMODEL_FILENAME = "WORDEMBEDDING_FILENAME";
        private static readonly string PRETRAINEDMODEL_RAW_FILENAME = "WORDEMBEDDING_RAW_FILENAME";
        private static readonly string PRETRAINEDMODEL_COLUMN = "WORDEMBEDDING_COLUMN";

        //Settings for auto encoder model
        private static readonly string AUTOENCODER_CONFIG = "AUTOENCODER_CONFIG";

        //Settings for auto encoder model in sequence-to-sequence task
        private static readonly string SEQ2SEQ_AUTOENCODER_CONFIG = "SEQ2SEQ_AUTOENCODER_CONFIG";

        //Settings for model direction type
        private static readonly string MODEL_DIRECTION = "MODEL_DIRECTION";

        //Raw configuration
        private static ConfigUtils config;

        private RNNDecoder autoEncoder;

        //Feature context offset for sparse and dense feature set
        private Dictionary<string, List<int>> featureContext;

        public bool IsCRFTraining;
        public LayerConfig OutputLayerConfig;
        private WordEMWrapFeaturizer preTrainedModel;
        private int preTrainedModelColumn;
        private PRETRAIN_TYPE preTrainType;
        public RNNDecoder Seq2SeqAutoEncoder;
        public int SparseFeatureSize;
        private TFEATURE_WEIGHT_TYPE_ENUM tFeatureWeightType;
        private TemplateFeaturizer tFeaturizer;

        public Config(string strFeatureConfigFileName, TagSet tagSet)
        {
            LoadFeatureConfigFromFile(strFeatureConfigFileName);
            TagSet = tagSet;
            ComputingFeatureSize();
        }

        public TagSet TagSet { get; set; }
        public MODELTYPE ModelType { get; set; }
        private string currentDirectory { get; set; }
        public List<LayerConfig> HiddenLayersConfig { get; set; }
        public string ModelFilePath { get; set; }
        public MODELDIRECTION ModelDirection { get; set; }

        private string GetFilePath(string currentDirectory, string filePath)
        {
            return Path.IsPathRooted(filePath) ? filePath : Path.Combine(currentDirectory, filePath);
        }

        //The format of configuration file
        public void LoadFeatureConfigFromFile(string configFilePath)
        {
            //Load configuration file
            config = new ConfigUtils();
            config.LoadFile(configFilePath);

            //Get current directory from configuration file
            currentDirectory = config.GetValueOptional(CURRENT_DIRECTORY);
            if (string.IsNullOrEmpty(currentDirectory))
            {
                currentDirectory = Environment.CurrentDirectory;
            }
            Logger.WriteLine($"Current directory : {currentDirectory}");

            //Get model file path
            ModelFilePath = GetFilePath(currentDirectory, config.GetValueRequired(MODEL_FILEPATH));
            Logger.WriteLine($"Main model is located at {ModelFilePath}");

            featureContext = new Dictionary<string, List<int>>();

            var isCRFTraining = config.GetValueOptional(CRF_LAYER);
            IsCRFTraining = false;
            if (string.IsNullOrEmpty(isCRFTraining) == false)
            {
                IsCRFTraining = bool.Parse(isCRFTraining);
            }

            SetHiddenLayers();
            SetOutputLayers();
            SetPretrainedModel();
            SetTFeatures();

            //Load model type
            ModelType = config.GetValueRequired(MODEL_TYPE)
                .Equals(MODELTYPE.SeqLabel.ToString(), StringComparison.InvariantCultureIgnoreCase)
                ? MODELTYPE.SeqLabel
                : MODELTYPE.Seq2Seq;
            Logger.WriteLine($"Model type: {ModelType}");

            ModelDirection = config.GetValueRequired(MODEL_DIRECTION)
                .Equals(MODELDIRECTION.Forward.ToString(), StringComparison.InvariantCultureIgnoreCase)
                ? MODELDIRECTION.Forward
                : MODELDIRECTION.BiDirectional;
            Logger.WriteLine($"Model direction: {ModelDirection}");

            //Load auto-encoder model for sequence-to-sequence. This model is used to encode source sequence
            if (ModelType == MODELTYPE.Seq2Seq)
            {
                var seqAutoEncoderConfigFilePath = GetFilePath(currentDirectory,
                    config.GetValueRequired(SEQ2SEQ_AUTOENCODER_CONFIG));
                Logger.WriteLine(
                    $"Loading auto encoder model for sequnce-to-sequence task. Config file = '{seqAutoEncoderConfigFilePath}'");

                Seq2SeqAutoEncoder = InitializeAutoEncoder(seqAutoEncoderConfigFilePath);
            }

            //Check if settings are validated
            CheckSettings();
        }

        private void CheckSettings()
        {
            if (ModelDirection == MODELDIRECTION.BiDirectional && ModelType == MODELTYPE.Seq2Seq)
            {
                throw new Exception("Bi-directional RNN model doesn't support sequence-to-sequence model.");
            }

            if (IsRunTimeFeatureUsed() && ModelDirection == MODELDIRECTION.BiDirectional)
            {
                throw new Exception("Run time feature is not available for bi-directional model.");
            }
        }

        private void SetTFeatures()
        {
            //Load template feature set
            var tfeatureFilePath = GetFilePath(currentDirectory, config.GetValueRequired(TFEATURE_FILENAME));
            Logger.WriteLine($"Loading template feature set from {tfeatureFilePath}");
            tFeaturizer = new TemplateFeaturizer(tfeatureFilePath);

            var tfeatureWeightType = config.GetValueRequired(TFEATURE_WEIGHT_TYPE);
            tFeatureWeightType = tfeatureWeightType.Equals("binary", StringComparison.InvariantCultureIgnoreCase)
                ? TFEATURE_WEIGHT_TYPE_ENUM.BINARY
                : TFEATURE_WEIGHT_TYPE_ENUM.FREQUENCY;
            Logger.WriteLine($"TFeature weight type: {tfeatureWeightType}");

            var tfeatureContext = config.GetValueRequired(TFEATURE_CONTEXT);
            featureContext.Add(TFEATURE_CONTEXT, new List<int>());
            foreach (var contextOffset in tfeatureContext.Split(','))
            {
                featureContext[TFEATURE_CONTEXT].Add(int.Parse(contextOffset));
            }
            Logger.WriteLine($"TFeature context: {tfeatureContext}");
        }

        private void SetPretrainedModel()
        {
            //Load pre-trained model. It supports embedding model and auto-encoder model
            var preTrainTypeValue = config.GetValueRequired(PRETRAIN_TYPE);
            Logger.WriteLine("Pretrain type: {0}", preTrainTypeValue);

            if (preTrainTypeValue.Equals(RNNSharp.PRETRAIN_TYPE.AutoEncoder.ToString(),
                StringComparison.InvariantCultureIgnoreCase))
            {
                preTrainType = RNNSharp.PRETRAIN_TYPE.AutoEncoder;
                var autoEncoderConfigFilePath = GetFilePath(currentDirectory,
                    config.GetValueRequired(AUTOENCODER_CONFIG));
                Logger.WriteLine($"Loading auto encoder model. Config file = '{autoEncoderConfigFilePath}'");
                autoEncoder = InitializeAutoEncoder(autoEncoderConfigFilePath);
            }
            else if (preTrainTypeValue.Equals(RNNSharp.PRETRAIN_TYPE.Embedding.ToString(),
                StringComparison.InvariantCultureIgnoreCase))
            {
                preTrainType = RNNSharp.PRETRAIN_TYPE.Embedding;
                var preTrainedModelFilePath = config.GetValueOptional(PRETRAINEDMODEL_FILENAME);
                if (string.IsNullOrEmpty(preTrainedModelFilePath) == false)
                {
                    preTrainedModelFilePath = GetFilePath(currentDirectory, preTrainedModelFilePath);
                    if (preTrainedModel != null)
                    {
                        throw new ArgumentException(
                            "Static pretrained model has already been loaded. Please check if settings is duplicated in configuration file.");
                    }
                    Logger.WriteLine($"Loading pretrained embedding model: {preTrainedModelFilePath}");
                    preTrainedModel = new WordEMWrapFeaturizer(preTrainedModelFilePath);
                }

                var preTrainedRawModelFilePath = config.GetValueOptional(PRETRAINEDMODEL_RAW_FILENAME);
                if (string.IsNullOrEmpty(preTrainedRawModelFilePath) == false)
                {
                    preTrainedRawModelFilePath = GetFilePath(currentDirectory, preTrainedRawModelFilePath);
                    if (preTrainedModel != null)
                    {
                        throw new ArgumentException(
                            "Static pretrained model has already been loaded. Please check if settings is duplicated in configuration file.");
                    }
                    Logger.WriteLine($"Loading pretrained embedding model {preTrainedRawModelFilePath} in text format");
                    preTrainedModel = new WordEMWrapFeaturizer(preTrainedRawModelFilePath, true);
                }

                preTrainedModelColumn = int.Parse(config.GetValueRequired(PRETRAINEDMODEL_COLUMN));
                Logger.WriteLine("Pretrained model feature column: {0}", preTrainedModelColumn);

                var preTrainedModelContext = config.GetValueRequired(WORDEMBEDDING_CONTEXT);
                featureContext.Add(WORDEMBEDDING_CONTEXT, new List<int>());
                foreach (var contextOffset in preTrainedModelContext.Split(','))
                {
                    featureContext[WORDEMBEDDING_CONTEXT].Add(int.Parse(contextOffset));
                }
                Logger.WriteLine($"Pretrained model context offset : {preTrainedModelContext}");
            }
            else
            {
                preTrainType = RNNSharp.PRETRAIN_TYPE.None;
                Logger.WriteLine("No pretrained model for this training.");
            }
        }

        private void SetOutputLayers()
        {
            //Settings for output layer
            var outputLayer = config.GetValueRequired(OUTPUT_LAYER);
            var items = outputLayer.Split(':');
            var sLayerType = items[0];
            var outputLayerType = LayerType.None;
            foreach (
                var type in
                    Enum.GetValues(typeof(LayerType))
                        .Cast<LayerType>()
                        .Where(type => sLayerType.Equals(type.ToString(), StringComparison.InvariantCultureIgnoreCase)))
            {
                outputLayerType = type;
                break;
            }

            if (IsCRFTraining == true && outputLayerType != LayerType.Simple)
            {
                throw new ArgumentException($"For RNN-CRF model, its output layer type must be simple layer.");
            }

            switch (outputLayerType)
            {
                case LayerType.Softmax:
                    OutputLayerConfig = new SoftmaxLayerConfig();
                    Logger.WriteLine("Initialize configuration for softmax layer.");
                    break;

                case LayerType.SampledSoftmax:
                    var sampledSoftmaxLayerConfig = new SampledSoftmaxLayerConfig { NegativeSampleSize = int.Parse(items[1]) };
                    OutputLayerConfig = sampledSoftmaxLayerConfig;

                    Logger.WriteLine(
                        $"Initialize configuration for sampled Softmax layer. Negative sample size = '{sampledSoftmaxLayerConfig.NegativeSampleSize}'");
                    break;

                case LayerType.Simple:
                    OutputLayerConfig = new SimpleLayerConfig();
                    Logger.WriteLine("Initialize configuration for simple layer.");
                    break;
            }
        }

        private void SetHiddenLayers()
        {
            //Get hidden layer settings
            //Example: LSTM:200, Dropout:0.5
            HiddenLayersConfig = new List<LayerConfig>();
            var hiddenLayers = config.GetValueRequired(HIDDEN_LAYER);
            foreach (var layer in hiddenLayers.Split(','))
            {
                var items = layer.Split(':');
                var sLayerType = items[0].Trim();
                var layerType = LayerType.None;
                foreach (
                    var type in
                        Enum.GetValues(typeof(LayerType))
                            .Cast<LayerType>()
                            .Where(
                                type => sLayerType.Equals(type.ToString(), StringComparison.InvariantCultureIgnoreCase))
                    )
                {
                    layerType = type;
                    break;
                }

                LayerConfig baseLayerConfig;
                switch (layerType)
                {
                    case LayerType.LSTM:
                        {
                            var layerConfig = new LSTMLayerConfig();
                            layerConfig.LayerSize = int.Parse(items[1]);
                            layerConfig.LayerType = layerType;
                            baseLayerConfig = layerConfig;
                            Logger.WriteLine($"Initialize configuration for LSTM layer. Layer size = {layerConfig.LayerSize}");
                        }
                        break;

                    case LayerType.DropOut:
                        {
                            var layerConfig = new DropoutLayerConfig { DropoutRatio = float.Parse(items[1])};
                            layerConfig.LayerType = layerType;
                            baseLayerConfig = layerConfig;
                            Logger.WriteLine(
                                $"Initialize configuration for Dropout layer. Dropout ratio = '{layerConfig.DropoutRatio}'");
                        }
                        break;

                    default:
                        throw new ArgumentException($"Invalidated layer type: {sLayerType}");
                }

                HiddenLayersConfig.Add(baseLayerConfig);
            }
            Logger.WriteLine($"Hidden layer : {HiddenLayersConfig.Count}");
            Logger.WriteLine($"Hidden layer : {hiddenLayers}");
        }

        private RNNDecoder InitializeAutoEncoder(string autoEncoderFeatureConfigFile)
        {
            Logger.WriteLine("Auto encoder configuration file: {0}", autoEncoderFeatureConfigFile);

            //Create feature extractors and load word embedding data from file
            Logger.WriteLine("Initializing feature set for auto-encoder model...");
            var featurizer = new Config(autoEncoderFeatureConfigFile, null);

            //Create instance for decoder
            Logger.WriteLine("Initializing auto-encoder model...");
            return new RNNDecoder(featurizer);
        }

        // truncate current to range [lower, upper)
        public int TruncPosition(int current, int lower, int upper)
        {
            return current < lower ? lower : (current >= upper ? upper - 1 : current);
        }

        private void ComputingFeatureSize()
        {
            var fc = featureContext;
            SparseFeatureSize = 0;
            if (tFeaturizer != null)
            {
                if (fc.ContainsKey(TFEATURE_CONTEXT))
                {
                    SparseFeatureSize += tFeaturizer.GetFeatureSize() * fc[TFEATURE_CONTEXT].Count;
                }
            }

            if (fc.ContainsKey(RT_FEATURE_CONTEXT))
            {
                SparseFeatureSize += TagSet.GetSize() * fc[RT_FEATURE_CONTEXT].Count;
            }
        }

        private bool IsRunTimeFeatureUsed()
        {
            var fc = featureContext;
            return fc.ContainsKey(RT_FEATURE_CONTEXT);
        }

        public void ShowFeatureSize()
        {
            var fc = featureContext;

            if (tFeaturizer != null)
                Logger.WriteLine("Template feature size: {0}", tFeaturizer.GetFeatureSize());

            if (fc.ContainsKey(TFEATURE_CONTEXT))
                Logger.WriteLine("Template feature context size: {0}",
                    tFeaturizer.GetFeatureSize() * fc[TFEATURE_CONTEXT].Count);

            if (fc.ContainsKey(RT_FEATURE_CONTEXT))
                Logger.WriteLine("Run time feature size: {0}", TagSet.GetSize() * fc[RT_FEATURE_CONTEXT].Count);

            if (fc.ContainsKey(WORDEMBEDDING_CONTEXT))
                Logger.WriteLine("Pretrained dense feature size: {0}",
                    preTrainedModel.GetDimension() * fc[WORDEMBEDDING_CONTEXT].Count);
        }

        private void ExtractSparseFeature(int currentState, int numStates, List<string[]> features, State pState)
        {
            var sparseFeature = new Dictionary<int, float>();
            var start = 0;
            var fc = featureContext;

            //Extract TFeatures in given context window
            if (tFeaturizer != null)
            {
                if (fc.ContainsKey(TFEATURE_CONTEXT))
                {
                    var v = fc[TFEATURE_CONTEXT];
                    for (var j = 0; j < v.Count; j++)
                    {
                        var offset = TruncPosition(currentState + v[j], 0, numStates);

                        var tfeatureList = tFeaturizer.GetFeatureIds(features, offset);
                        foreach (var featureId in tfeatureList)
                        {
                            if (tFeatureWeightType == TFEATURE_WEIGHT_TYPE_ENUM.BINARY)
                            {
                                sparseFeature[start + featureId] = 1;
                            }
                            else
                            {
                                if (sparseFeature.ContainsKey(start + featureId) == false)
                                {
                                    sparseFeature.Add(start + featureId, 1);
                                }
                                else
                                {
                                    sparseFeature[start + featureId]++;
                                }
                            }
                        }
                        start += tFeaturizer.GetFeatureSize();
                    }
                }
            }

            // Create place hold for run time feature
            // The real feature value is calculated at run time
            if (fc.ContainsKey(RT_FEATURE_CONTEXT))
            {
                var v = fc[RT_FEATURE_CONTEXT];
                pState.RuntimeFeatures = new PriviousLabelFeature[v.Count];
                for (var j = 0; j < v.Count; j++)
                {
                    if (v[j] < 0)
                    {
                        pState.AddRuntimeFeaturePlacehold(j, v[j], sparseFeature.Count, start);
                        sparseFeature[start] = 0; //Placehold a position
                        start += TagSet.GetSize();
                    }
                    else
                    {
                        throw new Exception("The offset of run time feature should be negative.");
                    }
                }
            }

            var spSparseFeature = pState.SparseFeature;
            spSparseFeature.SetLength(SparseFeatureSize);
            spSparseFeature.AddKeyValuePairData(sparseFeature);
        }

        //Extract word embedding features from current context
        public VectorBase ExtractDenseFeature(int currentState, int numStates, List<string[]> features)
        {
            var fc = featureContext;

            if (fc.ContainsKey(WORDEMBEDDING_CONTEXT))
            {
                var v = fc[WORDEMBEDDING_CONTEXT];
                if (v.Count == 1)
                {
                    var strKey = features[TruncPosition(currentState + v[0], 0, numStates)][preTrainedModelColumn];
                    return preTrainedModel.GetTermVector(strKey);
                }

                var dense = new CombinedVector();
                for (var j = 0; j < v.Count; j++)
                {
                    var offset = currentState + v[j];
                    if (offset >= 0 && offset < numStates)
                    {
                        var strKey = features[offset][preTrainedModelColumn];
                        dense.Append(preTrainedModel.GetTermVector(strKey));
                    }
                    else
                    {
                        dense.Append(preTrainedModel.m_UnkEmbedding);
                    }
                }

                return dense;
            }

            return new SingleVector();
        }

        public SequencePair ExtractFeatures(SentencePair sentence)
        {
            var sPair = new SequencePair
            {
                autoEncoder = Seq2SeqAutoEncoder,
                srcSentence = sentence.srcSentence,
                tgtSequence = BuildSequence(sentence.tgtSentence)
            };

            return sPair;
        }

        public State BuildState(string[] word)
        {
            var state = new State();
            var tokenList = new List<string[]> { word };

            ExtractSparseFeature(0, 1, tokenList, state);
            state.DenseFeature = ExtractDenseFeature(0, 1, tokenList);

            return state;
        }

        public Sequence BuildSequence(Sentence sentence)
        {
            var n = sentence.TokensList.Count;
            var sequence = new Sequence(n);

            //For each token, get its sparse and dense feature set according configuration and training corpus
            for (var i = 0; i < n; i++)
            {
                var state = sequence.States[i];
                ExtractSparseFeature(i, n, sentence.TokensList, state);
            }

            if (preTrainType == RNNSharp.PRETRAIN_TYPE.AutoEncoder)
            {
                var outputs = autoEncoder.ComputeTopHiddenLayerOutput(sentence);
                for (var i = 0; i < n; i++)
                {
                    var state = sequence.States[i];
                    state.DenseFeature = new SingleVector(outputs[i]);
                }
            }
            else
            {
                for (var i = 0; i < n; i++)
                {
                    var state = sequence.States[i];
                    state.DenseFeature = ExtractDenseFeature(i, n, sentence.TokensList);
                }
            }

            return sequence;
        }
    }
}