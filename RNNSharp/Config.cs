using System;
using System.Collections.Generic;
using System.IO;
using AdvUtils;

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

    public class BPTTLayerConfig : LayerConfig
    {
        public int Bptt;

        public BPTTLayerConfig()
        {
            LayerType = LayerType.BPTT;
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

    public class NCELayerConfig : LayerConfig
    {
        public int NegativeSampleSize;

        public NCELayerConfig()
        {
            LayerType = LayerType.NCESoftmax;
        }
    }

    public class SoftmaxLayerConfig : LayerConfig
    {
        public SoftmaxLayerConfig()
        {
            LayerType = LayerType.Softmax;
        }
    }

    public class Config
    {
        public TagSet TagSet { get; set; }
        public int SparseFeatureSize;

        //Feature context offset for sparse and dense feature set
        Dictionary<string, List<int>> featureContext;

        //Settings for model type
        static string MODEL_TYPE = "MODEL_TYPE";
        public MODELTYPE ModelType { get; set; }

        //Settings for current directory
        static string CURRENT_DIRECTORY = "CURRENT_DIRECTORY";
        string currentDirectory { get; set; }

        //Settings for hidden layers
        //Format: [Type:Size], example:
        //HIDDEN_LAYER = 300:LSTM,200:BPTT
        static string HIDDEN_LAYER = "HIDDEN_LAYER";
        public List<LayerConfig> HiddenLayersConfig { get; set; }

        //Settings for output layer
        //Example: OUTPUT_LAYER = Softmax
        static string OUTPUT_LAYER = "OUTPUT_LAYER";
        public LayerConfig OutputLayerConfig;

        static string CRF_LAYER = "CRF_LAYER";
        public bool IsCRFTraining;

        //Settings for model file path
        static string MODEL_FILEPATH = "MODEL_FILEPATH";
        public string ModelFilePath { get; set; }

        //Settings for sparse feature
        static string TFEATURE_CONTEXT = "TFEATURE_CONTEXT";
        static string TFEATURE_FILENAME = "TFEATURE_FILENAME";
        TemplateFeaturizer tFeaturizer;

        static string RT_FEATURE_CONTEXT = "RTFEATURE_CONTEXT";

        //Settings for sparse feature weight type
        static string TFEATURE_WEIGHT_TYPE = "TFEATURE_WEIGHT_TYPE";
        TFEATURE_WEIGHT_TYPE_ENUM tFeatureWeightType;

        //Settings for pretrained model type
        static string PRETRAIN_TYPE = "PRETRAIN_TYPE";
        PRETRAIN_TYPE preTrainType;

        //Settings for word embedding model
        static string WORDEMBEDDING_CONTEXT = "WORDEMBEDDING_CONTEXT";
        static string PRETRAINEDMODEL_FILENAME = "WORDEMBEDDING_FILENAME";
        static string PRETRAINEDMODEL_RAW_FILENAME = "WORDEMBEDDING_RAW_FILENAME";
        static string PRETRAINEDMODEL_COLUMN = "WORDEMBEDDING_COLUMN";
        int preTrainedModelColumn;
        WordEMWrapFeaturizer preTrainedModel;

        //Settings for auto encoder model
        static string AUTOENCODER_CONFIG = "AUTOENCODER_CONFIG";
        RNNDecoder autoEncoder;

        //Settings for auto encoder model in sequence-to-sequence task
        static string SEQ2SEQ_AUTOENCODER_CONFIG = "SEQ2SEQ_AUTOENCODER_CONFIG";
        public RNNDecoder Seq2SeqAutoEncoder;

        //Settings for model direction type
        static string MODEL_DIRECTION = "MODEL_DIRECTION";
        public MODELDIRECTION ModelDirection { get; set; }

        //Raw configuration
        static ConfigUtils config;


        private string GetFilePath(string currentDirectory, string filePath)
        {
            if (Path.IsPathRooted(filePath))
            {
                return filePath;
            }

            return Path.Combine(currentDirectory, filePath);
        }

        //The format of configuration file
        public void LoadFeatureConfigFromFile(string configFilePath)
        {
            //Load configuration file
            config = new ConfigUtils();
            config.LoadFile(configFilePath);

            //Get current directory from configuration file
            currentDirectory = config.GetValueOptional(CURRENT_DIRECTORY);
            if (String.IsNullOrEmpty(currentDirectory))
            {
                currentDirectory = Environment.CurrentDirectory;
            }
            Logger.WriteLine($"Current directory : {currentDirectory}");

            //Get model file path
            ModelFilePath = GetFilePath(currentDirectory, config.GetValueRequired(MODEL_FILEPATH));
            Logger.WriteLine($"Main model is located at {ModelFilePath}");

            featureContext = new Dictionary<string, List<int>>();

            SetHiddenLayers();
            SetOutputLayers();
            SetPretrainedModel();
            SetTFeatures();

            string isCRFTraining = config.GetValueOptional(CRF_LAYER);
            IsCRFTraining = false;
            if (String.IsNullOrEmpty(isCRFTraining) == false)
            {
                IsCRFTraining = bool.Parse(isCRFTraining);
            }

            //Load model type
            if (config.GetValueRequired(MODEL_TYPE).Equals(MODELTYPE.SeqLabel.ToString(), StringComparison.InvariantCultureIgnoreCase))
            {
                ModelType = MODELTYPE.SeqLabel;
            }
            else
            {
                ModelType = MODELTYPE.Seq2Seq;
            }
            Logger.WriteLine($"Model type: {ModelType}");

            if (config.GetValueRequired(MODEL_DIRECTION).Equals(MODELDIRECTION.Forward.ToString(), StringComparison.InvariantCultureIgnoreCase))
            {
                ModelDirection = MODELDIRECTION.Forward;
            }
            else
            {
                ModelDirection = MODELDIRECTION.BiDirectional;
            }
            Logger.WriteLine($"Model direction: {ModelDirection}");

            //Load auto-encoder model for sequence-to-sequence. This model is used to encode source sequence
            if (ModelType == MODELTYPE.Seq2Seq)
            {
                string seqAutoEncoderConfigFilePath = GetFilePath(currentDirectory, config.GetValueRequired(SEQ2SEQ_AUTOENCODER_CONFIG));
                Logger.WriteLine($"Loading auto encoder model for sequnce-to-sequence task. Config file = '{seqAutoEncoderConfigFilePath}'");

                Seq2SeqAutoEncoder = InitializeAutoEncoder(seqAutoEncoderConfigFilePath);
            }

            //Check if settings are validated
            CheckSettings();
        }

        private void CheckSettings()
        {
            if (ModelDirection == MODELDIRECTION.BiDirectional && ModelType == MODELTYPE.Seq2Seq)
            {
                throw new System.Exception("Bi-directional RNN model doesn't support sequence-to-sequence model.");
            }

            if (IsRunTimeFeatureUsed() == true && ModelDirection == MODELDIRECTION.BiDirectional)
            {
                throw new System.Exception("Run time feature is not available for bi-directional model.");
            }
        }

        private void SetTFeatures()
        {
            //Load template feature set
            string tfeatureFilePath = GetFilePath(currentDirectory, config.GetValueRequired(TFEATURE_FILENAME));
            Logger.WriteLine($"Loading template feature set from {tfeatureFilePath}");
            tFeaturizer = new TemplateFeaturizer(tfeatureFilePath);

            string tfeatureWeightType = config.GetValueRequired(TFEATURE_WEIGHT_TYPE);
            if (tfeatureWeightType.Equals("binary", StringComparison.InvariantCultureIgnoreCase))
            {
                tFeatureWeightType = TFEATURE_WEIGHT_TYPE_ENUM.BINARY;
            }
            else
            {
                tFeatureWeightType = TFEATURE_WEIGHT_TYPE_ENUM.FREQUENCY;
            }
            Logger.WriteLine($"TFeature weight type: {tfeatureWeightType}");

            string tfeatureContext = config.GetValueRequired(TFEATURE_CONTEXT);
            featureContext.Add(TFEATURE_CONTEXT, new List<int>());
            foreach (string contextOffset in tfeatureContext.Split(','))
            {
                featureContext[TFEATURE_CONTEXT].Add(int.Parse(contextOffset));
            }
            Logger.WriteLine($"TFeature context: {tfeatureContext}");
        }

        private void SetPretrainedModel()
        {
            //Load pre-trained model. It supports embedding model and auto-encoder model
            string preTrainTypeValue = config.GetValueRequired(PRETRAIN_TYPE);
            Logger.WriteLine("Pretrain type: {0}", preTrainTypeValue);

            if (preTrainTypeValue.Equals(RNNSharp.PRETRAIN_TYPE.AutoEncoder.ToString(), StringComparison.InvariantCultureIgnoreCase))
            {
                preTrainType = RNNSharp.PRETRAIN_TYPE.AutoEncoder;
                string autoEncoderConfigFilePath = GetFilePath(currentDirectory, config.GetValueRequired(AUTOENCODER_CONFIG));
                Logger.WriteLine($"Loading auto encoder model. Config file = '{autoEncoderConfigFilePath}'");
                autoEncoder = InitializeAutoEncoder(autoEncoderConfigFilePath);
            }
            else
            {
                preTrainType = RNNSharp.PRETRAIN_TYPE.Embedding;
                string preTrainedModelFilePath = config.GetValueOptional(PRETRAINEDMODEL_FILENAME);
                if (String.IsNullOrEmpty(preTrainedModelFilePath) == false)
                {
                    preTrainedModelFilePath = GetFilePath(currentDirectory, preTrainedModelFilePath);
                    if (preTrainedModel != null)
                    {
                        throw new ArgumentException("Static pretrained model has already been loaded. Please check if settings is duplicated in configuration file.");
                    }
                    Logger.WriteLine($"Loading pretrained embedding model: {preTrainedModelFilePath}");
                    preTrainedModel = new WordEMWrapFeaturizer(preTrainedModelFilePath);
                }

                string preTrainedRawModelFilePath = config.GetValueOptional(PRETRAINEDMODEL_RAW_FILENAME);
                if (String.IsNullOrEmpty(preTrainedRawModelFilePath) == false)
                {
                    preTrainedRawModelFilePath = GetFilePath(currentDirectory, preTrainedRawModelFilePath);
                    if (preTrainedModel != null)
                    {
                        throw new ArgumentException("Static pretrained model has already been loaded. Please check if settings is duplicated in configuration file.");
                    }
                    Logger.WriteLine($"Loading pretrained embedding model {preTrainedRawModelFilePath} in text format");
                    preTrainedModel = new WordEMWrapFeaturizer(preTrainedRawModelFilePath, true);
                }

                preTrainedModelColumn = int.Parse(config.GetValueRequired(PRETRAINEDMODEL_COLUMN));
                Logger.WriteLine("Pretrained model feature column: {0}", preTrainedModelColumn);

                string preTrainedModelContext = config.GetValueRequired(WORDEMBEDDING_CONTEXT);
                featureContext.Add(WORDEMBEDDING_CONTEXT, new List<int>());
                foreach (string contextOffset in preTrainedModelContext.Split(','))
                {
                    featureContext[WORDEMBEDDING_CONTEXT].Add(int.Parse(contextOffset));
                }
                Logger.WriteLine($"Pretrained model context offset : {preTrainedModelContext}");
            }
        }

        private void SetOutputLayers()
        {
            //Settings for output layer
            string outputLayer = config.GetValueRequired(OUTPUT_LAYER);
            string[] items = outputLayer.Split(':');
            string sLayerType = items[0];
            LayerType outputLayerType = LayerType.None;
            foreach (LayerType type in Enum.GetValues(typeof(LayerType)))
            {
                if (sLayerType.Equals(type.ToString(), StringComparison.InvariantCultureIgnoreCase))
                {
                    outputLayerType = type;
                    break;
                }
            }

            if (outputLayerType == LayerType.Softmax)
            {
                SoftmaxLayerConfig softmaxLayerConfig = new SoftmaxLayerConfig();
                OutputLayerConfig = softmaxLayerConfig;

                Logger.WriteLine($"Initialize configuration for softmax layer.");
            }
            else if (outputLayerType == LayerType.NCESoftmax)
            {
                NCELayerConfig nceLayerConfig = new NCELayerConfig();
                nceLayerConfig.NegativeSampleSize = int.Parse(items[1]);
                OutputLayerConfig = nceLayerConfig;

                Logger.WriteLine($"Initialize configuration for NCESoftmax layer. Negative sample size = '{nceLayerConfig.NegativeSampleSize}'");
            }
        }

        private void SetHiddenLayers()
        {
            //Get hidden layer settings
            HiddenLayersConfig = new List<LayerConfig>();
            string hiddenLayers = config.GetValueRequired(HIDDEN_LAYER);
            foreach (string layer in hiddenLayers.Split(','))
            {
                string[] items = layer.Split(':');
                string sLayerSize = items[0].Trim();
                string sLayerType = items[1].Trim();

                //Parse layer size and type
                int layerSize = int.Parse(sLayerSize);
                LayerType layerType = LayerType.None;

                foreach (LayerType type in Enum.GetValues(typeof(LayerType)))
                {
                    if (sLayerType.Equals(type.ToString(), StringComparison.InvariantCultureIgnoreCase))
                    {
                        layerType = type;
                        break;
                    }
                }

                LayerConfig baseLayerConfig = null;
                if (layerType == LayerType.LSTM)
                {
                    LSTMLayerConfig layerConfig = new LSTMLayerConfig();
                    baseLayerConfig = layerConfig;
                    Logger.WriteLine("Initialize configuration for LSTM layer.");
                }
                else if (layerType == LayerType.BPTT)
                {
                    if (items.Length != 3)
                    {
                        throw new ArgumentException($"Invalidated settings for BPTT: {layer}, it should be [size:BPTT:bptt_value], such as [200:BPTT:5] .");
                    }

                    BPTTLayerConfig layerConfig = new BPTTLayerConfig();
                    layerConfig.Bptt = int.Parse(items[2]);
                    baseLayerConfig = layerConfig;
                    Logger.WriteLine($"Initialize configuration for BPTT layer. BPTT = '{layerConfig.Bptt}'");
                }
                else if (layerType == LayerType.DropOut)
                {
                    DropoutLayerConfig layerConfig = new DropoutLayerConfig();
                    layerConfig.DropoutRatio = float.Parse(items[2]);
                    baseLayerConfig = layerConfig;
                    Logger.WriteLine($"Initialize configuration for Dropout layer. Dropout ratio = '{layerConfig.DropoutRatio}'");
                }
                else
                {
                    throw new ArgumentException($"Invalidated layer type: {sLayerType}");
                }


                baseLayerConfig.LayerType = layerType;
                baseLayerConfig.LayerSize = layerSize;

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
            Config featurizer = new Config(autoEncoderFeatureConfigFile, null);

            //Create instance for decoder
            Logger.WriteLine("Initializing auto-encoder model...");
            return new RNNDecoder(featurizer);

        }


        // truncate current to range [lower, upper)
        public int TruncPosition(int current, int lower, int upper)
        {
            return (current < lower) ? lower : ((current >= upper) ? upper - 1 : current);
        }

        public Config(string strFeatureConfigFileName, TagSet tagSet)
        {
            LoadFeatureConfigFromFile(strFeatureConfigFileName);
            TagSet = tagSet;
            ComputingFeatureSize();
        }

        void ComputingFeatureSize()
        {
            var fc = featureContext;
            SparseFeatureSize = 0;
            if (tFeaturizer != null)
            {
                if (fc.ContainsKey(TFEATURE_CONTEXT) == true)
                {
                    SparseFeatureSize += tFeaturizer.GetFeatureSize() * fc[TFEATURE_CONTEXT].Count;
                }
            }

            if (fc.ContainsKey(RT_FEATURE_CONTEXT) == true)
            {
                SparseFeatureSize += TagSet.GetSize() * fc[RT_FEATURE_CONTEXT].Count;
            }

        }

        bool IsRunTimeFeatureUsed()
        {
            var fc = featureContext;
            return fc.ContainsKey(RT_FEATURE_CONTEXT);
        }

        public void ShowFeatureSize()
        {
            var fc = featureContext;

            if (tFeaturizer != null)
                Logger.WriteLine("Template feature size: {0}", tFeaturizer.GetFeatureSize());

            if (fc.ContainsKey(TFEATURE_CONTEXT) == true)
                Logger.WriteLine("Template feature context size: {0}", tFeaturizer.GetFeatureSize() * fc[TFEATURE_CONTEXT].Count);

            if (fc.ContainsKey(RT_FEATURE_CONTEXT) == true)
                Logger.WriteLine("Run time feature size: {0}", TagSet.GetSize() * fc[RT_FEATURE_CONTEXT].Count);

            if (fc.ContainsKey(WORDEMBEDDING_CONTEXT) == true)
                Logger.WriteLine("Pretrained dense feature size: {0}", preTrainedModel.GetDimension() * fc[WORDEMBEDDING_CONTEXT].Count);
        }

        void ExtractSparseFeature(int currentState, int numStates, List<string[]> features, State pState)
        {
            Dictionary<int, float> sparseFeature = new Dictionary<int, float>();
            int start = 0;
            var fc = featureContext;

            //Extract TFeatures in given context window
            if (tFeaturizer != null)
            {
                if (fc.ContainsKey(TFEATURE_CONTEXT) == true)
                {
                    List<int> v = fc[TFEATURE_CONTEXT];
                    for (int j = 0; j < v.Count; j++)
                    {
                        int offset = TruncPosition(currentState + v[j], 0, numStates);

                        List<int> tfeatureList = tFeaturizer.GetFeatureIds(features, offset);
                        foreach (int featureId in tfeatureList)
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
            if (fc.ContainsKey(RT_FEATURE_CONTEXT) == true)
            {
                List<int> v = fc[RT_FEATURE_CONTEXT];
                pState.RuntimeFeatures = new PriviousLabelFeature[v.Count];
                for (int j = 0; j < v.Count; j++)
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

            SparseVector spSparseFeature = pState.SparseFeature;
            spSparseFeature.SetLength(SparseFeatureSize);
            spSparseFeature.AddKeyValuePairData(sparseFeature);
        }

        //Extract word embedding features from current context
        public VectorBase ExtractDenseFeature(int currentState, int numStates, List<string[]> features)
        {
            var fc = featureContext;

            if (fc.ContainsKey(WORDEMBEDDING_CONTEXT) == true)
            {
                List<int> v = fc[WORDEMBEDDING_CONTEXT];
                if (v.Count == 1)
                {
                    string strKey = features[TruncPosition((int)currentState + v[0], 0, (int)numStates)][preTrainedModelColumn];
                    return preTrainedModel.GetTermVector(strKey);
                }

                CombinedVector dense = new CombinedVector();
                for (int j = 0;j < v.Count;j++)
                {
                    int offset = currentState + v[j];
                    if (offset >= 0 && offset < numStates)
                    {
                        string strKey = features[offset][preTrainedModelColumn];
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
            SequencePair sPair = new SequencePair();
            sPair.autoEncoder = Seq2SeqAutoEncoder;
            sPair.srcSentence = sentence.srcSentence;
            sPair.tgtSequence = ExtractFeatures(sentence.tgtSentence);

            return sPair;
        }

        public State ExtractFeatures(string[] word)
        {
            State state = new State();
            List<string[]> tokenList = new List<string[]>();
            tokenList.Add(word);

            ExtractSparseFeature(0, 1, tokenList, state);
            state.DenseFeature = ExtractDenseFeature(0, 1, tokenList);

            return state;
        }


        public Sequence ExtractFeatures(Sentence sentence)
        {
            int n = sentence.TokensList.Count;
            Sequence sequence = new Sequence(n);

            //For each token, get its sparse and dense feature set according configuration and training corpus
            for (int i = 0; i < n; i++)
            {
                State state = sequence.States[i];
                ExtractSparseFeature(i, n, sentence.TokensList, state);
            }

            if (preTrainType == RNNSharp.PRETRAIN_TYPE.AutoEncoder)
            {
                List<float[]> outputs = autoEncoder.ComputeTopHiddenLayerOutput(sentence);
                for (int i = 0; i < n; i++)
                {
                    State state = sequence.States[i];
                    state.DenseFeature = new SingleVector(outputs[i]);
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    State state = sequence.States[i];
                    state.DenseFeature = ExtractDenseFeature(i, n, sentence.TokensList);
                }
            }

            return sequence;
        }
    }
}
