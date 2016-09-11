using System;
using System.Collections.Generic;
using System.IO;
using AdvUtils;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    enum TFEATURE_WEIGHT_TYPE_ENUM
    {
        BINARY,
        FREQUENCY
    }

    enum PRETRAIN_TYPE
    {
        AUTOENCODER,
        EMBEDDING
    }

    public class Featurizer
    {
        public TagSet TagSet { get; set; }

        Dictionary<string, List<int>> FeatureContext;
        int SparseDimension;
        int PretrainedModelColumn;
        TFEATURE_WEIGHT_TYPE_ENUM TFeatureWeightType = TFEATURE_WEIGHT_TYPE_ENUM.BINARY;
        WordEMWrapFeaturizer PretainedModel;
        TemplateFeaturizer TFeaturizer;

        static string TFEATURE_CONTEXT = "TFEATURE_CONTEXT";
        static string TFEATURE_FILENAME = "TFEATURE_FILENAME";
        static string RT_FEATURE_CONTEXT = "RTFEATURE_CONTEXT";
        static string TFEATURE_WEIGHT_TYPE = "TFEATURE_WEIGHT_TYPE";

        static string PRETRAIN_TYPE = "PRETRAIN_TYPE";
        static string WORDEMBEDDING_CONTEXT = "WORDEMBEDDING_CONTEXT";
        static string PRETRAINEDMODEL_FILENAME = "WORDEMBEDDING_FILENAME";
        static string PRETRAINEDMODEL_RAW_FILENAME = "WORDEMBEDDING_RAW_FILENAME";
        static string PRETRAINEDMODEL_COLUMN = "WORDEMBEDDING_COLUMN";
        static string AUTOENCODER_MODEL = "AUTOENCODER_MODEL";
        static string AUTOENCODER_FEATURECONFIG = "AUTOENCODER_FEATURECONFIG";

        //The format of configuration file
        public void LoadFeatureConfigFromFile(string strFileName)
        {
            StreamReader sr = new StreamReader(strFileName);
            string strLine = null;

            FeatureContext = new Dictionary<string, List<int>>();
            while ((strLine = sr.ReadLine()) != null)
            {
                strLine = strLine.Trim();
                if (strLine.Length == 0)
                {
                    //Emtpy line, ignore it
                    continue;
                }

                if (strLine.StartsWith("#") == true)
                {
                    //Comments line, ignore it
                    continue;
                }

                int idxSeparator = strLine.IndexOf(':');
                string strKey = strLine.Substring(0, idxSeparator).Trim();
                string strValue = strLine.Substring(idxSeparator + 1).Trim();
                if (strKey == PRETRAINEDMODEL_FILENAME)
                {
                    if (PretainedModel != null)
                    {
                        throw new ArgumentException("Static pretrained model has already been loaded. Please check if settings is duplicated in configuration file.");
                    }
                    Logger.WriteLine("Loading pretrained dense feature set from model {0}", strValue);
                    PretainedModel = new WordEMWrapFeaturizer(strValue);
                }
                else if (strKey == PRETRAINEDMODEL_RAW_FILENAME)
                {
                    if (PretainedModel != null)
                    {
                        throw new ArgumentException("Static pretrained model has already been loaded. Please check if settings is duplicated in configuration file.");
                    }
                    Logger.WriteLine("Loading pretrained dense feature set from model {0} in text format", strValue);
                    PretainedModel = new WordEMWrapFeaturizer(strValue, true);
                }
                else if (strKey == TFEATURE_FILENAME)
                {
                    Logger.WriteLine("Loading template feature set...");
                    TFeaturizer = new TemplateFeaturizer(strValue);
                }
                else if (strKey == PRETRAINEDMODEL_COLUMN)
                {
                    PretrainedModelColumn = int.Parse(strValue);
                    Logger.WriteLine("Pretrained model feature column: {0}", PretrainedModelColumn);
                }
                else if (strKey == TFEATURE_WEIGHT_TYPE)
                {
                    Logger.WriteLine("TFeature weighting type: {0}", strValue);
                    if (strValue == "binary")
                    {
                        TFeatureWeightType = TFEATURE_WEIGHT_TYPE_ENUM.BINARY;
                    }
                    else
                    {
                        TFeatureWeightType = TFEATURE_WEIGHT_TYPE_ENUM.FREQUENCY;
                    }
                }
                else if (strKey == PRETRAIN_TYPE)
                {
                    if (strValue.Equals(RNNSharp.PRETRAIN_TYPE.AUTOENCODER.ToString(), StringComparison.InvariantCultureIgnoreCase))
                    {
                        preTrainType = RNNSharp.PRETRAIN_TYPE.AUTOENCODER;
                    }
                    else
                    {
                        preTrainType = RNNSharp.PRETRAIN_TYPE.EMBEDDING;
                    }

                    Logger.WriteLine("Pretrain type: {0}", preTrainType);
                }
                else if (strKey == AUTOENCODER_FEATURECONFIG)
                {
                    autoEncoderFeatureConfigFile = strValue;
                    Logger.WriteLine("Auto encoder configuration file: {0}", autoEncoderFeatureConfigFile);
                }
                else if (strKey == AUTOENCODER_MODEL)
                {
                    autoEncoderModelFile = strValue;
                    Logger.WriteLine("Auto encoder model file: {0}", autoEncoderModelFile);
                }
                else
                {
                    string[] values = strValue.Split(',');

                    if (FeatureContext.ContainsKey(strKey) == false)
                    {
                        FeatureContext.Add(strKey, new List<int>());
                    }

                    foreach (string value in values)
                    {
                        FeatureContext[strKey].Add(int.Parse(value));
                    }
                }
            }

            sr.Close();
        }

        // truncate current to range [lower, upper)
        public int TruncPosition(int current, int lower, int upper)
        {
            return (current < lower) ? lower : ((current >= upper) ? upper - 1 : current);
        }

        public Featurizer(string strFeatureConfigFileName, TagSet tagSet)
        {
            LoadFeatureConfigFromFile(strFeatureConfigFileName);
            TagSet = tagSet;
            InitComponentFeaturizer();
        }

        void InitComponentFeaturizer()
        {
            var fc = FeatureContext;
            SparseDimension = 0;
            if (TFeaturizer != null)
            {
                if (fc.ContainsKey(TFEATURE_CONTEXT) == true)
                {
                    SparseDimension += TFeaturizer.GetFeatureSize() * fc[TFEATURE_CONTEXT].Count;
                }
            }

            if (fc.ContainsKey(RT_FEATURE_CONTEXT) == true)
            {
                SparseDimension += TagSet.GetSize() * fc[RT_FEATURE_CONTEXT].Count;
            }

            if (preTrainType == RNNSharp.PRETRAIN_TYPE.AUTOENCODER)
            {
                InitializeAutoEncoder();
            }
        }

        public bool IsRunTimeFeatureUsed()
        {
            var fc = FeatureContext;
            return fc.ContainsKey(RT_FEATURE_CONTEXT);
        }

        public void ShowFeatureSize()
        {
            var fc = FeatureContext;

            if (TFeaturizer != null)
                Logger.WriteLine("Template feature size: {0}", TFeaturizer.GetFeatureSize());

            if (fc.ContainsKey(TFEATURE_CONTEXT) == true)
                Logger.WriteLine("Template feature context size: {0}", TFeaturizer.GetFeatureSize() * fc[TFEATURE_CONTEXT].Count);

            if (fc.ContainsKey(RT_FEATURE_CONTEXT) == true)
                Logger.WriteLine("Run time feature size: {0}", TagSet.GetSize() * fc[RT_FEATURE_CONTEXT].Count);

            if (fc.ContainsKey(WORDEMBEDDING_CONTEXT) == true)
                Logger.WriteLine("Pretrained dense feature size: {0}", PretainedModel.GetDimension() * fc[WORDEMBEDDING_CONTEXT].Count);
        }

        void ExtractSparseFeature(int currentState, int numStates, List<string[]> features, State pState)
        {
            Dictionary<int, float> sparseFeature = new Dictionary<int, float>();
            int start = 0;
            var fc = FeatureContext;

            //Extract TFeatures in given context window
            if (TFeaturizer != null)
            {
                if (fc.ContainsKey(TFEATURE_CONTEXT) == true)
                {
                    List<int> v = fc[TFEATURE_CONTEXT];
                    for (int j = 0; j < v.Count; j++)
                    {
                        int offset = TruncPosition(currentState + v[j], 0, numStates);

                        List<int> tfeatureList = TFeaturizer.GetFeatureIds(features, offset);
                        foreach (int featureId in tfeatureList)
                        {
                            if (TFeatureWeightType == TFEATURE_WEIGHT_TYPE_ENUM.BINARY)
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
                        start += TFeaturizer.GetFeatureSize();
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

            SparseVector spSparseFeature = pState.SparseData;
            spSparseFeature.SetDimension(SparseDimension);
            spSparseFeature.SetData(sparseFeature);
        }

        //Extract word embedding features from current context
        public VectorBase ExtractDenseFeature(int currentState, int numStates, List<string[]> features)
        {
            var fc = FeatureContext;

            if (fc.ContainsKey(WORDEMBEDDING_CONTEXT) == true)
            {
                List<int> v = fc[WORDEMBEDDING_CONTEXT];
                if (v.Count == 1)
                {
                    string strKey = features[TruncPosition((int)currentState + v[0], 0, (int)numStates)][PretrainedModelColumn];
                    return PretainedModel.GetTermVector(strKey);
                }

                CombinedVector dense = new CombinedVector();
                for (int j = 0;j < v.Count;j++)
                {
                    int offset = currentState + v[j];
                    if (offset >= 0 && offset < numStates)
                    {
                        string strKey = features[offset][PretrainedModelColumn];
                        dense.Append(PretainedModel.GetTermVector(strKey));
                    }
                    else
                    {
                        dense.Append(PretainedModel.m_UnkEmbedding);
                    }
                }


                return dense;
            }

            return new SingleVector();
        }


        PRETRAIN_TYPE preTrainType = RNNSharp.PRETRAIN_TYPE.EMBEDDING;
        string autoEncoderModelFile = String.Empty;
        string autoEncoderFeatureConfigFile = String.Empty;
        RNNSharp.RNNDecoder autoEncoder = null;
        public void InitializeAutoEncoder()
        {
            Logger.WriteLine("Initialize auto encoder...");

            //Create feature extractors and load word embedding data from file
            Featurizer featurizer = new Featurizer(autoEncoderFeatureConfigFile, null);

            //Create instance for decoder
            autoEncoder = new RNNSharp.RNNDecoder(autoEncoderModelFile, featurizer);
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

            if (preTrainType == RNNSharp.PRETRAIN_TYPE.AUTOENCODER)
            {
                List<double[]> outputs = autoEncoder.ComputeTopHiddenLayerOutput(sentence);
                for (int i = 0; i < n; i++)
                {
                    State state = sequence.States[i];
                    state.DenseData = new SingleVector(outputs[i].Length, outputs[i]);
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    State state = sequence.States[i];
                    state.DenseData = ExtractDenseFeature(i, n, sentence.TokensList);
                }
            }

            return sequence;
        }
    }
}
