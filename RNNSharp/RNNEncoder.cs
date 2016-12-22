using AdvUtils;
using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class RNNEncoder<T> where T: ISequence
    {
        ModelSetting ModelSettings;
        Config featurizer;
        MODELDIRECTION modelDirection { get { return featurizer.ModelDirection; } }
        string modelFilePath { get { return featurizer.ModelFilePath; } }
        List<LayerConfig> hiddenLayersConfig { get { return featurizer.HiddenLayersConfig; } }
        LayerConfig outputLayerConfig { get { return featurizer.OutputLayerConfig; } }

        public bool IsCRFTraining { get { return featurizer.IsCRFTraining; } }
        public MODELTYPE ModelType { get { return featurizer.ModelType; } }

        public DataSet<T> TrainingSet { get; set; }
        public DataSet<T> ValidationSet { get; set; }

        public RNNEncoder(ModelSetting modelSetting, Config featurizer)
        {
            ModelSettings = modelSetting;
            this.featurizer = featurizer;
        }

        private int GetCurrentLayerDenseFeatureSize(int inputDenseFeatureSize)
        {
            //If current training is for sequence-to-sequence, we append features from source sequence to standard features.
            if (ModelType == MODELTYPE.Seq2Seq)
            {
                //[Dense feature set of each state in target sequence][Dense feature set of entire source sequence]
                inputDenseFeatureSize += featurizer.Seq2SeqAutoEncoder.GetTopHiddenLayerSize() * 2;
            }

            return inputDenseFeatureSize;
        }

        public void Train()
        {
            RNN<T> rnn;

            if (modelDirection == MODELDIRECTION.Forward)
            {
                int sparseFeatureSize = TrainingSet.SparseFeatureSize;
                if (ModelType == MODELTYPE.Seq2Seq)
                {
                    //[Sparse feature set of each state in target sequence][Sparse feature set of entire source sequence]
                    sparseFeatureSize += featurizer.Seq2SeqAutoEncoder.Featurizer.SparseFeatureSize;
                    Logger.WriteLine("Sparse Feature Format: [{0}][{1}] = {2}",
                        TrainingSet.SparseFeatureSize, featurizer.Seq2SeqAutoEncoder.Featurizer.SparseFeatureSize, sparseFeatureSize);
                }


                List<SimpleLayer> hiddenLayers = new List<SimpleLayer>();
                for (int i = 0; i < hiddenLayersConfig.Count; i++)
                {
                    SimpleLayer layer = null;
                    if (hiddenLayersConfig[i].LayerType == LayerType.BPTT)
                    {
                        BPTTLayer bpttLayer = new BPTTLayer(hiddenLayersConfig[i] as BPTTLayerConfig);
                        layer = bpttLayer;
                        Logger.WriteLine($"Create BPTT layer.");
                    }
                    else if (hiddenLayersConfig[i].LayerType == LayerType.LSTM)
                    {
                        LSTMLayer lstmLayer = new LSTMLayer(hiddenLayersConfig[i] as LSTMLayerConfig);
                        layer = lstmLayer;
                        Logger.WriteLine($"Create LSTM layer.");
                    }
                    else if (hiddenLayersConfig[i].LayerType == LayerType.DropOut)
                    {
                        DropoutLayer dropoutLayer = new DropoutLayer(hiddenLayersConfig[i] as DropoutLayerConfig);
                        layer = dropoutLayer;
                        Logger.WriteLine($"Create Dropout layer.");
                    }

                    if (i == 0)
                    {
                        layer.InitializeWeights(sparseFeatureSize, GetCurrentLayerDenseFeatureSize(TrainingSet.DenseFeatureSize));
                    }
                    else
                    {
                        layer.InitializeWeights(sparseFeatureSize, GetCurrentLayerDenseFeatureSize(hiddenLayers[i - 1].LayerSize));
                    }

                    Logger.WriteLine($"Create hidden layer {i}: size = {layer.LayerSize}, sparse feature size = {layer.SparseFeatureSize}, dense feature size = {layer.DenseFeatureSize}");
                    hiddenLayers.Add(layer);
                }


                SimpleLayer outputLayer = null;
                outputLayerConfig.LayerSize = TrainingSet.TagSize;

                if (outputLayerConfig.LayerType == LayerType.NCESoftmax)
                {
                    Logger.WriteLine("Create NCESoftmax layer as output layer");
                    NCEOutputLayer nceOutputLayer = new NCEOutputLayer(outputLayerConfig as NCELayerConfig);
                    nceOutputLayer.BuildStatisticData<T>(TrainingSet);
                    nceOutputLayer.InitializeWeights(0, GetCurrentLayerDenseFeatureSize(hiddenLayers[hiddenLayers.Count - 1].LayerSize));
                    outputLayer = nceOutputLayer;
                }
                else if (outputLayerConfig.LayerType == LayerType.Softmax)
                {
                    Logger.WriteLine("Create Softmax layer as output layer.");
                    outputLayer = new SimpleLayer(outputLayerConfig);
                    outputLayer.InitializeWeights(sparseFeatureSize, GetCurrentLayerDenseFeatureSize(hiddenLayers[hiddenLayers.Count - 1].LayerSize));
                }


                rnn = new ForwardRNN<T>(hiddenLayers, outputLayer);
            }
            else
            {
                List<SimpleLayer> forwardHiddenLayers = new List<SimpleLayer>();
                List<SimpleLayer> backwardHiddenLayers = new List<SimpleLayer>();
                for (int i = 0; i < hiddenLayersConfig.Count; i++)
                {
                    SimpleLayer forwardLayer = null;
                    SimpleLayer backwardLayer = null;
                    if (hiddenLayersConfig[i].LayerType == LayerType.BPTT)
                    {
                        //For BPTT layer
                        BPTTLayer forwardBPTTLayer = new BPTTLayer(hiddenLayersConfig[i] as BPTTLayerConfig);
                        forwardLayer = forwardBPTTLayer;

                        BPTTLayer backwardBPTTLayer = new BPTTLayer(hiddenLayersConfig[i] as BPTTLayerConfig);
                        backwardLayer = backwardBPTTLayer;

                        Logger.WriteLine($"Create BPTT layer.");
                    }
                    else if (hiddenLayersConfig[i].LayerType == LayerType.LSTM)
                    {
                        //For LSTM layer
                        LSTMLayer forwardLSTMLayer = new LSTMLayer(hiddenLayersConfig[i] as LSTMLayerConfig);
                        forwardLayer = forwardLSTMLayer;

                        LSTMLayer backwardLSTMLayer = new LSTMLayer(hiddenLayersConfig[i] as LSTMLayerConfig);
                        backwardLayer = backwardLSTMLayer;

                        Logger.WriteLine($"Create LSTM layer.");
                    }
                    else if (hiddenLayersConfig[i].LayerType == LayerType.DropOut)
                    {
                        DropoutLayer forwardDropoutLayer = new DropoutLayer(hiddenLayersConfig[i] as DropoutLayerConfig);
                        DropoutLayer backwardDropoutLayer = new DropoutLayer(hiddenLayersConfig[i] as DropoutLayerConfig);

                        Logger.WriteLine($"Create Dropout layer.");
                    }

                    if (i == 0)
                    {
                        forwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, TrainingSet.DenseFeatureSize);
                        backwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, TrainingSet.DenseFeatureSize);
                    }
                    else
                    {
                        forwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, forwardHiddenLayers[i - 1].LayerSize);
                        backwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, backwardHiddenLayers[i - 1].LayerSize);
                    }

                    Logger.WriteLine($"Create hidden layer {i}: size = {forwardLayer.LayerSize}, sparse feature size = {forwardLayer.SparseFeatureSize}, dense feature size = {forwardLayer.DenseFeatureSize}");

                    forwardHiddenLayers.Add(forwardLayer);
                    backwardHiddenLayers.Add(backwardLayer);
                }

                SimpleLayer outputLayer = null;
                outputLayerConfig.LayerSize = TrainingSet.TagSize;
                if (outputLayerConfig.LayerType == LayerType.NCESoftmax)
                {
                    Logger.WriteLine("Create NCESoftmax layer as output layer.");
                    NCEOutputLayer nceOutputLayer = new NCEOutputLayer(outputLayerConfig as NCELayerConfig);
                    nceOutputLayer.BuildStatisticData<T>(TrainingSet);
                    nceOutputLayer.InitializeWeights(0, forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize);
                    outputLayer = nceOutputLayer;
                }
                else if (outputLayerConfig.LayerType == LayerType.Softmax)
                {
                    Logger.WriteLine("Create Softmax layer as output layer.");
                    outputLayer = new SimpleLayer(outputLayerConfig);
                    outputLayer.InitializeWeights(TrainingSet.SparseFeatureSize, forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize);
                }

                rnn = new BiRNN<T>(forwardHiddenLayers, backwardHiddenLayers, outputLayer);
            }

            rnn.bVQ = (ModelSettings.VQ != 0) ? true : false;
            rnn.SaveStep = ModelSettings.SaveStep;
            rnn.MaxIter = ModelSettings.MaxIteration;
            rnn.IsCRFTraining = IsCRFTraining;
            rnn.ModelType = ModelType;


            RNNHelper.LearningRate = ModelSettings.LearningRate;
            RNNHelper.GradientCutoff = ModelSettings.GradientCutoff;
            RNNHelper.IsConstAlpha = ModelSettings.IsConstAlpha;
            
            //Create tag-bigram transition probability matrix only for sequence RNN mode
            if (IsCRFTraining)
            {
                Logger.WriteLine("Initialize bigram transition for CRF output layer.");
                rnn.setTagBigramTransition(TrainingSet.CRFLabelBigramTransition);
            }

            Logger.WriteLine("");

            Logger.WriteLine("Iterative training begins ...");
            double lastPPL = double.MaxValue;
            float lastAlpha = RNNHelper.LearningRate;
            int iter = 0;
            while (true)
            {
                Logger.WriteLine("Cleaning training status...");
                rnn.CleanStatus();

                if (rnn.MaxIter > 0 && iter > rnn.MaxIter)
                {
                    Logger.WriteLine("We have trained this model {0} iteration, exit.");
                    break;
                }

                //Start to train model
                double ppl = rnn.TrainNet(TrainingSet, iter);
                if (ppl >= lastPPL && lastAlpha != RNNHelper.LearningRate)
                {
                    //Although we reduce alpha value, we still cannot get better result.
                    Logger.WriteLine("Current perplexity({0}) is larger than the previous one({1}). End training early.", ppl, lastPPL);
                    Logger.WriteLine("Current alpha: {0}, the previous alpha: {1}", RNNHelper.LearningRate, lastAlpha);
                    break;
                }
                lastAlpha = RNNHelper.LearningRate;

                //Validate the model by validated corpus
                if (ValidationSet != null)
                {
                    Logger.WriteLine("Verify model on validated corpus.");
                    if (rnn.ValidateNet(ValidationSet, iter) == true)
                    {
                        //We got better result on validated corpus, save this model
                        Logger.WriteLine("Saving better model into file {0}...", modelFilePath);
                        rnn.SaveModel(modelFilePath);
                    }
                }
                else if (ppl < lastPPL)
                {
                    //We don't have validate corpus, but we get a better result on training corpus
                    //We got better result on validated corpus, save this model
                    Logger.WriteLine("Saving better model into file {0}...", modelFilePath);
                    rnn.SaveModel(modelFilePath);
                }
                
                if (ppl >= lastPPL)
                {
                    //We cannot get a better result on training corpus, so reduce learning rate
                    RNNHelper.LearningRate = RNNHelper.LearningRate / 2.0f;
                }

                lastPPL = ppl;

                iter++;
            }
        }
    }
}
