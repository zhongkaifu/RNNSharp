using AdvUtils;
using System.Collections.Generic;
using System.Numerics;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp
{
    public class RNNEncoder<T> where T : ISequence
    {
        private readonly Config featurizer;
        private readonly ModelSetting ModelSettings;

        public RNNEncoder(ModelSetting modelSetting, Config featurizer)
        {
            ModelSettings = modelSetting;
            this.featurizer = featurizer;
        }

        private MODELDIRECTION modelDirection => featurizer.ModelDirection;

        private string modelFilePath => featurizer.ModelFilePath;

        private List<LayerConfig> hiddenLayersConfig => featurizer.HiddenLayersConfig;

        private LayerConfig outputLayerConfig => featurizer.OutputLayerConfig;

        public bool IsCRFTraining => featurizer.IsCRFTraining;

        public MODELTYPE ModelType => featurizer.ModelType;

        public DataSet<T> TrainingSet { get; set; }
        public DataSet<T> ValidationSet { get; set; }

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
            //Create neural net work
            Logger.WriteLine("Create a new network according settings in configuration file.");
            var rnn = CreateNetwork();

            //Assign model settings to RNN
            rnn.bVQ = ModelSettings.VQ != 0 ? true : false;
            rnn.SaveStep = ModelSettings.SaveStep;
            rnn.MaxIter = ModelSettings.MaxIteration;
            rnn.IsCRFTraining = IsCRFTraining;

            //Initialize RNNHelper
            RNNHelper.LearningRate = ModelSettings.LearningRate;
            RNNHelper.vecNormalLearningRate = new Vector<float>(RNNHelper.LearningRate);

            RNNHelper.GradientCutoff = ModelSettings.GradientCutoff;
            RNNHelper.vecMaxGrad = new Vector<float>(RNNHelper.GradientCutoff);
            RNNHelper.vecMinGrad = new Vector<float>(-RNNHelper.GradientCutoff);

            RNNHelper.IsConstAlpha = ModelSettings.IsConstAlpha;

            //Create tag-bigram transition probability matrix only for sequence RNN mode
            if (IsCRFTraining)
            {
                Logger.WriteLine("Initialize bigram transition for CRF output layer.");
                rnn.setTagBigramTransition(TrainingSet.CRFLabelBigramTransition);
            }

            Logger.WriteLine("");

            Logger.WriteLine("Iterative training begins ...");
            var lastPPL = double.MaxValue;
            var lastAlpha = RNNHelper.LearningRate;
            var iter = 0;
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
                var ppl = rnn.TrainNet(TrainingSet, iter);
                if (ppl >= lastPPL && lastAlpha != RNNHelper.LearningRate)
                {
                    //Although we reduce alpha value, we still cannot get better result.
                    Logger.WriteLine(
                        "Current perplexity({0}) is larger than the previous one({1}). End training early.", ppl,
                        lastPPL);
                    Logger.WriteLine("Current alpha: {0}, the previous alpha: {1}", RNNHelper.LearningRate, lastAlpha);
                    break;
                }
                lastAlpha = RNNHelper.LearningRate;

                //Validate the model by validated corpus
                if (ValidationSet != null)
                {
                    Logger.WriteLine("Verify model on validated corpus.");
                    if (rnn.ValidateNet(ValidationSet, iter))
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

        private RNN<T> CreateNetwork()
        {
            RNN<T> rnn;

            if (modelDirection == MODELDIRECTION.Forward)
            {
                var sparseFeatureSize = TrainingSet.SparseFeatureSize;
                if (ModelType == MODELTYPE.Seq2Seq)
                {
                    //[Sparse feature set of each state in target sequence][Sparse feature set of entire source sequence]
                    sparseFeatureSize += featurizer.Seq2SeqAutoEncoder.Config.SparseFeatureSize;
                    Logger.WriteLine("Sparse Feature Format: [{0}][{1}] = {2}",
                        TrainingSet.SparseFeatureSize, featurizer.Seq2SeqAutoEncoder.Config.SparseFeatureSize,
                        sparseFeatureSize);
                }

                var hiddenLayers = new List<SimpleLayer>();
                for (var i = 0; i < hiddenLayersConfig.Count; i++)
                {
                    SimpleLayer layer = null;
                    switch (hiddenLayersConfig[i].LayerType)
                    {
                        case LayerType.LSTM:
                            var lstmLayer = new LSTMLayer(hiddenLayersConfig[i] as LSTMLayerConfig);
                            layer = lstmLayer;
                            Logger.WriteLine("Create LSTM layer.");
                            break;

                        case LayerType.DropOut:
                            var dropoutLayer = new DropoutLayer(hiddenLayersConfig[i] as DropoutLayerConfig);
                            layer = dropoutLayer;
                            Logger.WriteLine("Create Dropout layer.");
                            break;
                    }

                    layer.InitializeWeights(sparseFeatureSize,
                        i == 0
                            ? GetCurrentLayerDenseFeatureSize(TrainingSet.DenseFeatureSize)
                            : GetCurrentLayerDenseFeatureSize(hiddenLayers[i - 1].LayerSize));

                    Logger.WriteLine(
                        $"Create hidden layer {i}: size = {layer.LayerSize}, sparse feature size = {layer.SparseFeatureSize}, dense feature size = {layer.DenseFeatureSize}");
                    hiddenLayers.Add(layer);
                }

                SimpleLayer outputLayer = null;
                outputLayerConfig.LayerSize = TrainingSet.TagSize;

                switch (outputLayerConfig.LayerType)
                {
                    case LayerType.SampledSoftmax:
                        Logger.WriteLine("Create sampled softmax layer as output layer");
                        outputLayer = new SampledSoftmaxLayer(outputLayerConfig as SampledSoftmaxLayerConfig);
                        outputLayer.InitializeWeights(0,
                            GetCurrentLayerDenseFeatureSize(hiddenLayers[hiddenLayers.Count - 1].LayerSize));
                        break;

                    case LayerType.Softmax:
                        Logger.WriteLine("Create softmax layer as output layer.");
                        outputLayer = new SoftmaxLayer(outputLayerConfig as SoftmaxLayerConfig);
                        outputLayer.InitializeWeights(sparseFeatureSize,
                            GetCurrentLayerDenseFeatureSize(hiddenLayers[hiddenLayers.Count - 1].LayerSize));
                        break;

                    case LayerType.Simple:
                        Logger.WriteLine("Create simple layer as output layer.");
                        outputLayer = new SimpleLayer(outputLayerConfig as SimpleLayerConfig);
                        outputLayer.InitializeWeights(sparseFeatureSize,
                            GetCurrentLayerDenseFeatureSize(hiddenLayers[hiddenLayers.Count - 1].LayerSize));
                        break;
                }

                rnn = new ForwardRNN<T>(hiddenLayers, outputLayer);
            }
            else
            {
                var forwardHiddenLayers = new List<SimpleLayer>();
                var backwardHiddenLayers = new List<SimpleLayer>();
                for (var i = 0; i < hiddenLayersConfig.Count; i++)
                {
                    SimpleLayer forwardLayer = null;
                    SimpleLayer backwardLayer = null;
                    switch (hiddenLayersConfig[i].LayerType)
                    {
                        case LayerType.LSTM:
                            //For LSTM layer
                            var forwardLSTMLayer = new LSTMLayer(hiddenLayersConfig[i] as LSTMLayerConfig);
                            forwardLayer = forwardLSTMLayer;

                            var backwardLSTMLayer = new LSTMLayer(hiddenLayersConfig[i] as LSTMLayerConfig);
                            backwardLayer = backwardLSTMLayer;

                            Logger.WriteLine("Create LSTM layer.");
                            break;

                        case LayerType.DropOut:
                            DropoutLayerConfig dropoutLayerConfig = hiddenLayersConfig[i] as DropoutLayerConfig;
                            dropoutLayerConfig.LayerSize = hiddenLayersConfig[i - 1].LayerSize * 2;

                            var forwardDropoutLayer = new DropoutLayer(dropoutLayerConfig);
                            forwardLayer = forwardDropoutLayer;
                            var backwardDropoutLayer = new DropoutLayer(dropoutLayerConfig);
                            backwardLayer = backwardDropoutLayer;

                            Logger.WriteLine("Create Dropout layer.");
                            break;
                    }

                    int denseFeatureSize = TrainingSet.DenseFeatureSize;
                    if (i > 0)
                    {
                        denseFeatureSize = forwardHiddenLayers[i - 1].LayerSize * 2;
                    }

                    forwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, denseFeatureSize);
                    backwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, denseFeatureSize);

                    Logger.WriteLine(
                        $"Create hidden layer {i}: size = {forwardLayer.LayerSize}, sparse feature size = {forwardLayer.SparseFeatureSize}, dense feature size = {forwardLayer.DenseFeatureSize}");

                    forwardHiddenLayers.Add(forwardLayer);
                    backwardHiddenLayers.Add(backwardLayer);
                }

                SimpleLayer outputLayer = null;
                outputLayerConfig.LayerSize = TrainingSet.TagSize;
                switch (outputLayerConfig.LayerType)
                {
                    case LayerType.SampledSoftmax:
                        Logger.WriteLine("Create sampled Softmax layer as output layer.");
                        outputLayer = new SampledSoftmaxLayer(outputLayerConfig as SampledSoftmaxLayerConfig);
                        outputLayer.InitializeWeights(0, forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize * 2);
                        break;

                    case LayerType.Softmax:
                        Logger.WriteLine("Create Softmax layer as output layer.");
                        outputLayer = new SoftmaxLayer(outputLayerConfig as SoftmaxLayerConfig);
                        outputLayer.InitializeWeights(TrainingSet.SparseFeatureSize,
                            forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize * 2);
                        break;

                    case LayerType.Simple:
                        Logger.WriteLine("Create simple layer as output layer.");
                        outputLayer = new SimpleLayer(outputLayerConfig as SimpleLayerConfig);
                        outputLayer.InitializeWeights(TrainingSet.SparseFeatureSize,
                            forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize * 2);
                        break;
                }

                rnn = new BiRNN<T>(forwardHiddenLayers, backwardHiddenLayers, outputLayer);
            }

            return rnn;
        }
    }
}