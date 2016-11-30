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
        public DataSet<T> TrainingSet { get; set; }
        public DataSet<T> ValidationSet { get; set; }
        public Featurizer Featurizer { get; set; }

        public RNNEncoder(ModelSetting modelSetting)
        {
            ModelSettings = modelSetting;
        }


        private int GetCurrentLayerDenseFeatureSize(int inputDenseFeatureSize)
        {
            //If current training is for sequence-to-sequence, we append features from source sequence to standard features.
            if (ModelSettings.IsSeq2SeqTraining)
            {
                //[Dense feature set of each state in target sequence][Dense feature set of entire source sequence]
                inputDenseFeatureSize += Featurizer.AutoEncoder.GetTopHiddenLayerSize();
            }

            return inputDenseFeatureSize;
        }

        public void Train()
        {
            RNN<T> rnn;

            if (ModelSettings.ModelDirection == 0)
            {
                int sparseFeatureSize = TrainingSet.SparseFeatureSize;
                if (ModelSettings.IsSeq2SeqTraining)
                {
                    //[Sparse feature set of each state in target sequence][Sparse feature set of entire source sequence]
                    sparseFeatureSize += Featurizer.AutoEncoder.Featurizer.SparseFeatureSize;
                    Logger.WriteLine("Sparse Feature Format: [{0}][{1}] = {2}",
                        TrainingSet.SparseFeatureSize, Featurizer.AutoEncoder.Featurizer.SparseFeatureSize, sparseFeatureSize);
                }


                List<SimpleLayer> hiddenLayers = new List<SimpleLayer>();
                for (int i = 0; i < ModelSettings.HiddenLayerSizeList.Count; i++)
                {
                    SimpleLayer layer = null;
                    if (ModelSettings.ModelType == LayerType.BPTT)
                    {
                        BPTTLayer bpttLayer = new BPTTLayer(ModelSettings.HiddenLayerSizeList[i], ModelSettings);
                        layer = bpttLayer;
                    }
                    else if (ModelSettings.ModelType == LayerType.LSTM)
                    {
                        LSTMLayer lstmLayer = new LSTMLayer(ModelSettings.HiddenLayerSizeList[i], ModelSettings);
                        layer = lstmLayer;
                    }
                    else
                    {
                        throw new System.Exception(string.Format("Invalidate hidden layer type: {0}", ModelSettings.ModelType.ToString()));
                    }

                    if (i == 0)
                    {
                        Logger.WriteLine("Create hidden layer {0}: size = {1}, sparse feature size = {2}, dense feature size = {3}",
                            i, ModelSettings.HiddenLayerSizeList[i], sparseFeatureSize, TrainingSet.DenseFeatureSize);

                        if (ModelSettings.IsSeq2SeqTraining)
                        {
                            Logger.WriteLine("For seq2seq training, we have {0} sprase feature and {1} dense feature from source sequence.",
                                Featurizer.AutoEncoder.Featurizer.SparseFeatureSize, Featurizer.AutoEncoder.GetTopHiddenLayerSize());
                        }

                        layer.InitializeWeights(sparseFeatureSize, GetCurrentLayerDenseFeatureSize(TrainingSet.DenseFeatureSize));
                    }
                    else
                    {
                        Logger.WriteLine("Create hidden layer {0}: size = {1}, sparse feature size = {2}, dense feature size = {3}",
                            i, ModelSettings.HiddenLayerSizeList[i], sparseFeatureSize, hiddenLayers[i - 1].LayerSize);

                        layer.InitializeWeights(sparseFeatureSize, GetCurrentLayerDenseFeatureSize(hiddenLayers[i - 1].LayerSize));
                    }
                    hiddenLayers.Add(layer);
                }

                if (ModelSettings.Dropout > 0)
                {
                    Logger.WriteLine("Adding dropout layer");

                    DropoutLayer dropoutLayer = new DropoutLayer(hiddenLayers[hiddenLayers.Count - 1].LayerSize, ModelSettings);
                    dropoutLayer.InitializeWeights(0, GetCurrentLayerDenseFeatureSize(hiddenLayers[hiddenLayers.Count - 1].LayerSize));
                    hiddenLayers.Add(dropoutLayer);
                }

                SimpleLayer outputLayer;
                if (ModelSettings.OutputLayerType == LayerType.NCESoftmax)
                {
                    Logger.WriteLine("Create NCESoftmax layer as output layer");
                    NCEOutputLayer nceOutputLayer = new NCEOutputLayer(TrainingSet.TagSize, ModelSettings);
                    nceOutputLayer.BuildStatisticData<T>(TrainingSet);
                    nceOutputLayer.InitializeWeights(0, GetCurrentLayerDenseFeatureSize(hiddenLayers[hiddenLayers.Count - 1].LayerSize));
                    outputLayer = nceOutputLayer;
                }
                else if (ModelSettings.OutputLayerType == LayerType.Softmax)
                {
                    Logger.WriteLine("Create Softmax layer as output layer.");
                    outputLayer = new SimpleLayer(TrainingSet.TagSize);
                    outputLayer.InitializeWeights(sparseFeatureSize, GetCurrentLayerDenseFeatureSize(hiddenLayers[hiddenLayers.Count - 1].LayerSize));
                }
                else
                {
                    throw new System.Exception(string.Format("Invalidate output layer type: {0}", ModelSettings.OutputLayerType.ToString()));
                }

                rnn = new ForwardRNN<T>(hiddenLayers, outputLayer);
            }
            else
            {
                List<SimpleLayer> forwardHiddenLayers = new List<SimpleLayer>();
                List<SimpleLayer> backwardHiddenLayers = new List<SimpleLayer>();
                for (int i = 0; i < ModelSettings.HiddenLayerSizeList.Count; i++)
                {
                    SimpleLayer forwardLayer = null;
                    SimpleLayer backwardLayer = null;
                    if (ModelSettings.ModelType == LayerType.BPTT)
                    {
                        //For BPTT layer
                        BPTTLayer forwardBPTTLayer = new BPTTLayer(ModelSettings.HiddenLayerSizeList[i], ModelSettings);
                        forwardLayer = forwardBPTTLayer;

                        BPTTLayer backwardBPTTLayer = new BPTTLayer(ModelSettings.HiddenLayerSizeList[i], ModelSettings);
                        backwardLayer = backwardBPTTLayer;
                    }
                    else if (ModelSettings.ModelType == LayerType.LSTM)
                    {
                        //For LSTM layer
                        LSTMLayer forwardLSTMLayer = new LSTMLayer(ModelSettings.HiddenLayerSizeList[i], ModelSettings);
                        forwardLayer = forwardLSTMLayer;

                        LSTMLayer backwardLSTMLayer = new LSTMLayer(ModelSettings.HiddenLayerSizeList[i], ModelSettings);
                        backwardLayer = backwardLSTMLayer;
                    }
                    else
                    {
                        throw new System.Exception(string.Format("Invalidate hidden layer type: {0}", ModelSettings.ModelType.ToString()));
                    }

                    if (i == 0)
                    {
                        Logger.WriteLine("Create hidden layer {0}: size = {1}, sparse feature size = {2}, dense feature size = {3}",
                            i, ModelSettings.HiddenLayerSizeList[i], TrainingSet.SparseFeatureSize, TrainingSet.DenseFeatureSize);

                        forwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, TrainingSet.DenseFeatureSize);
                        backwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, TrainingSet.DenseFeatureSize);
                    }
                    else
                    {
                        Logger.WriteLine("Create hidden layer {0}: size = {1}, sparse feature size = {2}, dense feature size = {3}",
                            i, ModelSettings.HiddenLayerSizeList[i], TrainingSet.SparseFeatureSize, forwardHiddenLayers[i - 1].LayerSize);

                        forwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, forwardHiddenLayers[i - 1].LayerSize);
                        backwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, backwardHiddenLayers[i - 1].LayerSize);
                    }

                    forwardHiddenLayers.Add(forwardLayer);
                    backwardHiddenLayers.Add(backwardLayer);
                }

                if (ModelSettings.Dropout > 0)
                {
                    Logger.WriteLine("Adding dropout layers");
                    DropoutLayer forwardDropoutLayer = new DropoutLayer(forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize, ModelSettings);
                    DropoutLayer backwardDropoutLayer = new DropoutLayer(backwardHiddenLayers[backwardHiddenLayers.Count - 1].LayerSize, ModelSettings);

                    forwardDropoutLayer.InitializeWeights(0, forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize);
                    backwardDropoutLayer.InitializeWeights(0, backwardHiddenLayers[backwardHiddenLayers.Count - 1].LayerSize);

                    forwardHiddenLayers.Add(forwardDropoutLayer);
                    backwardHiddenLayers.Add(backwardDropoutLayer);
                }

                SimpleLayer outputLayer;
                if (ModelSettings.OutputLayerType == LayerType.NCESoftmax)
                {
                    Logger.WriteLine("Create NCESoftmax layer as output layer.");
                    NCEOutputLayer nceOutputLayer = new NCEOutputLayer(TrainingSet.TagSize, ModelSettings);
                    nceOutputLayer.BuildStatisticData<T>(TrainingSet);
                    nceOutputLayer.InitializeWeights(0, forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize);
                    outputLayer = nceOutputLayer;
                }
                else if (ModelSettings.OutputLayerType == LayerType.Softmax)
                {
                    Logger.WriteLine("Create Softmax layer as output layer.");
                    outputLayer = new SimpleLayer(TrainingSet.TagSize);
                    outputLayer.InitializeWeights(TrainingSet.SparseFeatureSize, forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize);
                }
                else
                {
                    throw new System.Exception(string.Format("Invalidate output layer type: {0}", ModelSettings.OutputLayerType.ToString()));
                }

                rnn = new BiRNN<T>(forwardHiddenLayers, backwardHiddenLayers, outputLayer);
            }

            rnn.ModelDirection = (MODELDIRECTION)ModelSettings.ModelDirection;
            rnn.bVQ = (ModelSettings.VQ != 0) ? true : false;
            rnn.ModelFile = ModelSettings.ModelFile;
            rnn.SaveStep = ModelSettings.SaveStep;
            rnn.MaxIter = ModelSettings.MaxIteration;
            rnn.IsCRFTraining = ModelSettings.IsCRFTraining;
            rnn.ModelType = ModelSettings.IsSeq2SeqTraining ? MODELTYPE.SEQ2SEQ : MODELTYPE.SEQLABEL;

            if (rnn.ModelDirection == MODELDIRECTION.BI_DIRECTIONAL && rnn.ModelType == MODELTYPE.SEQ2SEQ)
            {
                throw new System.Exception("Bi-directional RNN model doesn't support sequence-to-sequence model.");
            }

            RNNHelper.LearningRate = ModelSettings.LearningRate;
            RNNHelper.GradientCutoff = ModelSettings.GradientCutoff;
            RNNHelper.IsConstAlpha = ModelSettings.IsConstAlpha;
            
            //Create tag-bigram transition probability matrix only for sequence RNN mode
            if (ModelSettings.IsCRFTraining)
            {
                rnn.setTagBigramTransition(TrainingSet.CRFLabelBigramTransition);
            }

            Logger.WriteLine("");

            Logger.WriteLine("Iterative training begins ...");
            double lastPPL = double.MaxValue;
            double lastAlpha = RNNHelper.LearningRate;
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
                        Logger.WriteLine("Saving better model into file {0}...", ModelSettings.ModelFile);
                        rnn.SaveModel(ModelSettings.ModelFile);
                    }
                }
                else if (ppl < lastPPL)
                {
                    //We don't have validate corpus, but we get a better result on training corpus
                    //We got better result on validated corpus, save this model
                    Logger.WriteLine("Saving better model into file {0}...", ModelSettings.ModelFile);
                    rnn.SaveModel(ModelSettings.ModelFile);
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
