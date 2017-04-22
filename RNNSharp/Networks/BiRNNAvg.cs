using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp.Networks
{
    class BiRNNAvg<T> : BiRNN<T> where T : ISequence
    {
        public BiRNNAvg() : base()
        {

        }

        protected override void InitOutputLayerCache()
        {
            layersOutput = new List<float[][]>();
            for (int i = 0; i < numOfLayers; i++)
            {
                var layerOutputs = new float[MaxSeqLength][];
                for (int j = 0; j < MaxSeqLength; j++)
                {
                    layerOutputs[j] = new float[forwardHiddenLayers[i].LayerSize];
                }

                layersOutput.Add(layerOutputs);
            }

        }

        public override void CreateNetwork(List<LayerConfig> hiddenLayersConfig, LayerConfig outputLayerConfig, DataSet<T> TrainingSet, Config featurizer)
        {
            var forwardHiddenLayers = CreateLayers(hiddenLayersConfig);
            var backwardHiddenLayers = CreateLayers(hiddenLayersConfig);

            for (var i = 0; i < hiddenLayersConfig.Count; i++)
            {
                SimpleLayer forwardLayer = forwardHiddenLayers[i];
                SimpleLayer backwardLayer = backwardHiddenLayers[i];

                var denseFeatureSize = TrainingSet.DenseFeatureSize;
                if (i > 0)
                {
                    denseFeatureSize = forwardHiddenLayers[i - 1].LayerSize;
                }

                forwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, denseFeatureSize);
                backwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, denseFeatureSize);

                forwardLayer.SetRunningMode(RunningMode.Training);
                backwardLayer.SetRunningMode(RunningMode.Training);

                Logger.WriteLine($"Create hidden layer {i}: size = {forwardLayer.LayerSize}, sparse feature size = {forwardLayer.SparseFeatureSize}, dense feature size = {forwardLayer.DenseFeatureSize}");
            }

            outputLayerConfig.LayerSize = TrainingSet.TagSize;
            SimpleLayer outputLayer = CreateOutputLayer(outputLayerConfig, TrainingSet.SparseFeatureSize, forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize);
            outputLayer.SetRunningMode(RunningMode.Training);

            Logger.WriteLine($"Create a bi-directional recurrent neural network with {forwardHiddenLayers.Count} hidden layers. Forward and backward layers are concatnated.");
            InitCache(forwardHiddenLayers, backwardHiddenLayers, outputLayer);
        }

        public override RNN<T> Clone()
        {
            List<SimpleLayer> forwardLayers = new List<SimpleLayer>();
            List<SimpleLayer> backwardLayers = new List<SimpleLayer>();

            foreach (SimpleLayer layer in forwardHiddenLayers)
            {
                forwardLayers.Add(layer.CreateLayerSharedWegiths());
            }

            foreach (SimpleLayer layer in backwardHiddenLayers)
            {
                backwardLayers.Add(layer.CreateLayerSharedWegiths());
            }

            BiRNNAvg<T> rnn = new BiRNNAvg<T>();
            rnn.InitCache(forwardLayers, backwardLayers, OutputLayer.CreateLayerSharedWegiths());
            rnn.CRFTagTransWeights = CRFTagTransWeights;
            rnn.MaxSeqLength = MaxSeqLength;
            rnn.crfLocker = crfLocker;

            return rnn;
        }

        protected override void MergeForwardBackwardLayers(int numStates, int layerSize, int layerIdx)
        {
            //Merge forward and backward
            float[][] stateOutputs = layersOutput[layerIdx];
            for (var curState = 0; curState < numStates; curState++)
            {
                var forwardCells = forwardCellList[layerIdx][curState].Cells;
                var backwardCells = backwardCellList[layerIdx][curState].Cells;
                var mergedLayer = stateOutputs[curState];

                var i = 0;
                Vector<float> vecDiv2 = new Vector<float>(2.0f);
                while (i < layerSize)
                {
                    var v1 = new Vector<float>(forwardCells, i);
                    var v2 = new Vector<float>(backwardCells, i);
                    var vec = (v1 + v2) / vecDiv2;

                    vec.CopyTo(mergedLayer, i);

                    i += Vector<float>.Count;
                }
            }
        }

        public override int GetTopHiddenLayerSize()
        {
                return forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize;
        }
    }
}
