using AdvUtils;
using RNNSharp.Layers;
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

        protected override void InitLayersOutputCache()
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
                ILayer forwardLayer = forwardHiddenLayers[i];
                ILayer backwardLayer = backwardHiddenLayers[i];

                var denseFeatureSize = TrainingSet.DenseFeatureSize;
                var sparseFeatureSize = TrainingSet.SparseFeatureSize;
                if (i > 0)
                {
                    denseFeatureSize = forwardHiddenLayers[i - 1].LayerSize;
                    sparseFeatureSize = 0;
                }

                forwardLayer.InitializeWeights(sparseFeatureSize, denseFeatureSize);
                backwardLayer.InitializeWeights(sparseFeatureSize, denseFeatureSize);

                forwardLayer.SetRunningMode(RunningMode.Training);
                backwardLayer.SetRunningMode(RunningMode.Training);

                Logger.WriteLine($"Create hidden layer {i}: size = {forwardLayer.LayerSize}, sparse feature size = {forwardLayer.SparseFeatureSize}, dense feature size = {forwardLayer.DenseFeatureSize}");
            }

            outputLayerConfig.LayerSize = TrainingSet.TagSize;
            IOutputLayer outputLayer = CreateOutputLayer(outputLayerConfig, 0, forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize);
            outputLayer.SetRunningMode(RunningMode.Training);

            Logger.WriteLine($"Create a bi-directional recurrent neural network with {forwardHiddenLayers.Count} hidden layers. Forward and backward layers are concatnated.");
            InitCache(forwardHiddenLayers, backwardHiddenLayers, outputLayer);
        }

        public override RNN<T> Clone()
        {
            List<ILayer> forwardLayers = new List<ILayer>();
            List<ILayer> backwardLayers = new List<ILayer>();

            foreach (ILayer layer in forwardHiddenLayers)
            {
                forwardLayers.Add(layer.CreateLayerSharedWegiths());
            }

            foreach (ILayer layer in backwardHiddenLayers)
            {
                backwardLayers.Add(layer.CreateLayerSharedWegiths());
            }

            BiRNNAvg<T> rnn = new BiRNNAvg<T>();
            rnn.InitCache(forwardLayers, backwardLayers, (IOutputLayer)OutputLayer.CreateLayerSharedWegiths());
            rnn.CRFWeights = CRFWeights;
            rnn.MaxSeqLength = MaxSeqLength;
            rnn.bVQ = bVQ;
            rnn.IsCRFTraining = IsCRFTraining;
            if (rnn.IsCRFTraining)
            {
                rnn.InitializeCRFVariablesForTraining();
            }

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

                var moreItems = (layerSize % Vector<float>.Count);
                while (i < layerSize - moreItems)
                {
                    var v1 = new Vector<float>(forwardCells, i);
                    var v2 = new Vector<float>(backwardCells, i);
                    var vec = (v1 + v2) / vecDiv2;

                    vec.CopyTo(mergedLayer, i);

                    i += Vector<float>.Count;
                }

                while (i < layerSize)
                {
                    mergedLayer[i] = (forwardCells[i] + backwardCells[i]) / 2.0f;
                    i++;
                }
            }
        }

        public override int GetTopHiddenLayerSize()
        {
                return forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize;
        }


        /// <summary>
        ///     Pass error from the last layer to the first layer
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="seqFinalOutput"></param>
        /// <returns></returns>
        protected override void ComputeDeepErr(Sequence pSequence)
        {
            var numStates = pSequence.States.Length;
            var numLayers = forwardHiddenLayers.Count;

            //Calculate output layer error
            for (var curState = 0; curState < numStates; curState++)
            {
                OutputLayer.Cells = OutputCells[curState].Cells;
                OutputLayer.Errs = OutputCells[curState].Errs;
                OutputLayer.ComputeOutputLoss(CRFSeqOutput, pSequence.States[curState], curState);
            }


            ////Now we already have err in output layer, let's pass them back to other layers
            ////Pass error from i+1 to i layer
            var errLayer1 = forwardCellList[numLayers - 1];
            var errLayer2 = backwardCellList[numLayers - 1];

            for (var curState = 0; curState < numStates; curState++)
            {
                OutputLayer.Errs = OutputCells[curState].Errs;
                OutputLayer.ComputeLayerErr(errLayer1[curState].Errs);

                errLayer1[curState].Errs.CopyTo(errLayer2[curState].Errs, 0);
            }

            for (var i = numLayers - 2; i >= 0; i--)
            {
                var lastForwardLayer = forwardHiddenLayers[i + 1];
                var errLayerF = forwardCellList[i];
                var srcErrLayerF = forwardCellList[i + 1];

                var lastBackwardLayer = backwardHiddenLayers[i + 1];
                var errLayerB = backwardCellList[i];
                var srcErrLayerB = backwardCellList[i + 1];

                for (var curState = 0; curState < numStates; curState++)
                {
                    var errLayerFCur = errLayerF[curState];
                    var errLayerBCur = errLayerB[curState];

                    lastForwardLayer.Errs = srcErrLayerF[curState].Errs;
                    lastForwardLayer.ComputeLayerErr(errLayerFCur.Errs);

                    lastBackwardLayer.Errs = srcErrLayerB[curState].Errs;
                    lastBackwardLayer.ComputeLayerErr(errLayerBCur.Errs);
                }

            }

        }
    }
}
