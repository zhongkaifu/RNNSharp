using AdvUtils;
using RNNSharp.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp.Networks
{
    internal class BiRNN<T> : RNN<T> where T : ISequence
    {
        protected List<ILayer> backwardHiddenLayers = new List<ILayer>();
        protected List<ILayer> forwardHiddenLayers = new List<ILayer>();
        protected int numOfLayers => forwardHiddenLayers.Count;
        protected List<float[][]> layersOutput;
        protected List<Neuron[]> forwardCellList;
        protected List<Neuron[]> backwardCellList;
 //IOutputLayer[] seqFinalOutput;
        protected Neuron[] OutputCells;

        public override void UpdateWeights()
        {
            foreach (var layer in backwardHiddenLayers)
            {
                layer.UpdateWeights();
            }

            foreach (var layer in forwardHiddenLayers)
            {
                layer.UpdateWeights();
            }

            OutputLayer.UpdateWeights();
        }

        protected void InitCache(List<ILayer> s_forwardRNN, List<ILayer> s_backwardRNN, IOutputLayer outputLayer)
        {
            forwardHiddenLayers = s_forwardRNN;
            backwardHiddenLayers = s_backwardRNN;

            //Initialize output layer
            OutputLayer = outputLayer;

            forwardCellList = new List<Neuron[]>();
            backwardCellList = new List<Neuron[]>();

            for (int i = 0; i < numOfLayers; i++)
            {
                var forwardCells = new Neuron[MaxSeqLength];
                var backwardCells = new Neuron[MaxSeqLength];

                for (int j = 0; j < MaxSeqLength; j++)
                {
                    if (forwardHiddenLayers[i] is DropoutLayer)
                    {
                        forwardCells[j] = new DropoutNeuron(forwardHiddenLayers[i].LayerSize);
                        backwardCells[j] = new DropoutNeuron(forwardHiddenLayers[i].LayerSize);
                    }
                    else if (forwardHiddenLayers[i] is LSTMLayer)
                    {
                        forwardCells[j] = new LSTMNeuron(forwardHiddenLayers[i].LayerSize);
                        backwardCells[j] = new LSTMNeuron(forwardHiddenLayers[i].LayerSize);
                    }
                    else
                    {
                        forwardCells[j] = new Neuron(forwardHiddenLayers[i].LayerSize);
                        backwardCells[j] = new Neuron(forwardHiddenLayers[i].LayerSize);
                    }
                }

                forwardCellList.Add(forwardCells);
                backwardCellList.Add(backwardCells);
            }

            OutputCells = new Neuron[MaxSeqLength];
            for (var i = 0; i < MaxSeqLength; i++)
            {
                OutputCells[i] = new Neuron(OutputLayer.LayerSize);
            }

            InitLayersOutputCache();
        }

        public BiRNN()
        {
        }

     
        protected virtual void InitLayersOutputCache()
        {
            layersOutput = new List<float[][]>();
            for (int i = 0; i < numOfLayers; i++)
            {
                var layerOutputs = new float[MaxSeqLength][];
                for (int j = 0; j < MaxSeqLength; j++)
                {
                    layerOutputs[j] = new float[forwardHiddenLayers[i].LayerSize * 2];
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
                if (i > 0)
                {
                    denseFeatureSize = forwardHiddenLayers[i - 1].LayerSize * 2;
                }

                forwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, denseFeatureSize);
                backwardLayer.InitializeWeights(TrainingSet.SparseFeatureSize, denseFeatureSize);

                forwardLayer.SetRunningMode(RunningMode.Training);
                backwardLayer.SetRunningMode(RunningMode.Training);

                Logger.WriteLine($"Create hidden layer {i}: size = {forwardLayer.LayerSize}, sparse feature size = {forwardLayer.SparseFeatureSize}, dense feature size = {forwardLayer.DenseFeatureSize}");
            }

            outputLayerConfig.LayerSize = TrainingSet.TagSize;
            IOutputLayer outputLayer = CreateOutputLayer(outputLayerConfig, TrainingSet.SparseFeatureSize, forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize * 2);
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

            BiRNN<T> rnn = new BiRNN<T>();
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

        public override void CleanStatusForTraining()
        {
            foreach (var layer in forwardHiddenLayers)
            {
                layer.CleanForTraining();
            }

            foreach (var layer in backwardHiddenLayers)
            {
                layer.CleanForTraining();
            }

            OutputLayer.CleanForTraining();
        }

        protected virtual void MergeForwardBackwardLayers(int numStates, int layerSize, int layerIdx)
        {
            //Merge forward and backward
            float[][] stateOutputs = layersOutput[layerIdx];
            for (var curState = 0; curState < numStates; curState++)
            {
                var forwardCells = forwardCellList[layerIdx][curState].Cells;
                var backwardCells = backwardCellList[layerIdx][curState].Cells;
                var mergedLayer = stateOutputs[curState];

                var i = 0;
                var moreItems = (layerSize % Vector<float>.Count);
                while (i < layerSize - moreItems)
                {
                    var v1 = new Vector<float>(forwardCells, i);
                    var v2 = new Vector<float>(backwardCells, i);

                    v1.CopyTo(mergedLayer, i);
                    v2.CopyTo(mergedLayer, layerSize + i);

                    i += Vector<float>.Count;
                }

                while (i < layerSize)
                {
                    mergedLayer[i] = forwardCells[i];
                    mergedLayer[layerSize + i] = backwardCells[i];
                    i++;
                }
            }
        }



        private void ComputeMiddleLayers(Sequence pSequence, ILayer forwardLayer, ILayer backwardLayer, RunningMode runningMode, int layerIdx)
        {
            var numStates = pSequence.States.Length;
            float[][] lastLayerOutputs = layersOutput[layerIdx - 1];


            //Computing forward RNN
            forwardLayer.Reset();
            for (var curState = 0; curState < numStates; curState++)
            {
                var state = pSequence.States[curState];
                forwardLayer.ForwardPass(state.SparseFeature, lastLayerOutputs[curState]);
                forwardLayer.CopyNeuronTo(forwardCellList[layerIdx][curState]);
            }

            //Computing backward RNN
            backwardLayer.Reset();
            for (var curState = numStates - 1; curState >= 0; curState--)
            {
                var state = pSequence.States[curState];
                backwardLayer.ForwardPass(state.SparseFeature, lastLayerOutputs[curState]);
                backwardLayer.CopyNeuronTo(backwardCellList[layerIdx][curState]);
            }


            //Merge forward and backward
            MergeForwardBackwardLayers(numStates, forwardLayer.LayerSize, layerIdx);
        }

        /// <summary>
        ///     Compute the output of bottom layer
        /// </summary>
        /// <param name="sequence"></param>
        /// <param name="forwardLayer"></param>
        /// <param name="backwardLayer"></param>
        /// <returns></returns>
        private void ComputeBottomLayer(Sequence sequence, ILayer forwardLayer, ILayer backwardLayer, RunningMode runningMode)
        {
            var numStates = sequence.States.Length;

            //Computing forward RNN
            forwardLayer.Reset();
            for (var curState = 0; curState < numStates; curState++)
            {
                var state = sequence.States[curState];
                forwardLayer.ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());
                forwardLayer.CopyNeuronTo(forwardCellList[0][curState]);
            }


            //Computing backward RNN
            backwardLayer.Reset();
            for (var curState = numStates - 1; curState >= 0; curState--)
            {
                var state = sequence.States[curState];
                backwardLayer.ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());
                backwardLayer.CopyNeuronTo(backwardCellList[0][curState]);
            }

            //Merge forward and backward
            MergeForwardBackwardLayers(numStates, forwardLayer.LayerSize, 0);
        }

        private void ComputeTopLayer(Sequence pSequence, out Matrix<float> rawOutputLayer, RunningMode runningMode, bool outputRawScore)
        {
            var numStates = pSequence.States.Length;
            var lastLayerOutputs = layersOutput[forwardHiddenLayers.Count - 1];

            //Calculate output layer
            Matrix<float> tmpOutputResult = null;
            if (outputRawScore)
            {
                tmpOutputResult = new Matrix<float>(numStates, OutputLayer.LayerSize);
            }

            OutputLayer.LabelShortList = pSequence.States.Select(state => state.Label).ToList();
            for (var curState = 0; curState < numStates; curState++)
            {
                var state = pSequence.States[curState];
                OutputLayer.ForwardPass(state.SparseFeature, lastLayerOutputs[curState]);
                OutputLayer.CopyNeuronTo(OutputCells[curState]);

                if (outputRawScore)
                {
                    OutputLayer.Cells.CopyTo(tmpOutputResult[curState], 0);
                }
            }

            rawOutputLayer = tmpOutputResult;
        }

        public override int GetTopHiddenLayerSize()
        {
            return forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize * 2;
        }

        public override float[][] ComputeTopHiddenLayerOutput(Sequence pSequence)
        {
            ComputeBottomLayer(pSequence, forwardHiddenLayers[0], backwardHiddenLayers[0], RunningMode.Test);
            for (var i = 1; i < forwardHiddenLayers.Count; i++)
            {
                ComputeMiddleLayers(pSequence, forwardHiddenLayers[i], backwardHiddenLayers[i], RunningMode.Test, i);
            }

            return layersOutput[forwardHiddenLayers.Count - 1];
        }

        /// <summary>
        ///     Computing the output of each layer in the neural network
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="isTraining"></param>
        /// <param name="layerList"></param>
        /// <param name="rawOutputLayer"></param>
        /// <returns></returns>
        private void ComputeLayers(Sequence pSequence, RunningMode runningMode, out Matrix<float> rawOutputLayer, bool outputRawScore)
        {
            ComputeBottomLayer(pSequence, forwardHiddenLayers[0], backwardHiddenLayers[0], runningMode);
            for (var i = 1; i < forwardHiddenLayers.Count; i++)
            {
                ComputeMiddleLayers(pSequence, forwardHiddenLayers[i], backwardHiddenLayers[i], runningMode, i);
            }

            ComputeTopLayer(pSequence, out rawOutputLayer, runningMode, outputRawScore);
        }

        /// <summary>
        ///     Pass error from the last layer to the first layer
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="seqFinalOutput"></param>
        /// <returns></returns>
        protected virtual void ComputeDeepErr(Sequence pSequence)
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
                List<float[]> destErrsList = new List<float[]>();
                destErrsList.Add(errLayer1[curState].Errs);
                destErrsList.Add(errLayer2[curState].Errs);

                OutputLayer.Errs = OutputCells[curState].Errs;
                OutputLayer.ComputeLayerErr(destErrsList);
            }

            Vector<float> vecTwo = new Vector<float>(2.0f);
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

                    List<float[]> destErrList = new List<float[]>();
                    destErrList.Add(errLayerFCur.Errs);
                    destErrList.Add(errLayerBCur.Errs);

                    lastForwardLayer.Errs = srcErrLayerF[curState].Errs;
                    lastForwardLayer.ComputeLayerErr(destErrList);

                    lastBackwardLayer.Errs = srcErrLayerB[curState].Errs;
                    lastBackwardLayer.ComputeLayerErr(destErrList, false);


                    int j = 0;
                    int errLength = errLayerFCur.Errs.Length;
                    var moreItems = (errLength % Vector<float>.Count);
                    while (j < errLength - moreItems)
                    {
                        Vector<float> vecErrLayerF = new Vector<float>(errLayerFCur.Errs, j);
                        Vector<float> vecErrLayerB = new Vector<float>(errLayerBCur.Errs, j);

                        vecErrLayerF /= vecTwo;
                        vecErrLayerB /= vecTwo;

                        vecErrLayerF.CopyTo(errLayerFCur.Errs, j);
                        vecErrLayerB.CopyTo(errLayerBCur.Errs, j);

                        j += Vector<float>.Count;
                    }

                    while (j < errLength)
                    {
                        errLayerFCur.Errs[j] /= 2.0f;
                        errLayerBCur.Errs[j] /= 2.0f;

                        j++;
                    }

                }

            }

        }
        protected virtual void DeepLearningNet(Sequence pSequence)
        {
            var numStates = pSequence.States.Length;
            var numLayers = forwardHiddenLayers.Count;

            //Learning output layer
            for (var curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.States[curState];
                UpdateWeightsAt(OutputLayer, OutputCells[curState], state.SparseFeature, layersOutput[numLayers - 1][curState]);
            }

            for (var i = 0; i < numLayers; i++)
            {
                float[][] layerOutputs_i = (i > 0) ? layersOutput[i - 1] : null;
                Neuron[] forwardNeurons = forwardCellList[i];
                var forwardLayer = forwardHiddenLayers[i];
                for (var curState = 0; curState < numStates; curState++)
                {
                    State state = pSequence.States[curState];
                    UpdateWeightsAt(forwardLayer, forwardNeurons[curState], state.SparseFeature, (i == 0) ? state.DenseFeature.CopyTo() : layerOutputs_i[curState]);
                }

                Neuron[] backwardNeurons = backwardCellList[i];
                var backwardLayer = backwardHiddenLayers[i];
                for (var curState2 = numStates - 1;curState2 >= 0;curState2--)
                {
                    State state = pSequence.States[curState2];
                    UpdateWeightsAt(backwardLayer, backwardNeurons[curState2], state.SparseFeature, (i == 0) ? state.DenseFeature.CopyTo() : layerOutputs_i[curState2]);
                }
            }
        }

        protected void UpdateWeightsAt(ILayer layer, Neuron neuron, SparseVector sparseFeature, float[] denseFeature)
        {
            layer.SparseFeatureGroups = new List<SparseVector>();
            layer.SparseFeatureGroups.Add(sparseFeature);
            layer.DenseFeatureGroups = new List<float[]>();
            layer.DenseFeatureGroups.Add(denseFeature);
            layer.SetNeuron(neuron);
            layer.BackwardPass();
        }

        public override int[] ProcessSequence(ISentence sentence, Config featurizer, RunningMode runningMode, bool outputRawScore, out Matrix<float> m)
        {
            var seq = featurizer.BuildSequence(sentence as Sentence);
            return ProcessSequence(seq, runningMode, outputRawScore, out m);
        }

        /// <summary>
        ///     Process a given sequence by bi-directional recurrent neural network
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="runningMode"></param>
        /// <returns></returns>
        public override int[] ProcessSequence(ISequence sequence, RunningMode runningMode, bool outputRawScore, out Matrix<float> rawOutputLayer)
        {
            Sequence pSequence = sequence as Sequence;
            //Forward process from bottom layer to top layer
            int[] seqBestOutput;
            ComputeLayers(pSequence, runningMode, out rawOutputLayer, outputRawScore);

            //Get best output result of each state
            var numStates = pSequence.States.Length;
            seqBestOutput = new int[numStates];

            for (var curState = 0; curState < numStates; curState++)
            {
                seqBestOutput[curState] = OutputCells[curState].GetMaxCellIndex();
            }

            if (runningMode == RunningMode.Training)
            {
                //In training mode, we calculate each layer's error and update their net weights
                ComputeDeepErr(pSequence);
                DeepLearningNet(pSequence);
            }

            return seqBestOutput;
        }

        /// <summary>
        ///     Process a given sequence by bi-directional recurrent neural network and CRF
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="runningMode"></param>
        /// <returns></returns>
        public override int[] ProcessSequenceCRF(Sequence pSequence, RunningMode runningMode)
        {
            //Reset the network
            var numStates = pSequence.States.Length;
            Matrix<float> rawOutputLayer;

            ComputeLayers(pSequence, runningMode, out rawOutputLayer, true);

            ForwardBackward(numStates, rawOutputLayer);

            var predict = Viterbi(rawOutputLayer, numStates);

            if (runningMode == RunningMode.Training)
            {
                UpdateBigramTransition(pSequence);
                ComputeDeepErr(pSequence);
                DeepLearningNet(pSequence);
            }

            return predict;
        }

        public override void SaveModel(string filename)
        {
            //Save meta data
            using (var sw = new StreamWriter(filename))
            {
                var fo = new BinaryWriter(sw.BaseStream);
                fo.Write(IsCRFTraining);
                fo.Write(forwardHiddenLayers.Count);
                //Save forward layers
                foreach (var layer in forwardHiddenLayers)
                {
                    fo.Write((int)layer.LayerType);
                    layer.Save(fo);
                }
                //Save backward layers
                foreach (var layer in backwardHiddenLayers)
                {
                    fo.Write((int)layer.LayerType);
                    layer.Save(fo);
                }
                //Save output layer
                fo.Write((int)OutputLayer.LayerType);
                OutputLayer.Save(fo);

                if (IsCRFTraining)
                {
                    // Save CRF features weights
                    RNNHelper.SaveMatrix(CRFWeights, fo);
                }
            }
        }

        public override void LoadModel(string filename, bool bTrain = false)
        {
            Logger.WriteLine(Logger.Level.info, "Loading bi-directional model: {0}", filename);

            using (var sr = new StreamReader(filename))
            {
                var br = new BinaryReader(sr.BaseStream);

                IsCRFTraining = br.ReadBoolean();
                var layerSize = br.ReadInt32();
                LayerType layerType = LayerType.None;

                //Load forward layers from file
                forwardHiddenLayers = new List<ILayer>();
                for (var i = 0; i < layerSize; i++)
                {
                    layerType = (LayerType)br.ReadInt32();
                    forwardHiddenLayers.Add(Load(layerType, br, bTrain));

                    ILayer layer = forwardHiddenLayers[forwardHiddenLayers.Count - 1];
                    if (bTrain)
                    {
                        layer.SetRunningMode(RunningMode.Training);
                    }
                    else
                    {
                        layer.SetRunningMode(RunningMode.Test);
                    }
                }

                //Load backward layers from file
                backwardHiddenLayers = new List<ILayer>();
                for (var i = 0; i < layerSize; i++)
                {
                    layerType = (LayerType)br.ReadInt32();
                    backwardHiddenLayers.Add(Load(layerType, br, bTrain));

                    ILayer layer = backwardHiddenLayers[backwardHiddenLayers.Count - 1];
                    if (bTrain)
                    {
                        layer.SetRunningMode(RunningMode.Training);
                    }
                    else
                    {
                        layer.SetRunningMode(RunningMode.Test);
                    }
                }

                Logger.WriteLine("Create output layer");
                layerType = (LayerType)br.ReadInt32();
                OutputLayer = (IOutputLayer)Load(layerType, br, bTrain);

                if (bTrain)
                {
                    OutputLayer.SetRunningMode(RunningMode.Training);
                }
                else
                {
                    OutputLayer.SetRunningMode(RunningMode.Test);
                }

                if (IsCRFTraining)
                {
                    Logger.WriteLine("Loading CRF tag trans weights...");
                    CRFWeights = RNNHelper.LoadMatrix(br);
                }

                if (bTrain)
                {
                    InitCache(forwardHiddenLayers, backwardHiddenLayers, (IOutputLayer)OutputLayer.CreateLayerSharedWegiths());
                }
            }
        }
    }
}