using AdvUtils;
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
        protected List<SimpleLayer> backwardHiddenLayers = new List<SimpleLayer>();
        protected List<SimpleLayer> forwardHiddenLayers = new List<SimpleLayer>();
        protected int numOfLayers => forwardHiddenLayers.Count;
        protected List<float[][]> layersOutput;
        protected List<Neuron[]> forwardCellList;
        protected List<Neuron[]> backwardCellList;

        List<float[][]> fErrLayers;
        List<float[][]> bErrLayers;
        SimpleLayer[] seqFinalOutput;

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

        public void InitCache(List<SimpleLayer> s_forwardRNN, List<SimpleLayer> s_backwardRNN, SimpleLayer outputLayer)
        {
            forwardHiddenLayers = s_forwardRNN;
            backwardHiddenLayers = s_backwardRNN;

            //Initialize output layer
            OutputLayer = outputLayer;

            forwardCellList = new List<Neuron[]>();
            backwardCellList = new List<Neuron[]>();
            fErrLayers = new List<float[][]>();
            bErrLayers = new List<float[][]>();

            for (int i = 0; i < numOfLayers; i++)
            {
                var forwardCells = new Neuron[MaxSeqLength];
                var backwardCells = new Neuron[MaxSeqLength];
                var fErrLayer = new float[MaxSeqLength][];
                var bErrLayer = new float[MaxSeqLength][];

                for (int j = 0; j < MaxSeqLength; j++)
                {
                    if (forwardHiddenLayers[i] is DropoutLayer)
                    {
                        forwardCells[j] = new DropoutNeuron();
                        backwardCells[j] = new DropoutNeuron();

                        ((DropoutNeuron)forwardCells[j]).mask = new bool[forwardHiddenLayers[i].LayerSize];
                        ((DropoutNeuron)backwardCells[j]).mask = new bool[forwardHiddenLayers[i].LayerSize];
                    }
                    else if (forwardHiddenLayers[i] is LSTMLayer)
                    {
                        var lstmForwardCell = new LSTMNeuron();
                        var lstmBackwardCell = new LSTMNeuron();

                        lstmForwardCell.LSTMCells = new LSTMCell[forwardHiddenLayers[i].LayerSize];
                        lstmBackwardCell.LSTMCells = new LSTMCell[forwardHiddenLayers[i].LayerSize];

                        for (int k = 0; k < forwardHiddenLayers[i].LayerSize; k++)
                        {
                            lstmForwardCell.LSTMCells[k] = new LSTMCell();
                            lstmBackwardCell.LSTMCells[k] = new LSTMCell();
                        }

                        forwardCells[j] = lstmForwardCell;
                        backwardCells[j] = lstmBackwardCell;


                    }
                    else
                    {
                        forwardCells[j] = new Neuron();
                        backwardCells[j] = new Neuron();
                    }

                    forwardCells[j].Cells = new float[forwardHiddenLayers[i].LayerSize];
                    forwardCells[j].PrevCellOutputs = new float[forwardHiddenLayers[i].LayerSize];

                    backwardCells[j].Cells = new float[forwardHiddenLayers[i].LayerSize];
                    backwardCells[j].PrevCellOutputs = new float[forwardHiddenLayers[i].LayerSize];

                    fErrLayer[j] = new float[forwardHiddenLayers[i].LayerSize];
                    bErrLayer[j] = new float[forwardHiddenLayers[i].LayerSize];
                }


                forwardCellList.Add(forwardCells);
                backwardCellList.Add(backwardCells);
                fErrLayers.Add(fErrLayer);
                bErrLayers.Add(bErrLayer);
            }

            InitOutputLayerCache();
        }

        public BiRNN()
        {
        }

        protected virtual void InitOutputLayerCache()
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
                SimpleLayer forwardLayer = forwardHiddenLayers[i];
                SimpleLayer backwardLayer = backwardHiddenLayers[i];

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
            SimpleLayer outputLayer = CreateOutputLayer(outputLayerConfig, TrainingSet.SparseFeatureSize, forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize * 2);
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

            BiRNN<T> rnn = new BiRNN<T>();
            rnn.InitCache(forwardLayers, backwardLayers, OutputLayer.CreateLayerSharedWegiths());
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



        private void ComputeMiddleLayers(Sequence pSequence, SimpleLayer forwardLayer, SimpleLayer backwardLayer, RunningMode runningMode, int layerIdx)
        {
            var numStates = pSequence.States.Length;
            float[][] lastLayerOutputs = layersOutput[layerIdx - 1];


            //Computing forward RNN
            forwardLayer.Reset();
            for (var curState = 0; curState < numStates; curState++)
            {
                var state = pSequence.States[curState];
                forwardLayer.ForwardPass(state.SparseFeature, lastLayerOutputs[curState]);
                forwardCellList[layerIdx][curState] = forwardLayer.CopyNeuronTo(forwardCellList[layerIdx][curState]);
            }

            //Computing backward RNN
            backwardLayer.Reset();
            for (var curState = numStates - 1; curState >= 0; curState--)
            {
                var state = pSequence.States[curState];
                backwardLayer.ForwardPass(state.SparseFeature, lastLayerOutputs[curState]);
                backwardCellList[layerIdx][curState] = backwardLayer.CopyNeuronTo(backwardCellList[layerIdx][curState]);
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
        private void ComputeBottomLayer(Sequence sequence, SimpleLayer forwardLayer, SimpleLayer backwardLayer, RunningMode runningMode)
        {
            var numStates = sequence.States.Length;

            //Computing forward RNN
            forwardLayer.Reset();
            for (var curState = 0; curState < numStates; curState++)
            {
                var state = sequence.States[curState];
                forwardLayer.ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());
                forwardCellList[0][curState] = forwardLayer.CopyNeuronTo(forwardCellList[0][curState]);
            }


            //Computing backward RNN
            backwardLayer.Reset();
            for (var curState = numStates - 1; curState >= 0; curState--)
            {
                var state = sequence.States[curState];
                backwardLayer.ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());
                backwardCellList[0][curState] = backwardLayer.CopyNeuronTo(backwardCellList[0][curState]);
            }



            //Merge forward and backward
            MergeForwardBackwardLayers(numStates, forwardLayer.LayerSize, 0);
        }

        private SimpleLayer[] ComputeTopLayer(Sequence pSequence, out Matrix<float> rawOutputLayer, RunningMode runningMode, bool outputRawScore)
        {
            var numStates = pSequence.States.Length;
            var lastLayerOutputs = layersOutput[forwardHiddenLayers.Count - 1];

            //Calculate output layer
            Matrix<float> tmpOutputResult = null;
            if (outputRawScore)
            {
                tmpOutputResult = new Matrix<float>(numStates, OutputLayer.LayerSize);
            }

            var labelSet = pSequence.States.Select(state => state.Label).ToList();

            //Initialize output layer or reallocate it
            if (seqFinalOutput == null || seqFinalOutput.Length < numStates)
            {
                seqFinalOutput = new SimpleLayer[numStates];
                for (var i = 0; i < numStates; i++)
                {
                    seqFinalOutput.SetValue(Activator.CreateInstance(OutputLayer.GetType(), OutputLayer.LayerConfig), i);
                    OutputLayer.ShallowCopyWeightTo(seqFinalOutput[i]);
                }
            }


            for (var curState = 0; curState < numStates; curState++)
            {
                var state = pSequence.States[curState];
                var outputCells = seqFinalOutput[curState];
                outputCells.LabelShortList = labelSet;
                outputCells.ForwardPass(state.SparseFeature, lastLayerOutputs[curState]);

                if (outputRawScore)
                {
                    outputCells.Cells.CopyTo(tmpOutputResult[curState], 0);
                }
            }

            rawOutputLayer = tmpOutputResult;
            return seqFinalOutput;
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
        private SimpleLayer[] ComputeLayers(Sequence pSequence, RunningMode runningMode, out Matrix<float> rawOutputLayer, bool outputRawScore)
        {
            ComputeBottomLayer(pSequence, forwardHiddenLayers[0], backwardHiddenLayers[0], runningMode);
            for (var i = 1; i < forwardHiddenLayers.Count; i++)
            {
                ComputeMiddleLayers(pSequence, forwardHiddenLayers[i], backwardHiddenLayers[i], runningMode, i);
            }

            return ComputeTopLayer(pSequence, out rawOutputLayer, runningMode, outputRawScore);
        }

        /// <summary>
        ///     Pass error from the last layer to the first layer
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="seqFinalOutput"></param>
        /// <returns></returns>
        private void ComputeDeepErr(Sequence pSequence, SimpleLayer[] seqFinalOutput)
        {
            var numStates = pSequence.States.Length;
            var numLayers = forwardHiddenLayers.Count;

            //Calculate output layer error


            for (var curState = 0; curState < numStates; curState++)
            {
                var layer = seqFinalOutput[curState];
                layer.ComputeLayerErr(CRFSeqOutput, pSequence.States[curState], curState);
            }

            //Now we already have err in output layer, let's pass them back to other layers

            //Pass error from i+1 to i layer
            var forwardLayer = forwardHiddenLayers[numLayers - 1];
            var backwardLayer = backwardHiddenLayers[numLayers - 1];

            var errLayer1 = fErrLayers[numLayers - 1];
            var errLayer2 = bErrLayers[numLayers - 1];


            for (var curState = 0; curState < numStates; curState++)
            {
                var curState2 = numStates - curState - 1;
                forwardLayer.ComputeLayerErr(seqFinalOutput[curState2], errLayer1[curState2], seqFinalOutput[curState2].Errs);
            }


            for (var curState = 0; curState < numStates; curState++)
            {
                backwardLayer.ComputeLayerErr(seqFinalOutput[curState], errLayer2[curState], seqFinalOutput[curState].Errs);
            }



            // Forward
            for (var i = numLayers - 2; i >= 0; i--)
            {
                forwardLayer = forwardHiddenLayers[i];
                var lastForwardLayer = forwardHiddenLayers[i + 1];
                var errLayer = fErrLayers[i];
                var srcErrLayer = fErrLayers[i + 1];

                for (var curState = 0; curState < numStates; curState++)
                {
                    var curState2 = numStates - curState - 1;
                    forwardLayer.ComputeLayerErr(lastForwardLayer, errLayer[curState2], srcErrLayer[curState2]);
                }
            }

            // Backward
            for (var i = numLayers - 2; i >= 0; i--)
            {
                backwardLayer = backwardHiddenLayers[i];
                var lastBackwardLayer = backwardHiddenLayers[i + 1];
                var errLayer = bErrLayers[i];
                var srcErrLayer = bErrLayers[i + 1];


                for (var curState = 0; curState < numStates; curState++)
                {
                    backwardLayer.ComputeLayerErr(lastBackwardLayer, errLayer[curState], srcErrLayer[curState]);
                }
            }

        }

        private void DeepLearningNet(Sequence pSequence, SimpleLayer[] seqOutput)
        {
            var numStates = pSequence.States.Length;
            var numLayers = forwardHiddenLayers.Count;


            //Learning output layer
            for (var curState = 0; curState < numStates; curState++)
            {
                seqOutput[curState].BackwardPass();
            }



            for (var i = 0; i < numLayers; i++)
            {
                float[][] layerOutputs_i = (i > 0) ? layersOutput[i - 1] : null;
                Neuron[] forwardNeurons = forwardCellList[i];
                float[][] forwardErrs = fErrLayers[i];
                var forwardLayer = forwardHiddenLayers[i];
                for (var curState = 0; curState < numStates; curState++)
                {
                    State state = pSequence.States[curState];

                    forwardLayer.SparseFeatureGroups = new List<SparseVector>();
                    forwardLayer.SparseFeatureGroups.Add(state.SparseFeature);
                    forwardLayer.DenseFeatureGroups = new List<float[]>();
                    forwardLayer.DenseFeatureGroups.Add((i == 0) ? state.DenseFeature.CopyTo() : layerOutputs_i[curState]);
                    forwardLayer.PreUpdateWeights(forwardNeurons[curState], forwardErrs[curState]);
                    forwardLayer.BackwardPass();
                }

                Neuron[] backwardNeurons = backwardCellList[i];
                float[][] backwardErrs = bErrLayers[i];
                var backwardLayer = backwardHiddenLayers[i];
                for (var curState = 0; curState < numStates; curState++)
                {
                    var curState2 = numStates - curState - 1;
                    State state = pSequence.States[curState2];

                    backwardLayer.SparseFeatureGroups = new List<SparseVector>();
                    backwardLayer.SparseFeatureGroups.Add(state.SparseFeature);
                    backwardLayer.DenseFeatureGroups = new List<float[]>();
                    backwardLayer.DenseFeatureGroups.Add((i == 0) ? state.DenseFeature.CopyTo() : layerOutputs_i[curState2]);
                    backwardLayer.PreUpdateWeights(backwardNeurons[curState2], backwardErrs[curState2]);
                    backwardLayer.BackwardPass();
                }
            }
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
            var seqOutput = ComputeLayers(pSequence, runningMode, out rawOutputLayer, outputRawScore);

            //Get best output result of each state
            var numStates = pSequence.States.Length;
            seqBestOutput = new int[numStates];

            for (var curState = 0; curState < numStates; curState++)
            {
                seqBestOutput[curState] = seqOutput[curState].GetBestOutputIndex();
            }

            if (runningMode == RunningMode.Training)
            {
                //In training mode, we calculate each layer's error and update their net weights
                ComputeDeepErr(pSequence, seqOutput);
                DeepLearningNet(pSequence, seqOutput);
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

            var seqOutput = ComputeLayers(pSequence, runningMode, out rawOutputLayer, true);

            ForwardBackward(numStates, rawOutputLayer);

            var predict = Viterbi(rawOutputLayer, numStates);

            if (runningMode == RunningMode.Training)
            {
                UpdateBigramTransition(pSequence);
                ComputeDeepErr(pSequence, seqOutput);
                DeepLearningNet(pSequence, seqOutput);
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
                forwardHiddenLayers = new List<SimpleLayer>();
                for (var i = 0; i < layerSize; i++)
                {
                    layerType = (LayerType)br.ReadInt32();
                    forwardHiddenLayers.Add(Load(layerType, br));

                    SimpleLayer layer = forwardHiddenLayers[forwardHiddenLayers.Count - 1];
                    if (bTrain)
                    {
                        layer.SetRunningMode(RunningMode.Training);
                        layer.InitializeInternalTrainingParameters();

                    }
                    else
                    {
                        layer.SetRunningMode(RunningMode.Test);
                    }
                }

                //Load backward layers from file
                backwardHiddenLayers = new List<SimpleLayer>();
                for (var i = 0; i < layerSize; i++)
                {
                    layerType = (LayerType)br.ReadInt32();
                    backwardHiddenLayers.Add(Load(layerType, br));

                    SimpleLayer layer = backwardHiddenLayers[backwardHiddenLayers.Count - 1];
                    if (bTrain)
                    {
                        layer.SetRunningMode(RunningMode.Training);
                        layer.InitializeInternalTrainingParameters();
                    }
                    else
                    {
                        layer.SetRunningMode(RunningMode.Test);
                    }
                }

                Logger.WriteLine("Create output layer");
                layerType = (LayerType)br.ReadInt32();
                OutputLayer = Load(layerType, br);

                if (bTrain)
                {
                    OutputLayer.SetRunningMode(RunningMode.Training);
                    OutputLayer.InitializeInternalTrainingParameters();
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
                    InitCache(forwardHiddenLayers, backwardHiddenLayers, OutputLayer.CreateLayerSharedWegiths());
                }
            }
        }
    }
}