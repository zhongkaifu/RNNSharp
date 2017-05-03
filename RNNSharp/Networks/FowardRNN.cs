using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>

namespace RNNSharp.Networks
{
    public class PAIR<T, K>
    {
        public T first;
        public K second;

        public PAIR(T f, K s)
        {
            first = f;
            second = s;
        }
    }

    public class ForwardRNN<T> : RNN<T> where T : ISequence
    {
        public ForwardRNN()
        {
            
        }

        public override void CreateNetwork(List<LayerConfig> hiddenLayersConfig, LayerConfig outputLayerConfig, DataSet<T> TrainingSet, Config featurizer)
        {
            HiddenLayerList = CreateLayers(hiddenLayersConfig);
            for (var i = 0; i < HiddenLayerList.Count; i++)
            {
                SimpleLayer layer = HiddenLayerList[i];
                layer.InitializeWeights(TrainingSet.SparseFeatureSize, i == 0 ? TrainingSet.DenseFeatureSize : HiddenLayerList[i - 1].LayerSize);
                layer.SetRunningMode(RunningMode.Training);

                Logger.WriteLine($"Create hidden layer {i}: size = {layer.LayerSize}, sparse feature size = {layer.SparseFeatureSize}, dense feature size = {layer.DenseFeatureSize}");
            }

            outputLayerConfig.LayerSize = TrainingSet.TagSize;
            OutputLayer = CreateOutputLayer(outputLayerConfig, TrainingSet.SparseFeatureSize, HiddenLayerList[HiddenLayerList.Count - 1].LayerSize);
            OutputLayer.SetRunningMode(RunningMode.Training);

            Logger.WriteLine($"Create a Forward recurrent neural network with {HiddenLayerList.Count} hidden layers");
        }

        public List<SimpleLayer> HiddenLayerList { get; set; }

        public override void UpdateWeights()
        {
            foreach (var layer in HiddenLayerList)
            {
                layer.UpdateWeights();
            }

            OutputLayer.UpdateWeights();
        }

        public override RNN<T> Clone()
        {
            List<SimpleLayer> hiddenLayers = new List<SimpleLayer>();

            foreach (SimpleLayer layer in HiddenLayerList)
            {
                hiddenLayers.Add(layer.CreateLayerSharedWegiths());
            }

            ForwardRNN<T> rnn = new ForwardRNN<T>();
            rnn.HiddenLayerList = hiddenLayers;
            rnn.OutputLayer = OutputLayer.CreateLayerSharedWegiths();
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

        public override int GetTopHiddenLayerSize()
        {
            return HiddenLayerList[HiddenLayerList.Count - 1].LayerSize;
        }

        public override float[][] ComputeTopHiddenLayerOutput(Sequence pSequence)
        {
            var numStates = pSequence.States.Length;
            var numLayers = HiddenLayerList.Count;

            //reset all layers
            foreach (var layer in HiddenLayerList)
            {
                layer.Reset();
            }

            var outputs = new float[numStates][];
            for (var curState = 0; curState < numStates; curState++)
            {
                //Compute first layer
                var state = pSequence.States[curState];
                SetRuntimeFeatures(state, curState, numStates, null);
                HiddenLayerList[0].ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());

                //Compute each layer
                for (var i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    HiddenLayerList[i].ForwardPass(state.SparseFeature, HiddenLayerList[i - 1].Cells);
                }

                var tmpOutput = new float[HiddenLayerList[numLayers - 1].Cells.Length];
                for (var i = 0; i < HiddenLayerList[numLayers - 1].Cells.Length; i++)
                {
                    tmpOutput[i] = HiddenLayerList[numLayers - 1].Cells[i];
                }
                outputs[curState] = tmpOutput;
            }

            return outputs;
        }

        public override int[] ProcessSequence(ISentence sentence, Config featurizer, RunningMode runningMode, bool outputRawScore, out Matrix<float> m)
        {
            var seq = featurizer.BuildSequence(sentence as Sentence);

            return ProcessSequence(seq, runningMode, outputRawScore, out m);
        }


        public override int[] ProcessSequence(ISequence sequence, RunningMode runningMode, bool outputRawScore, out Matrix<float> m)
        {
            Sequence pSequence = sequence as Sequence;

            var numStates = pSequence.States.Length;
            var numLayers = HiddenLayerList.Count;

            m = outputRawScore ? new Matrix<float>(numStates, OutputLayer.LayerSize) : null;

            var predicted = new int[numStates];
            var isTraining = runningMode == RunningMode.Training;

            //reset all layers
            foreach (var layer in HiddenLayerList)
            {
                layer.Reset();
            }

            //Set current sentence labels into short list in output layer
            OutputLayer.LabelShortList.Clear();
            foreach (var state in pSequence.States)
            {
                OutputLayer.LabelShortList.Add(state.Label);
            }

            for (var curState = 0; curState < numStates; curState++)
            {
                //Compute first layer
                var state = pSequence.States[curState];
                SetRuntimeFeatures(state, curState, numStates, predicted);
                HiddenLayerList[0].ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());

                //Compute each layer
                for (var i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    HiddenLayerList[i].ForwardPass(state.SparseFeature, HiddenLayerList[i - 1].Cells);
                }

                //Compute output layer
                OutputLayer.ForwardPass(state.SparseFeature, HiddenLayerList[numLayers - 1].Cells);

                if (m != null)
                {
                    OutputLayer.Cells.CopyTo(m[curState], 0);
                }

                predicted[curState] = OutputLayer.GetBestOutputIndex();

                if (runningMode == RunningMode.Training)
                {
                    // error propogation
                    OutputLayer.ComputeLayerErr(CRFSeqOutput, state, curState);

                    //propogate errors to each layer from output layer to input layer
                    HiddenLayerList[numLayers - 1].ComputeLayerErr(OutputLayer);
                    for (var i = numLayers - 2; i >= 0; i--)
                    {
                        HiddenLayerList[i].ComputeLayerErr(HiddenLayerList[i + 1]);
                    }

                    //Update net weights
                    OutputLayer.BackwardPass();

                    for (var i = 0; i < numLayers; i++)
                    { HiddenLayerList[i].BackwardPass(); }
                }
            }

            return predicted;
        }

        public override int[] ProcessSequenceCRF(Sequence pSequence, RunningMode runningMode)
        {
            var numStates = pSequence.States.Length;
            var numLayers = HiddenLayerList.Count;

            //Get network output without CRF
            Matrix<float> nnOutput;
            ProcessSequence(pSequence, RunningMode.Test, true, out nnOutput);

            //Compute CRF result
            ForwardBackward(numStates, nnOutput);

            //Compute best path in CRF result
            var predicted = Viterbi(nnOutput, numStates);

            if (runningMode == RunningMode.Training)
            {
                //Update tag bigram transition for CRF model
                UpdateBigramTransition(pSequence);

                //Reset all layer states
                foreach (var layer in HiddenLayerList)
                {
                    layer.Reset();
                }

                for (var curState = 0; curState < numStates; curState++)
                {
                    // error propogation
                    var state = pSequence.States[curState];
                    SetRuntimeFeatures(state, curState, numStates, null);
                    HiddenLayerList[0].SetRunningMode(runningMode);
                    HiddenLayerList[0].ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());

                    for (var i = 1; i < numLayers; i++)
                    {
                        HiddenLayerList[i].SetRunningMode(runningMode);
                        HiddenLayerList[i].ForwardPass(state.SparseFeature, HiddenLayerList[i - 1].Cells);
                    }

                    OutputLayer.ComputeLayerErr(CRFSeqOutput, state, curState);

                    HiddenLayerList[numLayers - 1].ComputeLayerErr(OutputLayer);
                    for (var i = numLayers - 2; i >= 0; i--)
                    {
                        HiddenLayerList[i].ComputeLayerErr(HiddenLayerList[i + 1]);
                    }

                    //Update net weights
                     OutputLayer.BackwardPass();

                    for (var i = 0; i < numLayers; i++)
                    { HiddenLayerList[i].BackwardPass(); }
 
                }
            }

            return predicted;
        }

        // save model as binary format
        public override void SaveModel(string filename)
        {
            var sw = new StreamWriter(filename);
            var fo = new BinaryWriter(sw.BaseStream);

            fo.Write(IsCRFTraining);
            fo.Write(HiddenLayerList.Count);
            foreach (var layer in HiddenLayerList)
            {
                fo.Write((int)layer.LayerType);
                layer.Save(fo);
            }

            fo.Write((int)OutputLayer.LayerType);
            OutputLayer.Save(fo);

            if (IsCRFTraining)
            {
                //Save CRF feature weights
                RNNHelper.SaveMatrix(CRFWeights, fo);
            }

            fo.Close();
        }

        public override void LoadModel(string filename, bool bTrain = false)
        {
            Logger.WriteLine("Loading SimpleRNN model: {0}", filename);

            var sr = new StreamReader(filename);
            var br = new BinaryReader(sr.BaseStream);

            IsCRFTraining = br.ReadBoolean();

            //Create cells of each layer
            var layerSize = br.ReadInt32();
            LayerType layerType = LayerType.None;
            HiddenLayerList = new List<SimpleLayer>();
            for (var i = 0; i < layerSize; i++)
            {
                layerType = (LayerType)br.ReadInt32();
                HiddenLayerList.Add(Load(layerType, br));

                SimpleLayer layer = HiddenLayerList[HiddenLayerList.Count - 1];
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

            sr.Close();
        }

        public override void CleanStatusForTraining()
        {
            foreach (var layer in HiddenLayerList)
            {
                layer.CleanForTraining();
            }

            OutputLayer.CleanForTraining();
        }
    }
}