using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using AdvUtils;
using System.Numerics;
using System.Runtime.CompilerServices;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public enum MODELDIRECTION
    {
        FORWARD = 0,
        BI_DIRECTIONAL
    }

    public enum RunningMode
    {
        Training = 0,
        Validate = 1,
        Test = 2
    }

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

    public class ForwardRNN : RNN
    {        
        public List<SimpleLayer> HiddenLayerList { get; set; }
        
        public ForwardRNN(List<SimpleLayer> hiddenLayerList, SimpleLayer outputLayer)
        {
            HiddenLayerList = hiddenLayerList;
            OutputLayer = outputLayer;
        }

        public ForwardRNN()
        {

        }

        public override List<double[]> ComputeTopHiddenLayerOutput(Sequence pSequence)
        {
            int numStates = pSequence.States.Length;
            int numLayers = HiddenLayerList.Count;

            //reset all layers
            foreach (SimpleLayer layer in HiddenLayerList)
            {
                layer.netReset(false);
            }

            List<double[]> outputs = new List<double[]>();
            for (int curState = 0; curState < numStates; curState++)
            {
                //Compute first layer
                State state = pSequence.States[curState];
                SetInputLayer(state, curState, numStates, null);
                HiddenLayerList[0].computeLayer(state.SparseData, state.DenseData.CopyTo(), false);

                //Compute each layer
                for (int i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    HiddenLayerList[i].computeLayer(state.SparseData, HiddenLayerList[i - 1].cellOutput, false);
                }

                double[] tmpOutput = new double[HiddenLayerList[numLayers - 1].cellOutput.Length];
                for (int i = 0; i < HiddenLayerList[numLayers - 1].cellOutput.Length; i++)
                {
                    tmpOutput[i] = HiddenLayerList[numLayers - 1].cellOutput[i];
                }
                outputs.Add(tmpOutput);
            }

            return outputs;
        }

        public override int[] ProcessSequence(Sequence pSequence, RunningMode runningMode, bool outputRawScore, out Matrix<double> m)
        {
            int numStates = pSequence.States.Length;
            int numLayers = HiddenLayerList.Count;

            if (outputRawScore == true)
            {
                m = new Matrix<double>(numStates, OutputLayer.LayerSize);
            }
            else
            {
                m = null;
            }

            int[] predicted = new int[numStates];
            bool isTraining = true;
            if (runningMode == RunningMode.Training)
            {
                isTraining = true;
            }
            else
            {
                isTraining = false;
            }

            //reset all layers
            foreach (SimpleLayer layer in HiddenLayerList)
            {
                layer.netReset(isTraining);
            }

            for (int curState = 0; curState < numStates; curState++)
            {
                //Compute first layer
                State state = pSequence.States[curState];
                SetInputLayer(state, curState, numStates, predicted);
                HiddenLayerList[0].computeLayer(state.SparseData, state.DenseData.CopyTo(), isTraining);

                //Compute each layer
                for (int i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    HiddenLayerList[i].computeLayer(state.SparseData, HiddenLayerList[i - 1].cellOutput, isTraining);
                }

                //Compute output layer
                OutputLayer.CurrentLabelId = state.Label;
                OutputLayer.computeLayer(state.SparseData, HiddenLayerList[numLayers - 1].cellOutput, isTraining);

                if (m != null)
                {
                    OutputLayer.cellOutput.CopyTo(m[curState], 0);
                }

                OutputLayer.Softmax(isTraining);

                predicted[curState] = OutputLayer.GetBestOutputIndex(isTraining);

                if (runningMode != RunningMode.Test)
                {
                    logp += Math.Log10(OutputLayer.cellOutput[state.Label] + 0.0001);
                }

                if (runningMode == RunningMode.Training)
                {
                    // error propogation
                    OutputLayer.ComputeLayerErr(CRFSeqOutput, state, curState);

                    //propogate errors to each layer from output layer to input layer
                    HiddenLayerList[numLayers - 1].ComputeLayerErr(OutputLayer);
                    for (int i = numLayers - 2; i >= 0; i--)
                    {
                        HiddenLayerList[i].ComputeLayerErr(HiddenLayerList[i + 1]);
                    }

                    //Update net weights
                    Parallel.Invoke(() =>
                    {
                        OutputLayer.LearnFeatureWeights(numStates, curState);
                    },
                    () =>
                    {
                        Parallel.For(0, numLayers, parallelOption, i =>
                        {
                            HiddenLayerList[i].LearnFeatureWeights(numStates, curState);
                        });
                    });
                }
            }

            return predicted;
        }

        public override int[] ProcessSequenceCRF(Sequence pSequence, RunningMode runningMode)
        {
            int numStates = pSequence.States.Length;
            int numLayers = HiddenLayerList.Count;

            //Get network output without CRF
            Matrix<double> nnOutput;
            ProcessSequence(pSequence, RunningMode.Test, true, out nnOutput);

            //Compute CRF result
            ForwardBackward(numStates, nnOutput);

            if (runningMode != RunningMode.Test)
            {
                //Get the best result
                for (int i = 0; i < numStates; i++)
                {
                    logp += Math.Log10(CRFSeqOutput[i][pSequence.States[i].Label] + 0.0001);
                }
            }

            //Compute best path in CRF result
            int[] predicted = Viterbi(nnOutput, numStates);

            if (runningMode == RunningMode.Training)
            {
                //Update tag bigram transition for CRF model
                UpdateBigramTransition(pSequence);

                //Reset all layer states
                foreach (SimpleLayer layer in HiddenLayerList)
                {
                    layer.netReset(true);
                }

                for (int curState = 0; curState < numStates; curState++)
                {
                    // error propogation
                    State state = pSequence.States[curState];
                    SetInputLayer(state, curState, numStates, null);
                    HiddenLayerList[0].computeLayer(state.SparseData, state.DenseData.CopyTo());

                    for (int i = 1; i < numLayers; i++)
                    {
                        HiddenLayerList[i].computeLayer(state.SparseData, HiddenLayerList[i - 1].cellOutput);
                    }

                    OutputLayer.ComputeLayerErr(CRFSeqOutput, state, curState);

                    HiddenLayerList[numLayers - 1].ComputeLayerErr(OutputLayer);
                    for (int i = numLayers - 2; i >= 0; i--)
                    {
                        HiddenLayerList[i].ComputeLayerErr(HiddenLayerList[i + 1]);
                    }

                    //Update net weights
                    Parallel.Invoke(() =>
                    {
                        OutputLayer.LearnFeatureWeights(numStates, curState);
                    },
                    () =>
                    {
                        Parallel.For(0, numLayers, parallelOption, i =>
                        {
                            HiddenLayerList[i].LearnFeatureWeights(numStates, curState);
                        });
                    });
                }
            }

            return predicted;
        }

        // save model as binary format
        public override void SaveModel(string filename)
        {
            StreamWriter sw = new StreamWriter(filename);
            BinaryWriter fo = new BinaryWriter(sw.BaseStream);

            if (HiddenLayerList[0] is BPTTLayer)
            {
                fo.Write(0);
            }
            else
            {
                fo.Write(1);
            }
            fo.Write((int)ModelDirection);
            fo.Write(IsCRFTraining);

            fo.Write(HiddenLayerList.Count);
            foreach (SimpleLayer layer in HiddenLayerList)
            {
                layer.Save(fo);
            }
            OutputLayer.Save(fo);

            if (IsCRFTraining == true)
            {
                //Save CRF feature weights
                RNNHelper.SaveMatrix(CRFTagTransWeights, fo);
            }

            fo.Close();
        }

        public override void LoadModel(string filename)
        {
            Logger.WriteLine("Loading SimpleRNN model: {0}", filename);

            StreamReader sr = new StreamReader(filename);
            BinaryReader br = new BinaryReader(sr.BaseStream);

            int modelType = br.ReadInt32();
            ModelDirection = (MODELDIRECTION)br.ReadInt32();
            IsCRFTraining = br.ReadBoolean();

            //Create cells of each layer
            int layerSize = br.ReadInt32();
            HiddenLayerList = new List<SimpleLayer>();
            for (int i = 0; i < layerSize; i++)
            {
                SimpleLayer layer = null;
                if (modelType == 0)
                {
                    layer = new BPTTLayer();
                }
                else
                {
                    layer = new LSTMLayer();
                }

                layer.Load(br);
                HiddenLayerList.Add(layer);
            }

            OutputLayer = new SimpleLayer();
            OutputLayer.Load(br);

            if (IsCRFTraining == true)
            {
                Logger.WriteLine("Loading CRF tag trans weights...");
                CRFTagTransWeights = RNNHelper.LoadMatrix(br);
            }

            sr.Close();
        }


        public override void CleanStatus()
        {
            foreach (SimpleLayer layer in HiddenLayerList)
            {
                layer.CleanLearningRate();
            }

            OutputLayer.CleanLearningRate();
        }    
    }
}
