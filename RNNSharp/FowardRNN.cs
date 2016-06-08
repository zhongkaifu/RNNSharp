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
        Train = 0,
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

        public override Matrix<double> ProcessSequence(Sequence pSequence, RunningMode runningMode)
        {
            int numStates = pSequence.States.Length;
            int numLayers = HiddenLayerList.Count;
            Matrix<double> m = new Matrix<double>(numStates, OutputLayer.LayerSize);
            int[] predicted = new int[numStates];
            bool isTraining = true;
            if (runningMode == RunningMode.Train)
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
                OutputLayer.computeLayer(state.SparseData, HiddenLayerList[numLayers - 1].cellOutput, isTraining);
                OutputLayer.cellOutput.CopyTo(m[curState], 0);
                OutputLayer.Softmax();

                predicted[curState] = GetBestOutputIndex();

                if (runningMode != RunningMode.Test)
                {
                    logp += Math.Log10(OutputLayer.cellOutput[state.Label] + 0.0001);
                }

                if (runningMode == RunningMode.Train)
                {
                    // error propogation
                    ComputeOutputLayerErr(OutputLayer, state, curState);

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

            return m;
        }

        public int GetBestOutputIndex()
        {
            int imax = 0;
            double dmax = OutputLayer.cellOutput[0];
            for (int k = 1; k < OutputLayer.LayerSize; k++)
            {
                if (OutputLayer.cellOutput[k] > dmax)
                {
                    dmax = OutputLayer.cellOutput[k];
                    imax = k;
                }
            }
            return imax;
        }

        public override int[] ProcessSequenceCRF(Sequence pSequence, RunningMode runningMode)
        {
            int numStates = pSequence.States.Length;
            int numLayers = HiddenLayerList.Count;

            //Get network output without CRF
            Matrix<double> nnOutput = ProcessSequence(pSequence, RunningMode.Test);

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

            if (runningMode == RunningMode.Train)
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

                    ComputeOutputLayerErr(OutputLayer, state, curState);

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

            // Signiture , 0 is for RNN or 1 is for RNN-CRF
            int iflag = 0;
            if (IsCRFTraining == true)
            {
                iflag = 1;
            }
            fo.Write(iflag);

            fo.Write(HiddenLayerList.Count);
            foreach (SimpleLayer layer in HiddenLayerList)
            {
                layer.Save(fo);
            }
            OutputLayer.Save(fo);

            if (iflag == 1)
            {
                // Save Bigram
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

            int iflag = br.ReadInt32();
            if (iflag == 1)
            {
                IsCRFTraining = true;
            }
            else
            {
                IsCRFTraining = false;
            }

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

            if (iflag == 1)
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

        public void ComputeOutputLayerErr(SimpleLayer outputLayer, State state, int timeat)
        {
            if (IsCRFTraining == true)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                for (int c = 0; c < outputLayer.LayerSize; c++)
                {
                    outputLayer.er[c] = -CRFSeqOutput[timeat][c];
                }
                outputLayer.er[state.Label] = 1 - CRFSeqOutput[timeat][state.Label];
            }
            else
            {
                //For standard RNN
                for (int c = 0; c < outputLayer.LayerSize; c++)
                {
                    outputLayer.er[c] = -outputLayer.cellOutput[c];
                }
                outputLayer.er[state.Label] = 1 - outputLayer.cellOutput[state.Label];
            }

        }

            
    }
}
