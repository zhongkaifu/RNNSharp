using System;
using System.IO;
using System.Threading.Tasks;
using AdvUtils;
using System.Collections.Generic;
using System.Numerics;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    class BiRNN : RNN
    {
        private Vector<double> vecConst2 = new Vector<double>(2.0f);

        List<SimpleLayer> forwardHiddenLayers = new List<SimpleLayer>();
        List<SimpleLayer> backwardHiddenLayers = new List<SimpleLayer>();

        public BiRNN(List<SimpleLayer> s_forwardRNN, List<SimpleLayer> s_backwardRNN, SimpleLayer outputLayer)
        {
            forwardHiddenLayers = s_forwardRNN;
            backwardHiddenLayers = s_backwardRNN;

            //Initialize output layer
            OutputLayer = outputLayer;
        }

        public BiRNN()
        {

        }

        public override void CleanStatus()
        {
            foreach (SimpleLayer layer in forwardHiddenLayers)
            {
                layer.CleanLearningRate();
            }

            foreach (SimpleLayer layer in backwardHiddenLayers)
            {
                layer.CleanLearningRate();
            }

            OutputLayer.CleanLearningRate();

            RNNHelper.vecMaxGrad = new Vector<double>(RNNHelper.GradientCutoff);
            RNNHelper.vecMinGrad = new Vector<double>(-RNNHelper.GradientCutoff);
            RNNHelper.vecNormalLearningRate = new Vector<double>(RNNHelper.LearningRate);
        }

        private SimpleLayer[] ComputeMiddleLayers(Sequence pSequence, SimpleLayer[] lastLayers, SimpleLayer forwardLayer, SimpleLayer backwardLayer)
        {
            int numStates = lastLayers.Length;

            SimpleLayer[] mForward = null;
            SimpleLayer[] mBackward = null;
            Parallel.Invoke(() =>
            {
                //Computing forward RNN      
                forwardLayer.netReset(false);
                mForward = new SimpleLayer[lastLayers.Length];
                for (int curState = 0; curState < lastLayers.Length; curState++)
                {
                    State state = pSequence.States[curState];
                    forwardLayer.computeLayer(state.SparseData, lastLayers[curState].cellOutput);
                    mForward[curState] = forwardLayer.GetHiddenLayer();
                }
            },
             () =>
             {
                 //Computing backward RNN
                 backwardLayer.netReset(false);
                 mBackward = new SimpleLayer[lastLayers.Length];
                 for (int curState = lastLayers.Length - 1; curState >= 0; curState--)
                 {
                     State state = pSequence.States[curState];
                     backwardLayer.computeLayer(state.SparseData, lastLayers[curState].cellOutput);
                     mBackward[curState] = backwardLayer.GetHiddenLayer();
                 }
             });

            //Merge forward and backward
            SimpleLayer[] mergedLayer = new SimpleLayer[numStates];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                State state = pSequence.States[curState];
                mergedLayer[curState] = new SimpleLayer(forwardLayer.LayerSize);
                mergedLayer[curState].SparseFeature = state.SparseData;
                mergedLayer[curState].DenseFeature = lastLayers[curState].cellOutput;

                SimpleLayer forwardCells = mForward[curState];
                SimpleLayer backwardCells = mBackward[curState];

                int i = 0;
                while (i < forwardLayer.LayerSize - Vector<double>.Count)
                {
                    Vector<double> v1 = new Vector<double>(forwardCells.cellOutput, i);
                    Vector<double> v2 = new Vector<double>(backwardCells.cellOutput, i);
                    Vector<double> v = (v1 + v2) / vecConst2;

                    v.CopyTo(mergedLayer[curState].cellOutput, i);

                    i += Vector<float>.Count;
                }

                while (i < forwardLayer.LayerSize)
                {
                    mergedLayer[curState].cellOutput[i] = (forwardCells.cellOutput[i] + backwardCells.cellOutput[i]) / 2.0;
                    i++;
                }
            });

            return mergedLayer;
        }

        /// <summary>
        /// Compute the output of bottom layer
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="forwardLayer"></param>
        /// <param name="backwardLayer"></param>
        /// <returns></returns>
        private SimpleLayer[] ComputeBottomLayer(Sequence pSequence, SimpleLayer forwardLayer, SimpleLayer backwardLayer)
        {
            int numStates = pSequence.States.Length;
            SimpleLayer[] mForward = null;
            SimpleLayer[] mBackward = null;
            Parallel.Invoke(() =>
            {
                //Computing forward RNN      
                forwardLayer.netReset(false);
                mForward = new SimpleLayer[numStates];
                for (int curState = 0; curState < numStates; curState++)
                {
                    State state = pSequence.States[curState];
                    SetInputLayer(state, curState, numStates, null);
                    forwardLayer.computeLayer(state.SparseData, state.DenseData.CopyTo());
                    mForward[curState] = forwardLayer.GetHiddenLayer();
                }
            },
             () =>
             {
                 //Computing backward RNN
                 backwardLayer.netReset(false);
                 mBackward = new SimpleLayer[numStates];
                 for (int curState = numStates - 1; curState >= 0; curState--)
                 {
                     State state = pSequence.States[curState];
                     SetInputLayer(state, curState, numStates, null, false);
                     backwardLayer.computeLayer(state.SparseData, state.DenseData.CopyTo());      //compute probability distribution

                     mBackward[curState] = backwardLayer.GetHiddenLayer();
                 }
             });

            SimpleLayer[] mergedLayer = new SimpleLayer[numStates];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                State state = pSequence.States[curState];
                mergedLayer[curState] = new SimpleLayer(forwardLayer.LayerSize);
                mergedLayer[curState].SparseFeature = state.SparseData;
                mergedLayer[curState].DenseFeature = state.DenseData.CopyTo();

                SimpleLayer forwardCells = mForward[curState];
                SimpleLayer backwardCells = mBackward[curState];

                int i = 0;
                while (i < forwardLayer.LayerSize - Vector<double>.Count)
                {
                    Vector<double> v1 = new Vector<double>(forwardCells.cellOutput, i);
                    Vector<double> v2 = new Vector<double>(backwardCells.cellOutput, i);
                    Vector<double> v = (v1 + v2) / vecConst2;

                    v.CopyTo(mergedLayer[curState].cellOutput, i);

                    i += Vector<float>.Count;
                }

                while (i < forwardLayer.LayerSize)
                {
                    mergedLayer[curState].cellOutput[i] = (forwardCells.cellOutput[i] + backwardCells.cellOutput[i]) / 2.0;
                    i++;
                }
            });

            return mergedLayer;
        }

        Array seqFinalOutput = null;
        private SimpleLayer[] ComputeTopLayer(Sequence pSequence, SimpleLayer[] lastLayer, out Matrix<double> rawOutputLayer, bool isTraining, bool outputRawScore, out int[] seqBestOutput)
        {
            int numStates = lastLayer.Length;
            seqBestOutput = new int[numStates];

            //Calculate output layer
            Matrix<double> tmp_rawOutputLayer = null;
            if (outputRawScore == true)
            {
                tmp_rawOutputLayer = new Matrix<double>(numStates, OutputLayer.LayerSize);
            }

            //Initialize output layer or reallocate it
            if (seqFinalOutput == null || seqFinalOutput.Length < numStates)
            {
                seqFinalOutput = Array.CreateInstance(OutputLayer.GetType(), numStates);
                for (int i = 0; i < numStates; i++)
                {
                    seqFinalOutput.SetValue(Activator.CreateInstance(OutputLayer.GetType(), OutputLayer.LayerSize), i);
                    OutputLayer.ShallowCopyWeightTo((SimpleLayer)seqFinalOutput.GetValue(i));
                }
            }

            Parallel.For(0, numStates, parallelOption, curState =>
            {
                State state = pSequence.States[curState];
                var outputCells = (SimpleLayer)seqFinalOutput.GetValue(curState);
                outputCells.CurrentLabelId = state.Label;
                outputCells.computeLayer(state.SparseData, lastLayer[curState].cellOutput, isTraining);

                if (outputRawScore == true)
                {
                    outputCells.cellOutput.CopyTo(tmp_rawOutputLayer[curState], 0);
                }
                outputCells.Softmax(isTraining);
            });
            SimpleLayer[] tmpSeqFinalOutput = new SimpleLayer[numStates];
            for (int i = 0; i < numStates; i++)
            {
                tmpSeqFinalOutput[i] = (SimpleLayer)seqFinalOutput.GetValue(i);
                seqBestOutput[i] = tmpSeqFinalOutput[i].GetBestOutputIndex(isTraining);
            }

            rawOutputLayer = tmp_rawOutputLayer;

            return tmpSeqFinalOutput;

        }

        public override List<double[]> ComputeTopHiddenLayerOutput(Sequence pSequence)
        {
            SimpleLayer[] layer = ComputeBottomLayer(pSequence, forwardHiddenLayers[0], backwardHiddenLayers[0]);
            for (int i = 1; i < forwardHiddenLayers.Count; i++)
            {
                layer = ComputeMiddleLayers(pSequence, layer, forwardHiddenLayers[i], backwardHiddenLayers[i]);
            }
            List<double[]> outputs = new List<double[]>(layer.Length);
            for (int i = 0; i < layer.Length; i++)
            {
                outputs.Add(layer[i].cellOutput);
            }
            return outputs;
        }

        /// <summary>
        /// Computing the output of each layer in the neural network
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="isTraining"></param>
        /// <param name="layerList"></param>
        /// <param name="rawOutputLayer"></param>
        /// <returns></returns>
        private SimpleLayer[] ComputeLayers(Sequence pSequence, bool isTraining, out List<SimpleLayer[]> layerList, out Matrix<double> rawOutputLayer, bool outputRawScore, out int[] seqBestOutput)
        {
            layerList = new List<SimpleLayer[]>();

            SimpleLayer[] layer = ComputeBottomLayer(pSequence, forwardHiddenLayers[0], backwardHiddenLayers[0]);
            if (isTraining == true)
            {
                layerList.Add(layer);
            }

            for (int i = 1; i < forwardHiddenLayers.Count; i++)
            {
                layer = ComputeMiddleLayers(pSequence, layer, forwardHiddenLayers[i], backwardHiddenLayers[i]);
                if (isTraining == true)
                {
                    layerList.Add(layer);
                }
            }

            return ComputeTopLayer(pSequence, layer, out rawOutputLayer, isTraining, outputRawScore, out seqBestOutput);
        }

        /// <summary>
        /// Pass error from the last layer to the first layer
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="seqFinalOutput"></param>
        /// <returns></returns>
        private void ComputeDeepErr(Sequence pSequence, SimpleLayer[] seqFinalOutput, out List<double[][]> fErrLayers, out List<double[][]> bErrLayers)
        {
            int numStates = pSequence.States.Length;
            int numLayers = forwardHiddenLayers.Count;

            //Calculate output layer error
            for (int curState = 0; curState < numStates; curState++)
            {
                int label = pSequence.States[curState].Label;
                SimpleLayer layer = seqFinalOutput[curState];
                layer.ComputeLayerErr(CRFSeqOutput, pSequence.States[curState], curState);
            }

            //Now we already have err in output layer, let's pass them back to other layers
            fErrLayers = new List<double[][]>();
            bErrLayers = new List<double[][]>();
            for (int i = 0; i < numLayers; i++)
            {
                fErrLayers.Add(null);
                bErrLayers.Add(null);
            }

            //Pass error from i+1 to i layer
            SimpleLayer forwardLayer = forwardHiddenLayers[numLayers - 1];
            SimpleLayer backwardLayer = backwardHiddenLayers[numLayers - 1];

            double[][] errLayer = new double[numStates][];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                errLayer[curState] = new double[forwardLayer.LayerSize];
                forwardLayer.ComputeLayerErr(seqFinalOutput[curState], errLayer[curState], seqFinalOutput[curState].er);
            });
            fErrLayers[numLayers - 1] = errLayer;
            bErrLayers[numLayers - 1] = errLayer;

            // Forward
            for (int i = numLayers - 2; i >= 0; i--)
            {
                forwardLayer = forwardHiddenLayers[i];
                errLayer = new double[numStates][];
                double[][] srcErrLayer = fErrLayers[i + 1];
                Parallel.For(0, numStates, parallelOption, curState =>
                {
                    int curState2 = numStates - curState - 1;

                    errLayer[curState2] = new double[forwardLayer.LayerSize];
                    forwardLayer.ComputeLayerErr(forwardHiddenLayers[i + 1], errLayer[curState2], srcErrLayer[curState2]);
                });

                fErrLayers[i] = errLayer;
            }

            // Backward
            for (int i = numLayers - 2; i >= 0; i--)
            {
                backwardLayer = backwardHiddenLayers[i];
                errLayer = new double[numStates][];
                double[][] srcErrLayer = bErrLayers[i + 1];
                Parallel.For(0, numStates, parallelOption, curState =>
                {
                    errLayer[curState] = new double[backwardLayer.LayerSize];
                    backwardLayer.ComputeLayerErr(backwardHiddenLayers[i + 1], errLayer[curState], srcErrLayer[curState]);
                });

                bErrLayers[i] = errLayer;
            }

        }

        private void DeepLearningNet(Sequence pSequence, SimpleLayer[] seqOutput, List<double[][]> fErrLayers, 
            List<double[][]> bErrLayers, List<SimpleLayer[]> layerList)
        {
            int numStates = pSequence.States.Length;
            int numLayers = forwardHiddenLayers.Count;

            //Learning output layer
            Parallel.Invoke(() =>
            {
                for (int curState = 0; curState < numStates; curState++)
                {
                    seqOutput[curState].LearnFeatureWeights(numStates, curState);
                }
            },
            () =>
            {
                Parallel.For(0, numLayers, parallelOption, i =>
                {
                    Parallel.Invoke(() =>
                    {
                        SimpleLayer forwardLayer = forwardHiddenLayers[i];
                        forwardLayer.netReset(true);
                        for (int curState = 0; curState < numStates; curState++)
                        {
                            forwardLayer.computeLayer(layerList[i][curState].SparseFeature, layerList[i][curState].DenseFeature, true);
                            forwardLayer.er = fErrLayers[i][curState];
                            forwardLayer.LearnFeatureWeights(numStates, curState);
                        }
                    },
                    () =>
                    {
                        SimpleLayer backwardLayer = backwardHiddenLayers[i];
                        backwardLayer.netReset(true);
                        for (int curState = 0; curState < numStates; curState++)
                        {
                            int curState2 = numStates - curState - 1;
                            backwardLayer.computeLayer(layerList[i][curState2].SparseFeature, layerList[i][curState2].DenseFeature, true);
                            backwardLayer.er = bErrLayers[i][curState2];
                            backwardLayer.LearnFeatureWeights(numStates, curState);
                        }
                    });
                });
            });
        }

        /// <summary>
        /// Process a given sequence by bi-directional recurrent neural network
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="runningMode"></param>
        /// <returns></returns>
        public override int[] ProcessSequence(Sequence pSequence, RunningMode runningMode, bool outputRawScore, out Matrix<double> rawOutputLayer)
        {
            List<SimpleLayer[]> layerList;

            //Forward process from bottom layer to top layer
            SimpleLayer[] seqOutput;
            int[] seqBestOutput;
            seqOutput = ComputeLayers(pSequence, runningMode == RunningMode.Training, out layerList, out rawOutputLayer, false, out seqBestOutput);

            if (runningMode != RunningMode.Test)
            {
                int numStates = pSequence.States.Length;
                for (int curState = 0; curState < numStates; curState++)
                {
                    logp += Math.Log10(seqOutput[curState].cellOutput[pSequence.States[curState].Label] + 0.0001);
                }
            }

            if (runningMode == RunningMode.Training)
            {
                //In training mode, we calculate each layer's error and update their net weights
                List<double[][]> fErrLayers;
                List<double[][]> bErrLayers;
                ComputeDeepErr(pSequence, seqOutput, out fErrLayers, out bErrLayers);
                DeepLearningNet(pSequence, seqOutput, fErrLayers, bErrLayers, layerList);
            }

            return seqBestOutput;
        }

        /// <summary>
        /// Process a given sequence by bi-directional recurrent neural network and CRF
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="runningMode"></param>
        /// <returns></returns>
        public override int[] ProcessSequenceCRF(Sequence pSequence, RunningMode runningMode)
        {
            //Reset the network
            int numStates = pSequence.States.Length;
            List<SimpleLayer[]> layerList;
            Matrix<double> rawOutputLayer;

            SimpleLayer[] seqOutput;
            int[] seqBestOutput;
            seqOutput = ComputeLayers(pSequence, runningMode == RunningMode.Training, out layerList, out rawOutputLayer, true, out seqBestOutput);

            ForwardBackward(numStates, rawOutputLayer);

            if (runningMode != RunningMode.Test)
            {
                //Merge forward and backward
                for (int curState = 0; curState < numStates; curState++)
                {
                    logp += Math.Log10(CRFSeqOutput[curState][pSequence.States[curState].Label] + 0.0001);
                }
            }

            int[] predict = Viterbi(rawOutputLayer, numStates);

            if (runningMode == RunningMode.Training)
            {
                UpdateBigramTransition(pSequence);

                List<double[][]> fErrLayers;
                List<double[][]> bErrLayers;
                ComputeDeepErr(pSequence, seqOutput, out fErrLayers, out bErrLayers);
                DeepLearningNet(pSequence, seqOutput, fErrLayers, bErrLayers, layerList);
            }

            return predict;
        }

        public override void SaveModel(string filename)
        {
            //Save meta data
            using (StreamWriter sw = new StreamWriter(filename))
            {
                BinaryWriter fo = new BinaryWriter(sw.BaseStream);

                if (forwardHiddenLayers[0] is BPTTLayer)
                {
                    fo.Write(0);
                }
                else
                {
                    fo.Write(1);
                }

                fo.Write((int)ModelDirection);
                fo.Write(IsCRFTraining);

                fo.Write(forwardHiddenLayers.Count);
                //Save forward layers
                foreach (SimpleLayer layer in forwardHiddenLayers)
                {
                    layer.Save(fo);
                }
                //Save backward layers
                foreach (SimpleLayer layer in backwardHiddenLayers)
                {
                    layer.Save(fo);
                }
                //Save output layer
                OutputLayer.Save(fo);

                if (IsCRFTraining == true)
                {
                    // Save CRF features weights
                    RNNHelper.SaveMatrix(CRFTagTransWeights, fo);
                }
            }
        }

        public override void LoadModel(string filename)
        {
            Logger.WriteLine(Logger.Level.info, "Loading bi-directional model: {0}", filename);

            using (StreamReader sr = new StreamReader(filename))
            {
                BinaryReader br = new BinaryReader(sr.BaseStream);

                int modelType = br.ReadInt32();
                ModelDirection = (MODELDIRECTION)br.ReadInt32();
                IsCRFTraining = br.ReadBoolean();
                
                int layerSize = br.ReadInt32();
                //Load forward layers from file
                forwardHiddenLayers = new List<SimpleLayer>();
                for (int i = 0; i < layerSize; i++)
                {
                    SimpleLayer layer = null;
                    if (modelType == 0)
                    {
                        Logger.WriteLine("Create BPTT hidden layer");
                        layer = new BPTTLayer();
                    }
                    else
                    {
                        Logger.WriteLine("Crate LSTM hidden layer");
                        layer = new LSTMLayer();
                    }

                    layer.Load(br);
                    forwardHiddenLayers.Add(layer);
                }

                //Load backward layers from file
                backwardHiddenLayers = new List<SimpleLayer>();
                for (int i = 0; i < layerSize; i++)
                {
                    SimpleLayer layer = null;
                    if (modelType == 0)
                    {
                        Logger.WriteLine("Create BPTT hidden layer");
                        layer = new BPTTLayer();
                    }
                    else
                    {
                        Logger.WriteLine("Crate LSTM hidden layer");
                        layer = new LSTMLayer();
                    }

                    layer.Load(br);
                    backwardHiddenLayers.Add(layer);
                }

                OutputLayer = new SimpleLayer();
                OutputLayer.Load(br);

                if (IsCRFTraining == true)
                {
                    Logger.WriteLine("Loading CRF tag trans weights...");
                    CRFTagTransWeights = RNNHelper.LoadMatrix(br);
                }
            }
        }
    }
}
