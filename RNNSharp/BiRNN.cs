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
    class BiRNN<T> : RNN<T> where T : ISequence
    {
        private Vector<float> vecConst2 = new Vector<float>(2.0f);
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

            RNNHelper.vecMaxGrad = new Vector<float>(RNNHelper.GradientCutoff);
            RNNHelper.vecMinGrad = new Vector<float>(-RNNHelper.GradientCutoff);
            RNNHelper.vecNormalLearningRate = new Vector<float>(RNNHelper.LearningRate);
        }

        private SimpleLayer[] ComputeMiddleLayers(Sequence pSequence, SimpleLayer[] lastLayers, SimpleLayer forwardLayer, SimpleLayer backwardLayer)
        {
            int numStates = lastLayers.Length;

            SimpleLayer[] mForward = null;
            SimpleLayer[] mBackward = null;
            Parallel.Invoke(() =>
            {
                //Computing forward RNN      
                forwardLayer.Reset(false);
                mForward = new SimpleLayer[lastLayers.Length];
                for (int curState = 0; curState < lastLayers.Length; curState++)
                {
                    State state = pSequence.States[curState];
                    forwardLayer.ForwardPass(state.SparseFeature, lastLayers[curState].cellOutput);
                    mForward[curState] = forwardLayer.CloneHiddenLayer();
                }
            },
             () =>
             {
                 //Computing backward RNN
                 backwardLayer.Reset(false);
                 mBackward = new SimpleLayer[lastLayers.Length];
                 for (int curState = lastLayers.Length - 1; curState >= 0; curState--)
                 {
                     State state = pSequence.States[curState];
                     backwardLayer.ForwardPass(state.SparseFeature, lastLayers[curState].cellOutput);
                     mBackward[curState] = backwardLayer.CloneHiddenLayer();
                 }
             });

            //Merge forward and backward
            SimpleLayer[] mergedLayer = new SimpleLayer[numStates];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                State state = pSequence.States[curState];
                mergedLayer[curState] = new SimpleLayer(forwardLayer.LayerConfig);
                mergedLayer[curState].SparseFeature = state.SparseFeature;
                mergedLayer[curState].DenseFeature = lastLayers[curState].cellOutput;

                SimpleLayer forwardCells = mForward[curState];
                SimpleLayer backwardCells = mBackward[curState];

                int i = 0;
                while (i < forwardLayer.LayerSize - Vector<float>.Count)
                {
                    Vector<float> v1 = new Vector<float>(forwardCells.cellOutput, i);
                    Vector<float> v2 = new Vector<float>(backwardCells.cellOutput, i);
                    Vector<float> v = (v1 + v2) / vecConst2;

                    v.CopyTo(mergedLayer[curState].cellOutput, i);

                    i += Vector<float>.Count;
                }

                while (i < forwardLayer.LayerSize)
                {
                    mergedLayer[curState].cellOutput[i] = (float)((forwardCells.cellOutput[i] + backwardCells.cellOutput[i]) / 2.0);
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
                forwardLayer.Reset(false);
                mForward = new SimpleLayer[numStates];
                for (int curState = 0; curState < numStates; curState++)
                {
                    State state = pSequence.States[curState];
                    SetRuntimeFeatures(state, curState, numStates, null);
                    forwardLayer.ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());
                    mForward[curState] = forwardLayer.CloneHiddenLayer();
                }
            },
             () =>
             {
                 //Computing backward RNN
                 backwardLayer.Reset(false);
                 mBackward = new SimpleLayer[numStates];
                 for (int curState = numStates - 1; curState >= 0; curState--)
                 {
                     State state = pSequence.States[curState];
                     SetRuntimeFeatures(state, curState, numStates, null, false);
                     backwardLayer.ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());      //compute probability distribution

                     mBackward[curState] = backwardLayer.CloneHiddenLayer();
                 }
             });

            SimpleLayer[] mergedLayer = new SimpleLayer[numStates];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                State state = pSequence.States[curState];
                mergedLayer[curState] = new SimpleLayer(forwardLayer.LayerConfig);
                mergedLayer[curState].SparseFeature = state.SparseFeature;
                mergedLayer[curState].DenseFeature = state.DenseFeature.CopyTo();

                SimpleLayer forwardCells = mForward[curState];
                SimpleLayer backwardCells = mBackward[curState];

                int i = 0;
                while (i < forwardLayer.LayerSize - Vector<float>.Count)
                {
                    Vector<float> v1 = new Vector<float>(forwardCells.cellOutput, i);
                    Vector<float> v2 = new Vector<float>(backwardCells.cellOutput, i);
                    Vector<float> v = (v1 + v2) / vecConst2;

                    v.CopyTo(mergedLayer[curState].cellOutput, i);

                    i += Vector<float>.Count;
                }

                while (i < forwardLayer.LayerSize)
                {
                    mergedLayer[curState].cellOutput[i] = (float)((forwardCells.cellOutput[i] + backwardCells.cellOutput[i]) / 2.0);
                    i++;
                }
            });

            return mergedLayer;
        }

        Array seqFinalOutput = null;
        private SimpleLayer[] ComputeTopLayer(Sequence pSequence, SimpleLayer[] lastLayer, out Matrix<float> rawOutputLayer, bool isTraining, bool outputRawScore, out int[] seqBestOutput)
        {
            int numStates = lastLayer.Length;
            seqBestOutput = new int[numStates];

            //Calculate output layer
            Matrix<float> tmp_rawOutputLayer = null;
            if (outputRawScore == true)
            {
                tmp_rawOutputLayer = new Matrix<float>(numStates, OutputLayer.LayerSize);
            }

            List<int> labelSet = new List<int>();
            foreach (State state in pSequence.States)
            {
                labelSet.Add(state.Label);
            }

            //Initialize output layer or reallocate it
            if (seqFinalOutput == null || seqFinalOutput.Length < numStates)
            {
                seqFinalOutput = Array.CreateInstance(OutputLayer.GetType(), numStates);
                for (int i = 0; i < numStates; i++)
                {
                    seqFinalOutput.SetValue(Activator.CreateInstance(OutputLayer.GetType(), OutputLayer.LayerConfig), i);
                    OutputLayer.ShallowCopyWeightTo((SimpleLayer)seqFinalOutput.GetValue(i));
                }
            }

            Parallel.For(0, numStates, parallelOption, curState =>
            {
                State state = pSequence.States[curState];
                var outputCells = (SimpleLayer)seqFinalOutput.GetValue(curState);
                outputCells.LabelShortList = labelSet;
                outputCells.ForwardPass(state.SparseFeature, lastLayer[curState].cellOutput, isTraining);

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

        public override int GetTopHiddenLayerSize()
        {
            return forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize;
        }

        public override List<float[]> ComputeTopHiddenLayerOutput(Sequence pSequence)
        {
            SimpleLayer[] layer = ComputeBottomLayer(pSequence, forwardHiddenLayers[0], backwardHiddenLayers[0]);
            for (int i = 1; i < forwardHiddenLayers.Count; i++)
            {
                layer = ComputeMiddleLayers(pSequence, layer, forwardHiddenLayers[i], backwardHiddenLayers[i]);
            }
            List<float[]> outputs = new List<float[]>(layer.Length);
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
        private SimpleLayer[] ComputeLayers(Sequence pSequence, bool isTraining, out List<SimpleLayer[]> layerList, out Matrix<float> rawOutputLayer, bool outputRawScore, out int[] seqBestOutput)
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
        private void ComputeDeepErr(Sequence pSequence, SimpleLayer[] seqFinalOutput, out List<float[][]> fErrLayers, out List<float[][]> bErrLayers)
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
            fErrLayers = new List<float[][]>();
            bErrLayers = new List<float[][]>();
            for (int i = 0; i < numLayers; i++)
            {
                fErrLayers.Add(null);
                bErrLayers.Add(null);
            }

            //Pass error from i+1 to i layer
            SimpleLayer forwardLayer = forwardHiddenLayers[numLayers - 1];
            SimpleLayer backwardLayer = backwardHiddenLayers[numLayers - 1];

            float[][] errLayer = new float[numStates][];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                errLayer[curState] = new float[forwardLayer.LayerSize];
                forwardLayer.ComputeLayerErr(seqFinalOutput[curState], errLayer[curState], seqFinalOutput[curState].er);
            });
            fErrLayers[numLayers - 1] = errLayer;
            bErrLayers[numLayers - 1] = errLayer;

            // Forward
            for (int i = numLayers - 2; i >= 0; i--)
            {
                forwardLayer = forwardHiddenLayers[i];
                errLayer = new float[numStates][];
                float[][] srcErrLayer = fErrLayers[i + 1];
                Parallel.For(0, numStates, parallelOption, curState =>
                {
                    int curState2 = numStates - curState - 1;

                    errLayer[curState2] = new float[forwardLayer.LayerSize];
                    forwardLayer.ComputeLayerErr(forwardHiddenLayers[i + 1], errLayer[curState2], srcErrLayer[curState2]);
                });

                fErrLayers[i] = errLayer;
            }

            // Backward
            for (int i = numLayers - 2; i >= 0; i--)
            {
                backwardLayer = backwardHiddenLayers[i];
                errLayer = new float[numStates][];
                float[][] srcErrLayer = bErrLayers[i + 1];
                Parallel.For(0, numStates, parallelOption, curState =>
                {
                    errLayer[curState] = new float[backwardLayer.LayerSize];
                    backwardLayer.ComputeLayerErr(backwardHiddenLayers[i + 1], errLayer[curState], srcErrLayer[curState]);
                });

                bErrLayers[i] = errLayer;
            }

        }

        private void DeepLearningNet(Sequence pSequence, SimpleLayer[] seqOutput, List<float[][]> fErrLayers, 
            List<float[][]> bErrLayers, List<SimpleLayer[]> layerList)
        {
            int numStates = pSequence.States.Length;
            int numLayers = forwardHiddenLayers.Count;

            //Learning output layer
            Parallel.Invoke(() =>
            {
                for (int curState = 0; curState < numStates; curState++)
                {
                    seqOutput[curState].BackwardPass(numStates, curState);
                }
            },
            () =>
            {
                Parallel.For(0, numLayers, parallelOption, i =>
                {
                    Parallel.Invoke(() =>
                    {
                        SimpleLayer forwardLayer = forwardHiddenLayers[i];
                        forwardLayer.Reset(true);
                        for (int curState = 0; curState < numStates; curState++)
                        {
                            forwardLayer.ForwardPass(layerList[i][curState].SparseFeature, layerList[i][curState].DenseFeature, true);
                            forwardLayer.er = fErrLayers[i][curState];
                            forwardLayer.BackwardPass(numStates, curState);
                        }
                    },
                    () =>
                    {
                        SimpleLayer backwardLayer = backwardHiddenLayers[i];
                        backwardLayer.Reset(true);
                        for (int curState = 0; curState < numStates; curState++)
                        {
                            int curState2 = numStates - curState - 1;
                            backwardLayer.ForwardPass(layerList[i][curState2].SparseFeature, layerList[i][curState2].DenseFeature, true);
                            backwardLayer.er = bErrLayers[i][curState2];
                            backwardLayer.BackwardPass(numStates, curState);
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
        public override int[] ProcessSequence(Sequence pSequence, RunningMode runningMode, bool outputRawScore, out Matrix<float> rawOutputLayer)
        {
            List<SimpleLayer[]> layerList;

            //Forward process from bottom layer to top layer
            SimpleLayer[] seqOutput;
            int[] seqBestOutput;
            seqOutput = ComputeLayers(pSequence, runningMode == RunningMode.Training, out layerList, out rawOutputLayer, outputRawScore, out seqBestOutput);

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
                List<float[][]> fErrLayers;
                List<float[][]> bErrLayers;
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
            Matrix<float> rawOutputLayer;

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

                List<float[][]> fErrLayers;
                List<float[][]> bErrLayers;
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
                    fo.Write((int)LAYERTYPE.BPTT);
                }
                else
                {
                    fo.Write((int)LAYERTYPE.LSTM);
                }

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

                LAYERTYPE layerType = (LAYERTYPE)br.ReadInt32();
                IsCRFTraining = br.ReadBoolean();
                
                int layerSize = br.ReadInt32();
                //Load forward layers from file
                forwardHiddenLayers = new List<SimpleLayer>();
                for (int i = 0; i < layerSize; i++)
                {
                    SimpleLayer layer = null;
                    if (layerType == LAYERTYPE.BPTT)
                    {
                        Logger.WriteLine("Create BPTT hidden layer");
                        layer = new BPTTLayer();
                    }
                    else
                    {
                        Logger.WriteLine("Create LSTM hidden layer");
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
                    if (layerType == LAYERTYPE.BPTT)
                    {
                        Logger.WriteLine("Create BPTT hidden layer");
                        layer = new BPTTLayer();
                    }
                    else
                    {
                        Logger.WriteLine("Create LSTM hidden layer");
                        layer = new LSTMLayer();
                    }

                    layer.Load(br);
                    backwardHiddenLayers.Add(layer);
                }

                Logger.WriteLine("Create output layer");
                OutputLayer = new SimpleLayer();
                OutputLayer.Load(br);

                if (IsCRFTraining == true)
                {
                    Logger.WriteLine("Loading CRF tag trans weights...");
                    CRFTagTransWeights = RNNHelper.LoadMatrix(br);
                }
            }
        }

        public override int[] ProcessSeq2Seq(SequencePair pSequence, RunningMode runningMode)
        {
            throw new NotImplementedException();
        }

        public override int[] TestSeq2Seq(Sentence srcSentence, Config featurizer)
        {
            throw new NotImplementedException();
        }
    }
}
