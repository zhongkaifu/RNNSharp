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

namespace RNNSharp
{
    internal class BiRNN<T> : RNN<T> where T : ISequence
    {
        private readonly Vector<float> vecConst2 = new Vector<float>(2.0f);
        private List<SimpleLayer> backwardHiddenLayers = new List<SimpleLayer>();
        private List<SimpleLayer> forwardHiddenLayers = new List<SimpleLayer>();

        private Array seqFinalOutput;

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
            foreach (var layer in forwardHiddenLayers)
            {
                layer.CleanLearningRate();
            }

            foreach (var layer in backwardHiddenLayers)
            {
                layer.CleanLearningRate();
            }

            OutputLayer.CleanLearningRate();
        }

        private SimpleLayer[] ComputeMiddleLayers(Sequence pSequence, SimpleLayer[] lastLayers, SimpleLayer forwardLayer,
            SimpleLayer backwardLayer)
        {
            var numStates = lastLayers.Length;

            SimpleLayer[] mForward = null;
            SimpleLayer[] mBackward = null;
            Parallel.Invoke(() =>
            {
                //Computing forward RNN
                forwardLayer.Reset(false);
                mForward = new SimpleLayer[lastLayers.Length];
                for (var curState = 0; curState < lastLayers.Length; curState++)
                {
                    var state = pSequence.States[curState];
                    forwardLayer.ForwardPass(state.SparseFeature, lastLayers[curState].Cell);
                    mForward[curState] = forwardLayer.CloneHiddenLayer();
                }
            },
                () =>
                {
                    //Computing backward RNN
                    backwardLayer.Reset(false);
                    mBackward = new SimpleLayer[lastLayers.Length];
                    for (var curState = lastLayers.Length - 1; curState >= 0; curState--)
                    {
                        var state = pSequence.States[curState];
                        backwardLayer.ForwardPass(state.SparseFeature, lastLayers[curState].Cell);
                        mBackward[curState] = backwardLayer.CloneHiddenLayer();
                    }
                });

            //Merge forward and backward
            var mergedLayer = new SimpleLayer[numStates];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                var state = pSequence.States[curState];
                mergedLayer[curState] = new SimpleLayer(forwardLayer.LayerConfig)
                {
                    SparseFeature = state.SparseFeature,
                    DenseFeature = lastLayers[curState].Cell
                };

                var forwardCells = mForward[curState];
                var backwardCells = mBackward[curState];

                var i = 0;
                while (i < forwardLayer.LayerSize - Vector<float>.Count)
                {
                    var v1 = new Vector<float>(forwardCells.Cell, i);
                    var v2 = new Vector<float>(backwardCells.Cell, i);
                    var v = (v1 + v2) / vecConst2;

                    v.CopyTo(mergedLayer[curState].Cell, i);

                    i += Vector<float>.Count;
                }

                while (i < forwardLayer.LayerSize)
                {
                    mergedLayer[curState].Cell[i] =
                        (float)((forwardCells.Cell[i] + backwardCells.Cell[i]) / 2.0);
                    i++;
                }
            });

            return mergedLayer;
        }

        /// <summary>
        ///     Compute the output of bottom layer
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="forwardLayer"></param>
        /// <param name="backwardLayer"></param>
        /// <returns></returns>
        private SimpleLayer[] ComputeBottomLayer(Sequence pSequence, SimpleLayer forwardLayer, SimpleLayer backwardLayer)
        {
            var numStates = pSequence.States.Length;
            SimpleLayer[] mForward = null;
            SimpleLayer[] mBackward = null;
            Parallel.Invoke(() =>
            {
                //Computing forward RNN
                forwardLayer.Reset(false);
                mForward = new SimpleLayer[numStates];
                for (var curState = 0; curState < numStates; curState++)
                {
                    var state = pSequence.States[curState];
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
                    for (var curState = numStates - 1; curState >= 0; curState--)
                    {
                        var state = pSequence.States[curState];
                        SetRuntimeFeatures(state, curState, numStates, null, false);
                        backwardLayer.ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());
                        //compute probability distribution

                        mBackward[curState] = backwardLayer.CloneHiddenLayer();
                    }
                });

            var mergedLayer = new SimpleLayer[numStates];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                var state = pSequence.States[curState];
                mergedLayer[curState] = new SimpleLayer(forwardLayer.LayerConfig)
                {
                    SparseFeature = state.SparseFeature,
                    DenseFeature = state.DenseFeature.CopyTo()
                };

                var forwardCells = mForward[curState];
                var backwardCells = mBackward[curState];

                var i = 0;
                while (i < forwardLayer.LayerSize - Vector<float>.Count)
                {
                    var v1 = new Vector<float>(forwardCells.Cell, i);
                    var v2 = new Vector<float>(backwardCells.Cell, i);
                    var v = (v1 + v2) / vecConst2;

                    v.CopyTo(mergedLayer[curState].Cell, i);

                    i += Vector<float>.Count;
                }

                while (i < forwardLayer.LayerSize)
                {
                    mergedLayer[curState].Cell[i] =
                        (float)((forwardCells.Cell[i] + backwardCells.Cell[i]) / 2.0);
                    i++;
                }
            });

            return mergedLayer;
        }

        private SimpleLayer[] ComputeTopLayer(Sequence pSequence, SimpleLayer[] lastLayer,
            out Matrix<float> rawOutputLayer, bool isTraining, bool outputRawScore, out int[] seqBestOutput)
        {
            var numStates = lastLayer.Length;
            seqBestOutput = new int[numStates];

            //Calculate output layer
            Matrix<float> tmp_rawOutputLayer = null;
            if (outputRawScore)
            {
                tmp_rawOutputLayer = new Matrix<float>(numStates, OutputLayer.LayerSize);
            }

            var labelSet = pSequence.States.Select(state => state.Label).ToList();

            //Initialize output layer or reallocate it
            if (seqFinalOutput == null || seqFinalOutput.Length < numStates)
            {
                seqFinalOutput = Array.CreateInstance(OutputLayer.GetType(), numStates);
                for (var i = 0; i < numStates; i++)
                {
                    seqFinalOutput.SetValue(Activator.CreateInstance(OutputLayer.GetType(), OutputLayer.LayerConfig), i);
                    OutputLayer.ShallowCopyWeightTo((SimpleLayer)seqFinalOutput.GetValue(i));
                }
            }

            Parallel.For(0, numStates, parallelOption, curState =>
            {
                var state = pSequence.States[curState];
                var outputCells = (SimpleLayer)seqFinalOutput.GetValue(curState);
                outputCells.LabelShortList = labelSet;
                outputCells.ForwardPass(state.SparseFeature, lastLayer[curState].Cell, isTraining);

                if (outputRawScore)
                {
                    outputCells.Cell.CopyTo(tmp_rawOutputLayer[curState], 0);
                }
                outputCells.Softmax(isTraining);
            });
            var tmpSeqFinalOutput = new SimpleLayer[numStates];
            for (var i = 0; i < numStates; i++)
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
            var layer = ComputeBottomLayer(pSequence, forwardHiddenLayers[0], backwardHiddenLayers[0]);
            for (var i = 1; i < forwardHiddenLayers.Count; i++)
            {
                layer = ComputeMiddleLayers(pSequence, layer, forwardHiddenLayers[i], backwardHiddenLayers[i]);
            }
            var outputs = new List<float[]>(layer.Length);
            outputs.AddRange(layer.Select(t => t.Cell));
            return outputs;
        }

        /// <summary>
        ///     Computing the output of each layer in the neural network
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="isTraining"></param>
        /// <param name="layerList"></param>
        /// <param name="rawOutputLayer"></param>
        /// <returns></returns>
        private SimpleLayer[] ComputeLayers(Sequence pSequence, bool isTraining, out List<SimpleLayer[]> layerList,
            out Matrix<float> rawOutputLayer, bool outputRawScore, out int[] seqBestOutput)
        {
            layerList = new List<SimpleLayer[]>();

            var layer = ComputeBottomLayer(pSequence, forwardHiddenLayers[0], backwardHiddenLayers[0]);
            if (isTraining)
            {
                layerList.Add(layer);
            }

            for (var i = 1; i < forwardHiddenLayers.Count; i++)
            {
                layer = ComputeMiddleLayers(pSequence, layer, forwardHiddenLayers[i], backwardHiddenLayers[i]);
                if (isTraining)
                {
                    layerList.Add(layer);
                }
            }

            return ComputeTopLayer(pSequence, layer, out rawOutputLayer, isTraining, outputRawScore, out seqBestOutput);
        }

        /// <summary>
        ///     Pass error from the last layer to the first layer
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="seqFinalOutput"></param>
        /// <returns></returns>
        private void ComputeDeepErr(Sequence pSequence, SimpleLayer[] seqFinalOutput, out List<float[][]> fErrLayers,
            out List<float[][]> bErrLayers)
        {
            var numStates = pSequence.States.Length;
            var numLayers = forwardHiddenLayers.Count;

            //Calculate output layer error
            for (var curState = 0; curState < numStates; curState++)
            {
                var label = pSequence.States[curState].Label;
                var layer = seqFinalOutput[curState];
                layer.ComputeLayerErr(CRFSeqOutput, pSequence.States[curState], curState);
            }

            //Now we already have err in output layer, let's pass them back to other layers
            fErrLayers = new List<float[][]>();
            bErrLayers = new List<float[][]>();
            for (var i = 0; i < numLayers; i++)
            {
                fErrLayers.Add(null);
                bErrLayers.Add(null);
            }

            //Pass error from i+1 to i layer
            var forwardLayer = forwardHiddenLayers[numLayers - 1];
            SimpleLayer backwardLayer;

            var errLayer = new float[numStates][];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                errLayer[curState] = new float[forwardLayer.LayerSize];
                forwardLayer.ComputeLayerErr(seqFinalOutput[curState], errLayer[curState], seqFinalOutput[curState].Err);
            });
            fErrLayers[numLayers - 1] = errLayer;
            bErrLayers[numLayers - 1] = errLayer;

            // Forward
            for (var i = numLayers - 2; i >= 0; i--)
            {
                forwardLayer = forwardHiddenLayers[i];
                errLayer = new float[numStates][];
                var srcErrLayer = fErrLayers[i + 1];
                Parallel.For(0, numStates, parallelOption, curState =>
                {
                    var curState2 = numStates - curState - 1;

                    errLayer[curState2] = new float[forwardLayer.LayerSize];
                    forwardLayer.ComputeLayerErr(forwardHiddenLayers[i + 1], errLayer[curState2], srcErrLayer[curState2]);
                });

                fErrLayers[i] = errLayer;
            }

            // Backward
            for (var i = numLayers - 2; i >= 0; i--)
            {
                backwardLayer = backwardHiddenLayers[i];
                errLayer = new float[numStates][];
                var srcErrLayer = bErrLayers[i + 1];
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
            var numStates = pSequence.States.Length;
            var numLayers = forwardHiddenLayers.Count;

            //Learning output layer
            Parallel.Invoke(() =>
            {
                for (var curState = 0; curState < numStates; curState++)
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
                            var forwardLayer = forwardHiddenLayers[i];
                            forwardLayer.Reset(true);
                            for (var curState = 0; curState < numStates; curState++)
                            {
                                forwardLayer.ForwardPass(layerList[i][curState].SparseFeature,
                                    layerList[i][curState].DenseFeature, true);
                                forwardLayer.Err = fErrLayers[i][curState];
                                forwardLayer.BackwardPass(numStates, curState);
                            }
                        },
                            () =>
                            {
                                var backwardLayer = backwardHiddenLayers[i];
                                backwardLayer.Reset(true);
                                for (var curState = 0; curState < numStates; curState++)
                                {
                                    var curState2 = numStates - curState - 1;
                                    backwardLayer.ForwardPass(layerList[i][curState2].SparseFeature,
                                        layerList[i][curState2].DenseFeature, true);
                                    backwardLayer.Err = bErrLayers[i][curState2];
                                    backwardLayer.BackwardPass(numStates, curState);
                                }
                            });
                    });
                });
        }

        /// <summary>
        ///     Process a given sequence by bi-directional recurrent neural network
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="runningMode"></param>
        /// <returns></returns>
        public override int[] ProcessSequence(Sequence pSequence, RunningMode runningMode, bool outputRawScore,
            out Matrix<float> rawOutputLayer)
        {
            List<SimpleLayer[]> layerList;

            //Forward process from bottom layer to top layer
            int[] seqBestOutput;
            var seqOutput = ComputeLayers(pSequence, runningMode == RunningMode.Training, out layerList,
                out rawOutputLayer,
                outputRawScore, out seqBestOutput);

            if (runningMode != RunningMode.Test)
            {
                var numStates = pSequence.States.Length;
                for (var curState = 0; curState < numStates; curState++)
                {
                    logp += Math.Log10(seqOutput[curState].Cell[pSequence.States[curState].Label] + 0.0001);
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
        ///     Process a given sequence by bi-directional recurrent neural network and CRF
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="runningMode"></param>
        /// <returns></returns>
        public override int[] ProcessSequenceCRF(Sequence pSequence, RunningMode runningMode)
        {
            //Reset the network
            var numStates = pSequence.States.Length;
            List<SimpleLayer[]> layerList;
            Matrix<float> rawOutputLayer;

            int[] seqBestOutput;
            var seqOutput = ComputeLayers(pSequence, runningMode == RunningMode.Training, out layerList,
                out rawOutputLayer,
                true, out seqBestOutput);

            ForwardBackward(numStates, rawOutputLayer);

            if (runningMode != RunningMode.Test)
            {
                //Merge forward and backward
                for (var curState = 0; curState < numStates; curState++)
                {
                    logp += Math.Log10(CRFSeqOutput[curState][pSequence.States[curState].Label] + 0.0001);
                }
            }

            var predict = Viterbi(rawOutputLayer, numStates);

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
            using (var sw = new StreamWriter(filename))
            {
                var fo = new BinaryWriter(sw.BaseStream);

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
                foreach (var layer in forwardHiddenLayers)
                {
                    layer.Save(fo);
                }
                //Save backward layers
                foreach (var layer in backwardHiddenLayers)
                {
                    layer.Save(fo);
                }
                //Save output layer
                OutputLayer.Save(fo);

                if (IsCRFTraining)
                {
                    // Save CRF features weights
                    RNNHelper.SaveMatrix(CRFTagTransWeights, fo);
                }
            }
        }

        public override void LoadModel(string filename)
        {
            Logger.WriteLine(Logger.Level.info, "Loading bi-directional model: {0}", filename);

            using (var sr = new StreamReader(filename))
            {
                var br = new BinaryReader(sr.BaseStream);

                var layerType = (LAYERTYPE)br.ReadInt32();
                IsCRFTraining = br.ReadBoolean();

                var layerSize = br.ReadInt32();
                //Load forward layers from file
                forwardHiddenLayers = new List<SimpleLayer>();
                for (var i = 0; i < layerSize; i++)
                {
                    SimpleLayer layer;
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
                for (var i = 0; i < layerSize; i++)
                {
                    SimpleLayer layer;
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

                if (IsCRFTraining)
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