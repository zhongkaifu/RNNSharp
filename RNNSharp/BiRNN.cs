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

        private SimpleLayer[] seqFinalOutput;

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

        private float[][] ComputeMiddleLayers(Sequence pSequence, float[][] lastLayerOutputs, SimpleLayer forwardLayer,
            SimpleLayer backwardLayer, RunningMode runningMode, out Neuron[] forwardNeuron, out Neuron[] backwardNeuron)
        {
            var numStates = lastLayerOutputs.Length;
            Neuron[] tmpForwardNeuron = null;
            Neuron[] tmpBackwardNeuron = null;

            Parallel.Invoke(() =>
            {
                //Computing forward RNN
                forwardLayer.Reset();
                tmpForwardNeuron = new Neuron[numStates];
                for (var curState = 0; curState < numStates; curState++)
                {
                    var state = pSequence.States[curState];
                    forwardLayer.SetRunningMode(runningMode);
                    forwardLayer.ForwardPass(state.SparseFeature, lastLayerOutputs[curState]);
                    tmpForwardNeuron[curState] = forwardLayer.CopyNeuronTo();
                }
            },
                () =>
                {
                    //Computing backward RNN
                    backwardLayer.Reset();
                    tmpBackwardNeuron = new Neuron[numStates];
                    for (var curState = numStates - 1; curState >= 0; curState--)
                    {
                        var state = pSequence.States[curState];
                        backwardLayer.SetRunningMode(runningMode);
                        backwardLayer.ForwardPass(state.SparseFeature, lastLayerOutputs[curState]);
                        tmpBackwardNeuron[curState] = backwardLayer.CopyNeuronTo();
                    }
                });

            //Merge forward and backward
            float[][] stateOutputs = new float[numStates][];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                stateOutputs[curState] = new float[forwardLayer.LayerSize * 2];

                var forwardCells = tmpForwardNeuron[curState].Cells;
                var backwardCells = tmpBackwardNeuron[curState].Cells;
                var mergedLayer = stateOutputs[curState];

                var i = 0;
            //    var vDiv2 = new Vector<float>(2.0f);
                while (i < forwardLayer.LayerSize - Vector<float>.Count)
                {
                    var v1 = new Vector<float>(forwardCells, i);
                    var v2 = new Vector<float>(backwardCells, i);
                    //      var v = (v1 + v2) / vDiv2;

                    //    v.CopyTo(mergedLayer, i);

                    v1.CopyTo(mergedLayer, i);
                    v2.CopyTo(mergedLayer, forwardLayer.LayerSize + i);

                    i += Vector<float>.Count;
                }

                while (i < forwardLayer.LayerSize)
                {
                    mergedLayer[i] = forwardCells[i];
                    mergedLayer[forwardLayer.LayerSize + i] = backwardCells[i];
                   // mergedLayer[i] = (forwardCells[i] + backwardCells[i]) / 2.0f;
                    i++;
                }
            });

            forwardNeuron = tmpForwardNeuron;
            backwardNeuron = tmpBackwardNeuron;

            return stateOutputs;
        }

        /// <summary>
        ///     Compute the output of bottom layer
        /// </summary>
        /// <param name="sequence"></param>
        /// <param name="forwardLayer"></param>
        /// <param name="backwardLayer"></param>
        /// <returns></returns>
        private float[][] ComputeBottomLayer(Sequence sequence, SimpleLayer forwardLayer, SimpleLayer backwardLayer, RunningMode runningMode, 
            out Neuron[] forwardNeurons, out Neuron[] backwardNeurons)
        {
            var numStates = sequence.States.Length;
            Neuron[] tmpForwardNeurons = null;
            Neuron[] tmpBackwardNeurons = null;
            Parallel.Invoke(() =>
            {
                //Computing forward RNN
                forwardLayer.Reset();
                tmpForwardNeurons = new Neuron[numStates];
                for (var curState = 0; curState < numStates; curState++)
                {
                    var state = sequence.States[curState];
                    SetRuntimeFeatures(state, curState, numStates, null);
                    forwardLayer.SetRunningMode(runningMode);
                    forwardLayer.ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());
                    tmpForwardNeurons[curState] = forwardLayer.CopyNeuronTo();
                }
            },
                () =>
                {
                    //Computing backward RNN
                    backwardLayer.Reset();
                    tmpBackwardNeurons = new Neuron[numStates];
                    for (var curState = numStates - 1; curState >= 0; curState--)
                    {
                        var state = sequence.States[curState];
                        SetRuntimeFeatures(state, curState, numStates, null, false);
                        backwardLayer.SetRunningMode(runningMode);
                        backwardLayer.ForwardPass(state.SparseFeature, state.DenseFeature.CopyTo());
                        tmpBackwardNeurons[curState] = backwardLayer.CopyNeuronTo();
                    }
                });

            float[][] stateOutputs = new float[numStates][];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                stateOutputs[curState] = new float[forwardLayer.LayerSize * 2];

                var forwardCells = tmpForwardNeurons[curState].Cells;
                var backwardCells = tmpBackwardNeurons[curState].Cells;
                var mergedLayer = stateOutputs[curState];

                var i = 0;
            //    var vDiv2 = new Vector<float>(2.0f);
                while (i < forwardLayer.LayerSize - Vector<float>.Count)
                {
                    var v1 = new Vector<float>(forwardCells, i);
                    var v2 = new Vector<float>(backwardCells, i);
                    //      var v = (v1 + v2) / vDiv2;

                    v1.CopyTo(mergedLayer, i);
                    v2.CopyTo(mergedLayer, forwardLayer.LayerSize + i);


//                    v.CopyTo(mergedLayer, i);

                    i += Vector<float>.Count;
                }

                while (i < forwardLayer.LayerSize)
                {
                    mergedLayer[i] = forwardCells[i];
                    mergedLayer[forwardLayer.LayerSize + i] = backwardCells[i];
                  //  mergedLayer[i] = (forwardCells[i] + backwardCells[i]) / 2.0f;
                    i++;
                }
            });

            forwardNeurons = tmpForwardNeurons;
            backwardNeurons = tmpBackwardNeurons;

            return stateOutputs;
        }

        private SimpleLayer[] ComputeTopLayer(Sequence pSequence, float[][] lastLayerOutputs,
            out Matrix<float> rawOutputLayer, RunningMode runningMode, bool outputRawScore)
        {
            var numStates = lastLayerOutputs.Length;

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

            Parallel.For(0, numStates, parallelOption, curState =>
            {
                var state = pSequence.States[curState];
                var outputCells = seqFinalOutput[curState];
                outputCells.LabelShortList = labelSet;
                outputCells.SetRunningMode(runningMode);
                outputCells.ForwardPass(state.SparseFeature, lastLayerOutputs[curState]);

                if (outputRawScore)
                {
                    outputCells.Cells.CopyTo(tmpOutputResult[curState], 0);
                }
            });

            rawOutputLayer = tmpOutputResult;
            return seqFinalOutput;
        }

        public override int GetTopHiddenLayerSize()
        {
            return forwardHiddenLayers[forwardHiddenLayers.Count - 1].LayerSize;
        }

        public override float[][] ComputeTopHiddenLayerOutput(Sequence pSequence)
        {
            Neuron[] forwardCell;
            Neuron[] backwardCell;
            var layerOutputs = ComputeBottomLayer(pSequence, forwardHiddenLayers[0], backwardHiddenLayers[0], RunningMode.Test, out forwardCell, out backwardCell);
            for (var i = 1; i < forwardHiddenLayers.Count; i++)
            {
                layerOutputs = ComputeMiddleLayers(pSequence, layerOutputs, forwardHiddenLayers[i], backwardHiddenLayers[i], RunningMode.Test, out forwardCell, out backwardCell);
            }

            return layerOutputs;
        }

        /// <summary>
        ///     Computing the output of each layer in the neural network
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="isTraining"></param>
        /// <param name="layerList"></param>
        /// <param name="rawOutputLayer"></param>
        /// <returns></returns>
        private SimpleLayer[] ComputeLayers(Sequence pSequence, RunningMode runningMode, out List<float[][]> layerList,
            out Matrix<float> rawOutputLayer, bool outputRawScore, out List<Neuron[]> forwardCellList, out List<Neuron[]> backwardCellList)
        {
            layerList = new List<float[][]>();
            forwardCellList = new List<Neuron[]>();
            backwardCellList = new List<Neuron[]>();

            Neuron[] forwardCell;
            Neuron[] backwardCell;
            var layerOutputs = ComputeBottomLayer(pSequence, forwardHiddenLayers[0], backwardHiddenLayers[0], runningMode, out forwardCell, out backwardCell);
            if (runningMode == RunningMode.Training)
            {
                layerList.Add(layerOutputs);
                forwardCellList.Add(forwardCell);
                backwardCellList.Add(backwardCell);
            }

            for (var i = 1; i < forwardHiddenLayers.Count; i++)
            {
                layerOutputs = ComputeMiddleLayers(pSequence, layerOutputs, forwardHiddenLayers[i], backwardHiddenLayers[i], runningMode, out forwardCell, out backwardCell);
                if (runningMode == RunningMode.Training)
                {
                    layerList.Add(layerOutputs);
                    forwardCellList.Add(forwardCell);
                    backwardCellList.Add(backwardCell);
                }
            }

            return ComputeTopLayer(pSequence, layerOutputs, out rawOutputLayer, runningMode, outputRawScore);
        }

        /// <summary>
        ///     Pass error from the last layer to the first layer
        /// </summary>
        /// <param name="pSequence"></param>
        /// <param name="seqFinalOutput"></param>
        /// <returns></returns>
        private void ComputeDeepErr(Sequence pSequence, SimpleLayer[] seqFinalOutput, out List<float[][]> forwardErrLayers,
            out List<float[][]> backwardErrLayers, List<Neuron[]> forwardNeuronsList, List<Neuron[]> backwardNeuronsList)
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
            List<float[][]> fErrLayers = new List<float[][]>();
            List<float[][]> bErrLayers = new List<float[][]>();
            for (var i = 0; i < numLayers; i++)
            {
                fErrLayers.Add(null);
                bErrLayers.Add(null);
            }

            //Pass error from i+1 to i layer
            var forwardLayer = forwardHiddenLayers[numLayers - 1];
            var backwardLayer = backwardHiddenLayers[numLayers - 1];

            var errLayer1 = new float[numStates][];
            var errLayer2 = new float[numStates][];
            var fNeuron = forwardNeuronsList[numLayers - 1];
            var bNeuron = backwardNeuronsList[numLayers - 1];

            Parallel.Invoke(() =>
            {
                Parallel.For(0, numStates, parallelOption, curState =>
                {
                    var curState2 = numStates - curState - 1;
                    errLayer1[curState2] = new float[forwardLayer.LayerSize];
                    forwardLayer.ComputeLayerErr(seqFinalOutput[curState2], errLayer1[curState2], seqFinalOutput[curState2].Errs, fNeuron[curState2]);
                });

            },
            () =>
            {
                Parallel.For(0, numStates, parallelOption, curState =>
                {         
                    errLayer2[curState] = new float[backwardLayer.LayerSize];
                    backwardLayer.ComputeLayerErr(seqFinalOutput[curState], errLayer2[curState], seqFinalOutput[curState].Errs, bNeuron[curState]);
                });

            });


            fErrLayers[numLayers - 1] = errLayer1;
            bErrLayers[numLayers - 1] = errLayer2;

            Parallel.Invoke(() =>
            {
                // Forward
                for (var i = numLayers - 2; i >= 0; i--)
                {
                    forwardLayer = forwardHiddenLayers[i];
                    var lastForwardLayer = forwardHiddenLayers[i + 1];
                    var errLayer = new float[numStates][];
                    var srcErrLayer = fErrLayers[i + 1];
                    var neurons = forwardNeuronsList[i];
                    Parallel.For(0, numStates, parallelOption, curState =>
                    {
                        var curState2 = numStates - curState - 1;

                        errLayer[curState2] = new float[forwardLayer.LayerSize];
                        forwardLayer.ComputeLayerErr(lastForwardLayer, errLayer[curState2], srcErrLayer[curState2], neurons[curState2]);
                    });

                    fErrLayers[i] = errLayer;
                }
            },
            () =>
            {
                // Backward
                for (var i = numLayers - 2; i >= 0; i--)
                {
                    backwardLayer = backwardHiddenLayers[i];
                    var lastBackwardLayer = backwardHiddenLayers[i + 1];
                    var errLayer = new float[numStates][];
                    var srcErrLayer = bErrLayers[i + 1];
                    var neurons = backwardNeuronsList[i];
                    Parallel.For(0, numStates, parallelOption, curState =>
                    {
                        errLayer[curState] = new float[backwardLayer.LayerSize];
                        backwardLayer.ComputeLayerErr(lastBackwardLayer, errLayer[curState], srcErrLayer[curState], neurons[curState]);
                    });

                    bErrLayers[i] = errLayer;
                }
            });

            forwardErrLayers = fErrLayers;
            backwardErrLayers = bErrLayers;
        }

        private void DeepLearningNet(Sequence pSequence, SimpleLayer[] seqOutput, List<float[][]> fErrLayers,
            List<float[][]> bErrLayers, List<float[][]> layerOutputs, List<Neuron[]> forwardNeuronsList, List<Neuron[]> backwardNeuronsList)
        {
            var numStates = pSequence.States.Length;
            var numLayers = forwardHiddenLayers.Count;

            //Learning output layer
            Parallel.Invoke(() =>
            {
                for (var curState = 0; curState < numStates; curState++)
                {
                    seqOutput[curState].BackwardPass();
                }
            },
                () =>
                {
                    Parallel.For(0, numLayers, parallelOption, i =>
                    {
                        float[][] layerOutputs_i = (i > 0) ? layerOutputs[i - 1] : null;

                        Parallel.Invoke(() =>
                        {
                            Neuron[] forwardNeurons = forwardNeuronsList[i];
                            float[][] forwardErrs = fErrLayers[i];
                            var forwardLayer = forwardHiddenLayers[i];
                            forwardLayer.Reset();
                            for (var curState = 0; curState < numStates; curState++)
                            {
                                State state = pSequence.States[curState];

                                forwardLayer.SparseFeature = state.SparseFeature;
                                forwardLayer.DenseFeature = (i == 0) ? state.DenseFeature.CopyTo() : layerOutputs_i[curState];
                                forwardLayer.PreUpdateWeights(forwardNeurons[curState], forwardErrs[curState]);
                                forwardLayer.BackwardPass();
                            }
                        },
                            () =>
                            {
                                Neuron[] backwardNeurons = backwardNeuronsList[i];
                                float[][] backwardErrs = bErrLayers[i];
                                var backwardLayer = backwardHiddenLayers[i];
                                backwardLayer.Reset();
                                for (var curState = 0; curState < numStates; curState++)
                                {
                                    var curState2 = numStates - curState - 1;
                                    State state = pSequence.States[curState2];
 
                                    backwardLayer.SparseFeature = state.SparseFeature;
                                    backwardLayer.DenseFeature = (i == 0) ? state.DenseFeature.CopyTo() : layerOutputs_i[curState2];
                                    backwardLayer.PreUpdateWeights(backwardNeurons[curState2], backwardErrs[curState2]);
                                    backwardLayer.BackwardPass();
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
            List<float[][]> layerList;
            List<Neuron[]> forwardCellList;
            List<Neuron[]> backwardCellList;

            //Forward process from bottom layer to top layer
            int[] seqBestOutput;
            var seqOutput = ComputeLayers(pSequence, runningMode, out layerList,
                out rawOutputLayer, outputRawScore, out forwardCellList, out backwardCellList);

            //Get best output result of each state
            var numStates = pSequence.States.Length;
            seqBestOutput = new int[numStates];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                seqBestOutput[curState] = seqOutput[curState].GetBestOutputIndex();
            });

            if (runningMode != RunningMode.Test)
            {
                for (var curState = 0; curState < numStates; curState++)
                {
                    logp += Math.Log10(seqOutput[curState].Cells[pSequence.States[curState].Label] + 0.0001);
                }
            }

            if (runningMode == RunningMode.Training)
            {
                //In training mode, we calculate each layer's error and update their net weights
                List<float[][]> fErrLayers;
                List<float[][]> bErrLayers;
                ComputeDeepErr(pSequence, seqOutput, out fErrLayers, out bErrLayers, forwardCellList, backwardCellList);
                DeepLearningNet(pSequence, seqOutput, fErrLayers, bErrLayers, layerList, forwardCellList, backwardCellList);
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
            List<float[][]> layerList;
            Matrix<float> rawOutputLayer;
            List<Neuron[]> forwardCellList;
            List<Neuron[]> backwardCellList;

            var seqOutput = ComputeLayers(pSequence, runningMode, out layerList,
                out rawOutputLayer, true, out forwardCellList, out backwardCellList);

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
                ComputeDeepErr(pSequence, seqOutput, out fErrLayers, out bErrLayers, forwardCellList, backwardCellList);
                DeepLearningNet(pSequence, seqOutput, fErrLayers, bErrLayers, layerList, forwardCellList, backwardCellList);
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

                IsCRFTraining = br.ReadBoolean();
                var layerSize = br.ReadInt32();
                LayerType layerType = LayerType.None;

                //Load forward layers from file
                forwardHiddenLayers = new List<SimpleLayer>();
                for (var i = 0; i < layerSize; i++)
                {
                    layerType = (LayerType)br.ReadInt32();
                    forwardHiddenLayers.Add(Load(layerType, br));
                }

                //Load backward layers from file
                backwardHiddenLayers = new List<SimpleLayer>();
                for (var i = 0; i < layerSize; i++)
                {
                    layerType = (LayerType)br.ReadInt32();
                    backwardHiddenLayers.Add(Load(layerType, br));
                }

                Logger.WriteLine("Create output layer");
                layerType = (LayerType)br.ReadInt32();
                OutputLayer = Load(layerType, br);

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