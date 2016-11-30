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

    public enum MODELTYPE
    {
        SEQLABEL = 0,
        SEQ2SEQ
    }

    public enum LAYERTYPE
    {
        BPTT = 0,
        LSTM
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

    public class ForwardRNN<T> : RNN<T> where T : ISequence
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

        public override int GetTopHiddenLayerSize()
        {
            return HiddenLayerList[HiddenLayerList.Count - 1].LayerSize;
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
                SetRuntimeFeatures(state, curState, numStates, null);
                HiddenLayerList[0].computeLayer(state.SparseFeature, state.DenseFeature.CopyTo(), false);

                //Compute each layer
                for (int i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    HiddenLayerList[i].computeLayer(state.SparseFeature, HiddenLayerList[i - 1].cellOutput, false);
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

        public override int[] TestSeq2Seq(Sentence srcSentence, Featurizer featurizer)
        {
            State curState = featurizer.ExtractFeatures(new string[] { "<s>" });
            curState.Label = featurizer.TagSet.GetIndex("<s>");

            //Reset all layers
            foreach (SimpleLayer layer in HiddenLayerList)
            {
                layer.netReset(false);
            }

            //Extract features from source sentence
            Sequence srcSequence = featurizer.AutoEncoder.Featurizer.ExtractFeatures(srcSentence);
            double[] srcHiddenAvgOutput;
            Dictionary<int, float> srcSparseFeatures;
            ExtractSourceSentenceFeature(featurizer.AutoEncoder, srcSequence, curState.SparseFeature.Length, out srcHiddenAvgOutput, out srcSparseFeatures);

            int numLayers = HiddenLayerList.Count;
            List<int> predicted = new List<int>();
            predicted.Add(curState.Label);
            while (true)
            {
                //Build sparse features
                SparseVector sparseVector = new SparseVector();
                sparseVector.SetLength(curState.SparseFeature.Length + srcSequence.SparseFeatureSize);
                sparseVector.AddKeyValuePairData(curState.SparseFeature);
                sparseVector.AddKeyValuePairData(srcSparseFeatures);

                //Compute first layer
                double[] denseFeatures = RNNHelper.ConcatenateVector(curState.DenseFeature, srcHiddenAvgOutput);
                HiddenLayerList[0].computeLayer(sparseVector, denseFeatures, false);

                //Compute middle layers
                for (int i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    denseFeatures = RNNHelper.ConcatenateVector(HiddenLayerList[i - 1].cellOutput, srcHiddenAvgOutput);
                    HiddenLayerList[i].computeLayer(sparseVector, denseFeatures, false);
                }

                //Compute output layer
                denseFeatures = RNNHelper.ConcatenateVector(HiddenLayerList[numLayers - 1].cellOutput, srcHiddenAvgOutput);
                OutputLayer.computeLayer(sparseVector, denseFeatures, false);

                OutputLayer.Softmax(false);

                int nextTagId = OutputLayer.GetBestOutputIndex(false);
                string nextWord = featurizer.TagSet.GetTagName(nextTagId);

                curState = featurizer.ExtractFeatures(new string[] { nextWord });
                curState.Label = nextTagId;

                predicted.Add(nextTagId);

                if (nextWord == "</s>" || predicted.Count >= 100)
                {
                    break;
                }
            }

            return predicted.ToArray();
        }


        private void ExtractSourceSentenceFeature(RNNDecoder decoder, Sequence srcSequence, int targetSparseFeatureSize, out double[] srcHiddenAvgOutput, out Dictionary<int, float> srcSparseFeatures)
        {
            List<double[]> srcOutputs = decoder.ComputeTopHiddenLayerOutput(srcSequence);
            srcHiddenAvgOutput = new double[srcOutputs[0].Length];
            for (int i = 0; i < srcOutputs[0].Length; i++)
            {
                srcHiddenAvgOutput[i] = (srcOutputs[0][i] + srcOutputs[srcOutputs.Count - 1][i]) / 2.0;
            }

            srcSparseFeatures = new Dictionary<int, float>();
            for (int i = 0; i < srcSequence.States.Length; i++)
            {
                foreach (KeyValuePair<int, float> kv in srcSequence.States[i].SparseFeature)
                {
                    int srcSparseFeatureIndex = kv.Key + targetSparseFeatureSize;

                    if (srcSparseFeatures.ContainsKey(srcSparseFeatureIndex) == false)
                    {
                        srcSparseFeatures.Add(srcSparseFeatureIndex, kv.Value);
                    }
                    else
                    {
                        srcSparseFeatures[srcSparseFeatureIndex] += kv.Value;
                    }
                }
            }
        }

        public override int[] ProcessSeq2Seq(SequencePair pSequence, RunningMode runningMode)
        {
            Sequence tgtSequence = pSequence.tgtSequence;
            bool isTraining = true;
            if (runningMode == RunningMode.Training)
            {
                isTraining = true;
            }
            else
            {
                isTraining = false;
            }

            //Reset all layers
            foreach (SimpleLayer layer in HiddenLayerList)
            {
                layer.netReset(isTraining);
            }

            //Extract features from source sentences
            Sequence srcSequence = pSequence.autoEncoder.Featurizer.ExtractFeatures(pSequence.srcSentence);
            double[] srcHiddenAvgOutput;
            Dictionary<int, float> srcSparseFeatures;
            ExtractSourceSentenceFeature(pSequence.autoEncoder, srcSequence, tgtSequence.SparseFeatureSize, out srcHiddenAvgOutput, out srcSparseFeatures);

            int numStates = pSequence.tgtSequence.States.Length;
            int numLayers = HiddenLayerList.Count;
            int[] predicted = new int[numStates];

            //Set target sentence labels into short list in output layer
            OutputLayer.LabelShortList = new List<int>();
            foreach (State state in tgtSequence.States)
            {
                OutputLayer.LabelShortList.Add(state.Label);
            }

            for (int curState = 0; curState < numStates; curState++)
            {
                //Build runtime features
                State state = tgtSequence.States[curState];
                SetRuntimeFeatures(state, curState, numStates, predicted);

                //Build sparse features for all layers
                SparseVector sparseVector = new SparseVector();
                sparseVector.SetLength(tgtSequence.SparseFeatureSize + srcSequence.SparseFeatureSize);
                sparseVector.AddKeyValuePairData(state.SparseFeature);
                sparseVector.AddKeyValuePairData(srcSparseFeatures);

                //Compute first layer
                double[] denseFeatures = RNNHelper.ConcatenateVector(state.DenseFeature, srcHiddenAvgOutput);
                HiddenLayerList[0].computeLayer(sparseVector, denseFeatures, isTraining);

                //Compute middle layers
                for (int i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    denseFeatures = RNNHelper.ConcatenateVector(HiddenLayerList[i - 1].cellOutput, srcHiddenAvgOutput);
                    HiddenLayerList[i].computeLayer(sparseVector, denseFeatures, isTraining);
                }

                //Compute output layer
                denseFeatures = RNNHelper.ConcatenateVector(HiddenLayerList[numLayers - 1].cellOutput, srcHiddenAvgOutput);
                OutputLayer.computeLayer(sparseVector, denseFeatures, isTraining);

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

            //Set current sentence labels into short list in output layer
            OutputLayer.LabelShortList = new List<int>();
            foreach (State state in pSequence.States)
            {
                OutputLayer.LabelShortList.Add(state.Label);
            }

            for (int curState = 0; curState < numStates; curState++)
            {
                //Compute first layer
                State state = pSequence.States[curState];
                SetRuntimeFeatures(state, curState, numStates, predicted);
                HiddenLayerList[0].computeLayer(state.SparseFeature, state.DenseFeature.CopyTo(), isTraining);

                //Compute each layer
                for (int i = 1; i < numLayers; i++)
                {
                    //We use previous layer's output as dense feature for current layer
                    HiddenLayerList[i].computeLayer(state.SparseFeature, HiddenLayerList[i - 1].cellOutput, isTraining);
                }

                //Compute output layer
                OutputLayer.computeLayer(state.SparseFeature, HiddenLayerList[numLayers - 1].cellOutput, isTraining);

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
                    SetRuntimeFeatures(state, curState, numStates, null);
                    HiddenLayerList[0].computeLayer(state.SparseFeature, state.DenseFeature.CopyTo());

                    for (int i = 1; i < numLayers; i++)
                    {
                        HiddenLayerList[i].computeLayer(state.SparseFeature, HiddenLayerList[i - 1].cellOutput);
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
                fo.Write((int)LAYERTYPE.BPTT);
            }
            else
            {
                fo.Write((int)LAYERTYPE.LSTM);
            }

            fo.Write((int)ModelDirection);
            fo.Write((int)ModelType);
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

            LAYERTYPE layerType = (LAYERTYPE)br.ReadInt32();
            ModelDirection = (MODELDIRECTION)br.ReadInt32();
            ModelType = (MODELTYPE)br.ReadInt32();
            IsCRFTraining = br.ReadBoolean();

            //Create cells of each layer
            int layerSize = br.ReadInt32();
            HiddenLayerList = new List<SimpleLayer>();
            for (int i = 0; i < layerSize; i++)
            {
                SimpleLayer layer = null;
                if (layerType == LAYERTYPE.BPTT)
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

            Logger.WriteLine("Create output layer");
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
