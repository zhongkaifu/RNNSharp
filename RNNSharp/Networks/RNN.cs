using AdvUtils;
using RNNSharp.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;

namespace RNNSharp.Networks
{
    public abstract class RNN<T> where T : ISequence
    {
        public virtual bool IsCRFTraining { get; set; }
        public virtual bool bVQ { get; set; }

        protected Matrix<float> CRFSeqOutput;
        protected Matrix<float> CRFWeights { get; set; }
        protected IOutputLayer OutputLayer { get; set; }

        private Matrix<float> crfWeightsDelta;

        private double[][] alphaSet;
        private double[][] betaSet;

        public int MaxSeqLength = 1024;

        public abstract void CreateNetwork(List<LayerConfig> hiddenLayersConfig, LayerConfig outputLayerConfig, DataSet<T> TrainingSet, Config featurizer);

        public abstract int[] ProcessSequenceCRF(Sequence pSequence, RunningMode runningMode);

        public abstract int[] ProcessSequence(ISequence sequence, RunningMode runningMode, bool outputRawScore, out Matrix<float> m);

        public abstract int[] ProcessSequence(ISentence sentence, Config featurizer, RunningMode runningMode, bool outputRawScore, out Matrix<float> m);

        public abstract void CleanStatusForTraining();

        public abstract void SaveModel(string filename);

        public abstract void LoadModel(string filename, bool bTrain = false);

        public abstract float[][] ComputeTopHiddenLayerOutput(Sequence pSequence);

        public abstract int GetTopHiddenLayerSize();

        public abstract RNN<T> Clone();


        public virtual void UpdateWeights()
        {

        }

        public static RNN<T> CreateRNN(NETWORKTYPE networkType)
        {
            RNN<T> rnn = null;
            switch (networkType)
            {
                case NETWORKTYPE.Forward:
                    rnn = new ForwardRNN<T>();
                    break;
                case NETWORKTYPE.ForwardSeq2Seq:
                    rnn = new ForwardRNNSeq2Seq<T>();
                    break;
                case NETWORKTYPE.BiDirectional:
                    rnn = new BiRNN<T>();
                    break;
                case NETWORKTYPE.BiDirectionalAverage:
                    rnn = new BiRNNAvg<T>();
                    break;

            }
            return rnn;
        }

        protected IOutputLayer CreateOutputLayer(LayerConfig outputLayerConfig, int sparseFeatureSize, int denseFeatureSize)
        {
            IOutputLayer outputLayer = null;
            switch (outputLayerConfig.LayerType)
            {
                case LayerType.SampledSoftmax:
                    Logger.WriteLine("Create sampled softmax layer as output layer");
                    outputLayer = new SampledSoftmaxLayer(outputLayerConfig as SampledSoftmaxLayerConfig);
                    outputLayer.InitializeWeights(0, denseFeatureSize);
                    break;

                case LayerType.Softmax:
                    Logger.WriteLine("Create softmax layer as output layer.");
                    outputLayer = new SoftmaxLayer(outputLayerConfig as SoftmaxLayerConfig);
                    outputLayer.InitializeWeights(sparseFeatureSize, denseFeatureSize);
                    break;

                case LayerType.Simple:
                    Logger.WriteLine("Create simple layer as output layer.");
                    outputLayer = new SimpleLayer(outputLayerConfig as SimpleLayerConfig);
                    outputLayer.InitializeWeights(sparseFeatureSize, denseFeatureSize);
                    break;
            }
            outputLayer.LabelShortList = new List<int>();

            return outputLayer;
        }

        protected virtual List<ILayer> CreateLayers(List<LayerConfig> hiddenLayersConfig)
        {
            var hiddenLayers = new List<ILayer>();
            for (var i = 0; i < hiddenLayersConfig.Count; i++)
            {
                ILayer layer = null;
                switch (hiddenLayersConfig[i].LayerType)
                {
                    case LayerType.LSTM:
                        layer = new LSTMLayer(hiddenLayersConfig[i] as LSTMLayerConfig);
                        Logger.WriteLine("Create LSTM layer.");
                        break;

                    case LayerType.DropOut:
                        layer = new DropoutLayer(hiddenLayersConfig[i] as DropoutLayerConfig);
                        Logger.WriteLine("Create Dropout layer.");
                        break;
                }

                hiddenLayers.Add(layer);
            }

            return hiddenLayers;
        }

        public void SetRuntimeFeatures(State state, int curState, int numStates, int[] predicted, bool forward = true)
        {
            if (predicted != null && state.RuntimeFeatures != null)
            {
                // set runtime feature
                for (var i = 0; i < state.RuntimeFeatures.Length; i++)
                {
                    for (var j = 0; j < OutputLayer.LayerSize; j++)
                    {
                        //Clean up run time feature value and then set a new one
                        state.SetRuntimeFeature(i, j, 0);
                    }

                    var pos = curState + (forward ? 1 : -1) * state.RuntimeFeatures[i].OffsetToCurrentState;
                    if (pos >= 0 && pos < numStates)
                    {
                        state.SetRuntimeFeature(i, predicted[pos], 1);
                    }
                }
            }
        }

        public void ForwardBackward(int numStates, Matrix<float> m_RawOutput)
        {
            //forward
            var OutputLayerSize = OutputLayer.LayerSize;

            for (var i = 0; i < numStates; i++)
            {
                for (var j = 0; j < OutputLayerSize; j++)
                {
                    double dscore0 = 0;
                    if (i > 0)
                    {
                        for (var k = 0; k < OutputLayerSize; k++)
                        {
                            var fbgm = CRFWeights[j][k];
                            var finit = alphaSet[i - 1][k];
                            var ftmp = fbgm + finit;

                            dscore0 = MathUtil.logsumexp(dscore0, ftmp, k == 0);
                        }
                    }
                    alphaSet[i][j] = dscore0 + m_RawOutput[i][j];
                }
            }

            //backward
            for (var i = numStates - 1; i >= 0; i--)
            {
                for (var j = 0; j < OutputLayerSize; j++)
                {
                    double dscore0 = 0;
                    if (i < numStates - 1)
                    {
                        for (var k = 0; k < OutputLayerSize; k++)
                        {
                            var fbgm = CRFWeights[k][j];
                            var finit = betaSet[i + 1][k];
                            var ftmp = fbgm + finit;

                            dscore0 = MathUtil.logsumexp(dscore0, ftmp, k == 0);
                        }
                    }
                    betaSet[i][j] = dscore0 + m_RawOutput[i][j];
                }
            }


            //Z_
            double Z_ = 0.0f;
            var betaSet_0 = betaSet[0];
            for (var i = 0; i < OutputLayerSize; i++)
            {
                Z_ = MathUtil.logsumexp(Z_, betaSet_0[i], i == 0);
            }

            //Calculate the output probability of each node
            for (var i = 0; i < numStates; i++)
            {
                var CRFSeqOutput_i = CRFSeqOutput[i];
                var alphaSet_i = alphaSet[i];
                var betaSet_i = betaSet[i];
                var m_RawOutput_i = m_RawOutput[i];
                for (var j = 0; j < OutputLayerSize; j++)
                {
                    CRFSeqOutput_i[j] = (float)Math.Exp(alphaSet_i[j] + betaSet_i[j] - m_RawOutput_i[j] - Z_);
                }
            }
        }

        public int[] Viterbi(Matrix<float> ys, int seqLen)
        {
            var OutputLayerSize = OutputLayer.LayerSize;

            var vPath = new int[seqLen, OutputLayerSize];

            var vPreAlpha = new float[OutputLayerSize];
            var vAlpha = new float[OutputLayerSize];

            var nStartTagIndex = 0;
            //viterbi algorithm
            for (var i = 0; i < OutputLayerSize; i++)
            {
                vPreAlpha[i] = ys[0][i];
                if (i != nStartTagIndex)
                    vPreAlpha[i] += float.MinValue;
                vPath[0, i] = nStartTagIndex;
            }
            for (var t = 0; t < seqLen; t++)
            {
                for (var j = 0; j < OutputLayerSize; j++)
                {
                    vPath[t, j] = 0;
                    var CRFTagTransWeights_j = CRFWeights[j];
                    var ys_t = ys[t];
                    var maxScore = float.MinValue;
                    for (var i = 0; i < OutputLayerSize; i++)
                    {
                        var score = vPreAlpha[i] + CRFTagTransWeights_j[i] + ys_t[j];
                        if (score > maxScore)
                        {
                            maxScore = score;
                            vPath[t, j] = i;
                        }
                    }

                    vAlpha[j] = maxScore;
                }
                vPreAlpha = vAlpha;
                vAlpha = new float[OutputLayerSize];
            }

            //backtrace to get the best result path
            var tagOutputs = new int[seqLen];
            tagOutputs[seqLen - 1] = nStartTagIndex;
            var nNextTag = tagOutputs[seqLen - 1];
            for (var t = seqLen - 2; t >= 0; t--)
            {
                tagOutputs[t] = vPath[t + 1, nNextTag];
                nNextTag = tagOutputs[t];
            }

            return tagOutputs;
        }

        public void InitializeCRFWeights(List<List<float>> m)
        {
            var OutputLayerSize = OutputLayer.LayerSize;
            CRFWeights = new Matrix<float>(OutputLayerSize, OutputLayerSize);
            for (var i = 0; i < OutputLayerSize; i++)
            {
                for (var j = 0; j < OutputLayerSize; j++)
                {
                    CRFWeights[i][j] = m[i][j];
                }
            }
        }

        public void InitializeCRFVariablesForTraining()
        {
            var OutputLayerSize = OutputLayer.LayerSize;
            CRFSeqOutput = new Matrix<float>(MaxSeqLength, OutputLayerSize);

            alphaSet = new double[MaxSeqLength][];
            betaSet = new double[MaxSeqLength][];
            for (var i = 0; i < MaxSeqLength; i++)
            {
                alphaSet[i] = new double[OutputLayerSize];
                betaSet[i] = new double[OutputLayerSize];
            }

            crfWeightsDelta = new Matrix<float>(OutputLayerSize, OutputLayerSize);
        }

        //  private static object lockerCRF = new object();
        public void UpdateBigramTransition(Sequence seq)
        {
            var OutputLayerSize = OutputLayer.LayerSize;
            var numStates = seq.States.Length;

            crfWeightsDelta.Clean();

            for (var timeat = 1; timeat < numStates; timeat++)
            {
                var CRFSeqOutput_timeat = CRFSeqOutput[timeat];
                var CRFSeqOutput_pre_timeat = CRFSeqOutput[timeat - 1];
                for (var i = 0; i < OutputLayerSize; i++)
                {
                    var CRFSeqOutput_timeat_i = CRFSeqOutput_timeat[i];
                    var CRFTagTransWeights_i = CRFWeights[i];
                    var crfWeightsDelta_i = crfWeightsDelta[i];
                    var j = 0;

                    var vecCRFSeqOutput_timeat_i = new Vector<float>(CRFSeqOutput_timeat_i);

                    var moreItems = (OutputLayerSize % Vector<float>.Count);
                    while (j < OutputLayerSize - moreItems)
                    {
                        var v1 = new Vector<float>(CRFTagTransWeights_i, j);
                        var v2 = new Vector<float>(CRFSeqOutput_pre_timeat, j);
                        var v = new Vector<float>(crfWeightsDelta_i, j);

                        v -= v1 * vecCRFSeqOutput_timeat_i * v2;
                        v.CopyTo(crfWeightsDelta_i, j);

                        j += Vector<float>.Count;
                    }

                    while (j < OutputLayerSize)
                    {
                        crfWeightsDelta_i[j] -= CRFTagTransWeights_i[j] * CRFSeqOutput_timeat_i * CRFSeqOutput_pre_timeat[j];
                        j++;
                    }
                }

                var iTagId = seq.States[timeat].Label;
                var iLastTagId = seq.States[timeat - 1].Label;
                crfWeightsDelta[iTagId][iLastTagId] += 1;
            }

            //Update tag Bigram LM

            for (var i = 0; i < OutputLayerSize; i++)
            {
                var CRFWeights_i = CRFWeights[i];
                var CRFWeightsDelta_i = crfWeightsDelta[i];
                var j = 0;

                var moreItems = (OutputLayerSize % Vector<float>.Count);
                while (j < OutputLayerSize - moreItems)
                {
                    var vecDelta = new Vector<float>(CRFWeightsDelta_i, j);
                    //Normalize delta
                    vecDelta = RNNHelper.NormalizeGradient(vecDelta);

                    //Update weights
                    var vecWeights = new Vector<float>(CRFWeights_i, j);
                    vecWeights += RNNHelper.vecNormalLearningRate * vecDelta;
                    vecWeights.CopyTo(CRFWeights_i, j);

                    j += Vector<float>.Count;
                }

                while (j < OutputLayerSize)
                {
                    var delta = CRFWeightsDelta_i[j];
                    delta = RNNHelper.NormalizeGradient(delta);
                    CRFWeights_i[j] += RNNHelper.LearningRate * delta;
                    j++;
                }
            }
        }

        public int[][] DecodeNBestCRF(Sentence sent, Config config, int N)
        {
            //ys contains the output of RNN for each word
            Matrix<float> ys;
            ProcessSequence(sent, config, RunningMode.Test, true, out ys);

            var n = sent.TokensList.Count;
            var K = OutputLayer.LayerSize;
            var STP = CRFWeights;
            var vPath = new PAIR<int, int>[n, K, N];
            var DUMP_LABEL = -1;
            var vPreAlpha = new float[K, N];
            var vAlpha = new float[K, N];

            var nStartTagIndex = 0;
            var nEndTagIndex = 0;
            var MIN_VALUE = float.MinValue;

            //viterbi algorithm
            for (var i = 0; i < K; i++)
            {
                for (var j = 0; j < N; j++)
                {
                    vPreAlpha[i, j] = MIN_VALUE;
                    vPath[0, i, j] = new PAIR<int, int>(DUMP_LABEL, 0);
                }
            }
            vPreAlpha[nStartTagIndex, 0] = ys[0][nStartTagIndex];
            vPath[0, nStartTagIndex, 0].first = nStartTagIndex;

            var q = new PriorityQueue<float, PAIR<int, int>>();

            for (var t = 1; t < n; t++)
            {
                for (var j = 0; j < K; j++)
                {
                    while (q.Count() > 0)
                        q.Dequeue();
                    var _stp = STP[j][0];
                    var _y = ys[t][j];
                    for (var k = 0; k < N; k++)
                    {
                        var score = vPreAlpha[0, k] + _stp + _y;
                        q.Enqueue(score, new PAIR<int, int>(0, k));
                    }
                    for (var i = 1; i < K; i++)
                    {
                        _stp = STP[j][i];
                        for (var k = 0; k < N; k++)
                        {
                            var score = vPreAlpha[i, k] + _stp + _y;
                            if (score <= q.Peek().Key)
                                break;
                            q.Dequeue();
                            q.Enqueue(score, new PAIR<int, int>(i, k));
                        }
                    }
                    var idx = N - 1;
                    while (q.Count() > 0)
                    {
                        vAlpha[j, idx] = q.Peek().Key;
                        vPath[t, j, idx] = q.Peek().Value;
                        idx--;
                        q.Dequeue();
                    }
                }
                vPreAlpha = vAlpha;
                vAlpha = new float[K, N];
            }

            //backtrace to get the n-best result path
            var vTagOutput = new int[N][];
            for (var i = 0; i < N; i++)
            {
                vTagOutput[i] = new int[n];
            }

            for (var k = 0; k < N; k++)
            {
                vTagOutput[k][n - 1] = nEndTagIndex;
                var decision = new PAIR<int, int>(nEndTagIndex, k);
                for (var t = n - 2; t >= 0; t--)
                {
                    vTagOutput[k][t] = vPath[t + 1, decision.first, decision.second].first;
                    decision = vPath[t + 1, decision.first, decision.second];
                }
            }

            return vTagOutput;
        }

        public int[] DecodeNN(Sentence sent, Config config)
        {
            Matrix<float> ys;
            return ProcessSequence(sent, config, RunningMode.Test, false, out ys);
        }

        public int[] DecodeCRF(Sentence sent, Config config)
        {
            //ys contains the output of RNN for each word
            Matrix<float> ys;
            ProcessSequence(sent, config, RunningMode.Test, true, out ys);
            return Viterbi(ys, sent.TokensList.Count);
        }

        public static ILayer Load(LayerType layerType, BinaryReader br, bool forTraining = false)
        {
            ILayer layer = null;
            switch (layerType)
            {
                case LayerType.LSTM:
                    layer = new LSTMLayer();
                    break;
                case LayerType.DropOut:
                    layer = new DropoutLayer();
                    break;
                case LayerType.Softmax:
                    layer = new SoftmaxLayer();
                    break;
                case LayerType.SampledSoftmax:
                    layer = new SampledSoftmaxLayer();
                    break;
                case LayerType.Simple:
                    layer = new SimpleLayer();
                    break;
            }

            layer.Load(br, layerType, forTraining);

            return layer;
        }
    }
}