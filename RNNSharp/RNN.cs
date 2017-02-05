using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;

namespace RNNSharp
{
    public abstract class RNN<T> where T : ISequence
    {
        protected Matrix<float> CRFSeqOutput;
        public double logp;
        protected double minTknErrRatio = double.MaxValue;

        protected ParallelOptions parallelOption = new ParallelOptions();
        public virtual bool IsCRFTraining { get; set; }
        public virtual bool bVQ { get; set; }

        public virtual int MaxIter { get; set; }
        public virtual long SaveStep { get; set; }
        public Matrix<float> CRFTagTransWeights { get; set; }

        public SimpleLayer OutputLayer { get; set; }

        public abstract int[] ProcessSequenceCRF(Sequence pSequence, RunningMode runningMode);

        public abstract int[] ProcessSequence(Sequence pSequence, RunningMode runningMode, bool outputRawScore,
            out Matrix<float> m);

        public abstract int[] ProcessSeq2Seq(SequencePair pSequence, RunningMode runningMode);

        public abstract int[] TestSeq2Seq(Sentence srcSentence, Config featurizer);

        public abstract void CleanStatus();

        public abstract void SaveModel(string filename);

        public abstract void LoadModel(string filename);

        public abstract float[][] ComputeTopHiddenLayerOutput(Sequence pSequence);

        public abstract int GetTopHiddenLayerSize();

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
            var alphaSet = new double[numStates][];
            var betaSet = new double[numStates][];
            var OutputLayerSize = OutputLayer.LayerSize;

            Parallel.Invoke(() =>
            {
                for (var i = 0; i < numStates; i++)
                {
                    alphaSet[i] = new double[OutputLayerSize];
                    for (var j = 0; j < OutputLayerSize; j++)
                    {
                        double dscore0 = 0;
                        if (i > 0)
                        {
                            for (var k = 0; k < OutputLayerSize; k++)
                            {
                                var fbgm = CRFTagTransWeights[j][k];
                                var finit = alphaSet[i - 1][k];
                                var ftmp = fbgm + finit;

                                dscore0 = MathUtil.logsumexp(dscore0, ftmp, k == 0);
                            }
                        }
                        alphaSet[i][j] = dscore0 + m_RawOutput[i][j];
                    }
                }
            },
                () =>
                {
                    //backward
                    for (var i = numStates - 1; i >= 0; i--)
                    {
                        betaSet[i] = new double[OutputLayerSize];
                        for (var j = 0; j < OutputLayerSize; j++)
                        {
                            double dscore0 = 0;
                            if (i < numStates - 1)
                            {
                                for (var k = 0; k < OutputLayerSize; k++)
                                {
                                    var fbgm = CRFTagTransWeights[k][j];
                                    var finit = betaSet[i + 1][k];
                                    var ftmp = fbgm + finit;

                                    dscore0 = MathUtil.logsumexp(dscore0, ftmp, k == 0);
                                }
                            }
                            betaSet[i][j] = dscore0 + m_RawOutput[i][j];
                        }
                    }
                });

            //Z_
            double Z_ = 0.0f;
            var betaSet_0 = betaSet[0];
            for (var i = 0; i < OutputLayerSize; i++)
            {
                Z_ = MathUtil.logsumexp(Z_, betaSet_0[i], i == 0);
            }

            //Calculate the output probability of each node
            CRFSeqOutput = new Matrix<float>(numStates, OutputLayerSize);
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
                    var CRFTagTransWeights_j = CRFTagTransWeights[j];
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

        public virtual void setTagBigramTransition(List<List<float>> m)
        {
            var OutputLayerSize = OutputLayer.LayerSize;
            CRFTagTransWeights = new Matrix<float>(OutputLayerSize, OutputLayerSize);
            for (var i = 0; i < OutputLayerSize; i++)
            {
                for (var j = 0; j < OutputLayerSize; j++)
                {
                    CRFTagTransWeights[i][j] = m[i][j];
                }
            }
        }

        public void UpdateBigramTransition(Sequence seq)
        {
            var OutputLayerSize = OutputLayer.LayerSize;
            var numStates = seq.States.Length;
            var m_DeltaBigramLM = new Matrix<float>(OutputLayerSize, OutputLayerSize);

            for (var timeat = 1; timeat < numStates; timeat++)
            {
                var CRFSeqOutput_timeat = CRFSeqOutput[timeat];
                var CRFSeqOutput_pre_timeat = CRFSeqOutput[timeat - 1];
                for (var i = 0; i < OutputLayerSize; i++)
                {
                    var CRFSeqOutput_timeat_i = CRFSeqOutput_timeat[i];
                    var CRFTagTransWeights_i = CRFTagTransWeights[i];
                    var m_DeltaBigramLM_i = m_DeltaBigramLM[i];
                    var j = 0;

                    var vecCRFSeqOutput_timeat_i = new Vector<float>(CRFSeqOutput_timeat_i);
                    while (j < OutputLayerSize - Vector<float>.Count)
                    {
                        var v1 = new Vector<float>(CRFTagTransWeights_i, j);
                        var v2 = new Vector<float>(CRFSeqOutput_pre_timeat, j);
                        var v = new Vector<float>(m_DeltaBigramLM_i, j);

                        v -= v1 * vecCRFSeqOutput_timeat_i * v2;
                        v.CopyTo(m_DeltaBigramLM_i, j);

                        j += Vector<float>.Count;
                    }

                    while (j < OutputLayerSize)
                    {
                        m_DeltaBigramLM_i[j] -= CRFTagTransWeights_i[j] * CRFSeqOutput_timeat_i * CRFSeqOutput_pre_timeat[j];
                        j++;
                    }
                }

                var iTagId = seq.States[timeat].Label;
                var iLastTagId = seq.States[timeat - 1].Label;
                m_DeltaBigramLM[iTagId][iLastTagId] += 1;
            }

            //Update tag Bigram LM
            for (var b = 0; b < OutputLayerSize; b++)
            {
                var vector_b = CRFTagTransWeights[b];
                var vector_delta_b = m_DeltaBigramLM[b];
                var a = 0;

                while (a < OutputLayerSize - Vector<float>.Count)
                {
                    var v1 = new Vector<float>(vector_delta_b, a);
                    var v = new Vector<float>(vector_b, a);

                    //Normalize delta
                    v1 = RNNHelper.NormalizeGradient(v1);

                    //Update weights
                    v += RNNHelper.vecNormalLearningRate * v1;
                    v.CopyTo(vector_b, a);

                    a += Vector<float>.Count;
                }

                while (a < OutputLayerSize)
                {
                    vector_b[a] += RNNHelper.LearningRate * RNNHelper.NormalizeGradient(vector_delta_b[a]);
                    a++;
                }
            }
        }

        public double TrainNet(DataSet<T> trainingSet, int iter)
        {
            var start = DateTime.Now;
            Logger.WriteLine("Iter " + iter + " begins with learning rate alpha = " + RNNHelper.LearningRate + " ...");

            //Initialize varibles
            logp = 0;

            //Shffle training corpus
            trainingSet.Shuffle();

            var numSequence = trainingSet.SequenceList.Count;
            var wordCnt = 0;
            var tknErrCnt = 0;
            var sentErrCnt = 0;
            Logger.WriteLine("Progress = 0/" + numSequence / 1000.0 + "K\r");
            for (var curSequence = 0; curSequence < numSequence; curSequence++)
            {
                var pSequence = trainingSet.SequenceList[curSequence];

                if (pSequence is Sequence)
                {
                    wordCnt += (pSequence as Sequence).States.Length;
                }
                else
                {
                    wordCnt += (pSequence as SequencePair).tgtSequence.States.Length;
                }

                int[] predicted;
                if (IsCRFTraining)
                {
                    predicted = ProcessSequenceCRF(pSequence as Sequence, RunningMode.Training);
                }
                else if (pSequence is SequencePair)
                {
                    predicted = ProcessSeq2Seq(pSequence as SequencePair, RunningMode.Training);
                }
                else
                {
                    Matrix<float> m;
                    predicted = ProcessSequence(pSequence as Sequence, RunningMode.Training, false, out m);
                }

                int newTknErrCnt;

                if (pSequence is Sequence)
                {
                    newTknErrCnt = GetErrorTokenNum(pSequence as Sequence, predicted);
                }
                else
                {
                    newTknErrCnt = GetErrorTokenNum((pSequence as SequencePair).tgtSequence, predicted);
                }

                tknErrCnt += newTknErrCnt;
                if (newTknErrCnt > 0)
                {
                    sentErrCnt++;
                }

                if ((curSequence + 1) % 1000 == 0)
                {
                    Logger.WriteLine("Progress = {0} ", (curSequence + 1) / 1000 + "K/" + numSequence / 1000.0 + "K");
                    Logger.WriteLine(" Train cross-entropy = {0} ", -logp / Math.Log10(2.0) / wordCnt);
                    Logger.WriteLine(" Error token ratio = {0}%", (double)tknErrCnt / (double)wordCnt * 100.0);
                    Logger.WriteLine(" Error sentence ratio = {0}%", (double)sentErrCnt / (double)curSequence * 100.0);
                }

                if (SaveStep > 0 && (curSequence + 1) % SaveStep == 0)
                {
                    //After processed every m_SaveStep sentences, save current model into a temporary file
                    Logger.WriteLine("Saving temporary model into file...");
                    SaveModel("model.tmp");
                }
            }

            var now = DateTime.Now;
            var duration = now.Subtract(start);

            var entropy = -logp / Math.Log10(2.0) / wordCnt;
            var ppl = exp_10(-logp / wordCnt);
            Logger.WriteLine("Iter " + iter + " completed");
            Logger.WriteLine("Sentences = " + numSequence + ", time escape = " + duration + "s, speed = " +
                             numSequence / duration.TotalSeconds);
            Logger.WriteLine("In training: log probability = " + logp + ", cross-entropy = " + entropy +
                             ", perplexity = " + ppl);

            return ppl;
        }

        public double exp_10(double num)
        {
            return Math.Exp(num * 2.302585093);
        }

        public bool ValidateNet(DataSet<T> validationSet, int iter)
        {
            Logger.WriteLine("Start validation ...");
            var wordcn = 0;
            var tknErrCnt = 0;
            var sentErrCnt = 0;

            //Initialize varibles
            logp = 0;
            var numSequence = validationSet.SequenceList.Count;
            for (var curSequence = 0; curSequence < numSequence; curSequence++)
            {
                var pSequence = validationSet.SequenceList[curSequence];
                if (pSequence is Sequence)
                {
                    wordcn += (pSequence as Sequence).States.Length;
                }
                else
                {
                    wordcn += (pSequence as SequencePair).tgtSequence.States.Length;
                }

                int[] predicted;
                if (IsCRFTraining)
                {
                    predicted = ProcessSequenceCRF(pSequence as Sequence, RunningMode.Validate);
                }
                else if (pSequence is SequencePair)
                {
                    predicted = ProcessSeq2Seq(pSequence as SequencePair, RunningMode.Validate);
                }
                else
                {
                    Matrix<float> m;
                    predicted = ProcessSequence(pSequence as Sequence, RunningMode.Validate, false, out m);
                }

                int newTknErrCnt;
                if (pSequence is Sequence)
                {
                    newTknErrCnt = GetErrorTokenNum(pSequence as Sequence, predicted);
                }
                else
                {
                    newTknErrCnt = GetErrorTokenNum((pSequence as SequencePair).tgtSequence, predicted);
                }

                tknErrCnt += newTknErrCnt;
                if (newTknErrCnt > 0)
                {
                    sentErrCnt++;
                }
            }

            var entropy = -logp / Math.Log10(2.0) / wordcn;
            var ppl = exp_10(-logp / wordcn);
            var tknErrRatio = tknErrCnt / (double)wordcn * 100.0;
            var sentErrRatio = sentErrCnt / (double)numSequence * 100.0;

            Logger.WriteLine("In validation: error token ratio = {0}% error sentence ratio = {1}%", tknErrRatio,
                sentErrRatio);
            Logger.WriteLine("In training: log probability = " + logp + ", cross-entropy = " + entropy +
                             ", perplexity = " + ppl);
            Logger.WriteLine("");

            var bUpdate = false;
            if (tknErrRatio < minTknErrRatio)
            {
                //We have better result on validated set, save this model
                bUpdate = true;
                minTknErrRatio = tknErrRatio;
            }

            return bUpdate;
        }

        private int GetErrorTokenNum(Sequence seq, int[] predicted)
        {
            var tknErrCnt = 0;
            var numStates = seq.States.Length;
            for (var curState = 0; curState < numStates; curState++)
            {
                if (predicted[curState] != seq.States[curState].Label)
                {
                    tknErrCnt++;
                }
            }

            return tknErrCnt;
        }

        public int[][] DecodeNBestCRF(Sequence seq, int N)
        {
            //ys contains the output of RNN for each word
            Matrix<float> ys;
            ProcessSequence(seq, RunningMode.Test, true, out ys);

            var n = seq.States.Length;
            var K = OutputLayer.LayerSize;
            var STP = CRFTagTransWeights;
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

        public int[] DecodeNN(Sequence seq)
        {
            Matrix<float> ys;
            return ProcessSequence(seq, RunningMode.Test, false, out ys);
        }

        public int[] DecodeSeq2Seq(Sentence srcSent, Config feature)
        {
            return TestSeq2Seq(srcSent, feature);
        }

        public int[] DecodeCRF(Sequence seq)
        {
            //ys contains the output of RNN for each word
            Matrix<float> ys;
            ProcessSequence(seq, RunningMode.Test, true, out ys);
            return Viterbi(ys, seq.States.Length);
        }

        public static SimpleLayer Load(LayerType layerType, BinaryReader br)
        {
            switch (layerType)
            {
                case LayerType.LSTM:
                    return LSTMLayer.Load(br, LayerType.LSTM);
                case LayerType.DropOut:
                    return DropoutLayer.Load(br, LayerType.DropOut);
                case LayerType.Softmax:
                    return SoftmaxLayer.Load(br, LayerType.Softmax);
                case LayerType.SampledSoftmax:
                    return SampledSoftmaxLayer.Load(br, LayerType.SampledSoftmax);
                case LayerType.Simple:
                    return SimpleLayer.Load(br, LayerType.Simple);
            }

            return null;
        }
    }
}