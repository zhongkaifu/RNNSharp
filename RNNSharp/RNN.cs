using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    abstract public class RNN<T> where T: ISequence
    {
        public double logp;
        protected double minTknErrRatio = double.MaxValue;
        public virtual bool IsCRFTraining { get; set; }
        public virtual MODELTYPE ModelType { get; set; }
        public virtual string ModelFile { get; set; }
        public string ModelTempFile { get { return ModelFile + ".tmp"; } }
        public virtual MODELDIRECTION ModelDirection { get; set; }
        public virtual bool bVQ { get; set; }

        public virtual int MaxIter { get; set; }
        public virtual long SaveStep { get; set; }
        public Matrix<double> CRFTagTransWeights { get; set; }

        public SimpleLayer OutputLayer { get; set; }

        public abstract int[] ProcessSequenceCRF(Sequence pSequence, RunningMode runningMode);
        public abstract int[] ProcessSequence(Sequence pSequence, RunningMode runningMode, bool outputRawScore, out Matrix<double> m);

        public abstract int[] ProcessSeq2Seq(SequencePair pSequence, RunningMode runningMode);

        public abstract int[] TestSeq2Seq(Sentence srcSentence, Featurizer featurizer);

        public abstract void CleanStatus();
        public abstract void SaveModel(string filename);
        public abstract void LoadModel(string filename);

        public abstract List<double[]> ComputeTopHiddenLayerOutput(Sequence pSequence);
        public abstract int GetTopHiddenLayerSize();


        protected ParallelOptions parallelOption = new ParallelOptions();
        protected Matrix<double> CRFSeqOutput;

        public void SetRuntimeFeatures(State state, int curState, int numStates, int[] predicted, bool forward = true)
        {
            if (predicted != null && state.RuntimeFeatures != null)
            {
                // set runtime feature
                for (int i = 0; i < state.RuntimeFeatures.Length; i++)
                {
                    for (int j = 0; j < OutputLayer.LayerSize; j++)
                    {
                        //Clean up run time feature value and then set a new one
                        state.SetRuntimeFeature(i, j, 0);
                    }

                    int pos = curState + ((forward == true) ? 1 : -1) * state.RuntimeFeatures[i].OffsetToCurrentState;
                    if (pos >= 0 && pos < numStates)
                    {
                        state.SetRuntimeFeature(i, predicted[pos], 1);
                    }
                }
            }
        }

        public void ForwardBackward(int numStates, Matrix<double> m_RawOutput)
        {
            //forward
            double[][] alphaSet = new double[numStates][];
            double[][] betaSet = new double[numStates][];
            int OutputLayerSize = OutputLayer.LayerSize;

            Parallel.Invoke(() =>
            {
                for (int i = 0; i < numStates; i++)
                {
                    alphaSet[i] = new double[OutputLayerSize];
                    for (int j = 0; j < OutputLayerSize; j++)
                    {
                        double dscore0 = 0;
                        if (i > 0)
                        {
                            for (int k = 0; k < OutputLayerSize; k++)
                            {
                                double fbgm = CRFTagTransWeights[j][k];
                                double finit = alphaSet[i - 1][k];
                                double ftmp = fbgm + finit;

                                dscore0 = MathUtil.logsumexp(dscore0, ftmp, (k == 0));
                            }
                        }
                        alphaSet[i][j] = dscore0 + m_RawOutput[i][j];

                    }
                }
            },
            () =>
            {
                //backward
                for (int i = numStates - 1; i >= 0; i--)
                {
                    betaSet[i] = new double[OutputLayerSize];
                    for (int j = 0; j < OutputLayerSize; j++)
                    {
                        double dscore0 = 0;
                        if (i < numStates - 1)
                        {
                            for (int k = 0; k < OutputLayerSize; k++)
                            {
                                double fbgm = CRFTagTransWeights[k][j];
                                double finit = betaSet[i + 1][k];
                                double ftmp = fbgm + finit;

                                dscore0 = MathUtil.logsumexp(dscore0, ftmp, (k == 0));
                            }

                        }
                        betaSet[i][j] = dscore0 + m_RawOutput[i][j];

                    }
                }
            });

            //Z_
            double Z_ = 0.0;
            double[] betaSet_0 = betaSet[0];
            for (int i = 0; i < OutputLayerSize; i++)
            {
                Z_ = MathUtil.logsumexp(Z_, betaSet_0[i], i == 0);

            }

            //Calculate the output probability of each node
            CRFSeqOutput = new Matrix<double>(numStates, OutputLayerSize);
            for (int i = 0; i < numStates; i++)
            {
                double[] CRFSeqOutput_i = CRFSeqOutput[i];
                double[] alphaSet_i = alphaSet[i];
                double[] betaSet_i = betaSet[i];
                double[] m_RawOutput_i = m_RawOutput[i];
                for (int j = 0; j < OutputLayerSize; j++)
                {
                    CRFSeqOutput_i[j] = Math.Exp(alphaSet_i[j] + betaSet_i[j] - m_RawOutput_i[j] - Z_);
                }
            }

        }

        public int[] Viterbi(Matrix<double> ys, int seqLen)
        {
            int OutputLayerSize = OutputLayer.LayerSize;

            int[,] vPath = new int[seqLen, OutputLayerSize];

            double[] vPreAlpha = new double[OutputLayerSize];
            double[] vAlpha = new double[OutputLayerSize];


            int nStartTagIndex = 0;
            //viterbi algorithm
            for (int i = 0; i < OutputLayerSize; i++)
            {
                vPreAlpha[i] = ys[0][i];
                if (i != nStartTagIndex)
                    vPreAlpha[i] += double.MinValue;
                vPath[0, i] = nStartTagIndex;
            }
            for (int t = 0; t < seqLen; t++)
            {
                for (int j = 0; j < OutputLayerSize; j++)
                {
                    vPath[t, j] = 0;
                    double[] CRFTagTransWeights_j = CRFTagTransWeights[j];
                    double[] ys_t = ys[t];
                    double maxScore = double.MinValue;
                    for (int i = 0; i < OutputLayerSize; i++)
                    {
                        double score = vPreAlpha[i] + CRFTagTransWeights_j[i] + ys_t[j];
                        if (score > maxScore)
                        {
                            maxScore = score;
                            vPath[t, j] = i;
                        }
                    }

                    vAlpha[j] = maxScore;
                }
                vPreAlpha = vAlpha;
                vAlpha = new double[OutputLayerSize];
            }

            //backtrace to get the best result path
            int[] tagOutputs = new int[seqLen];
            tagOutputs[seqLen - 1] = nStartTagIndex;
            int nNextTag = tagOutputs[seqLen - 1];
            for (int t = seqLen - 2; t >= 0; t--)
            {
                tagOutputs[t] = vPath[t + 1, nNextTag];
                nNextTag = tagOutputs[t];
            }

            return tagOutputs;
        }

        public virtual void setTagBigramTransition(List<List<float>> m)
        {
            int OutputLayerSize = OutputLayer.LayerSize;
            CRFTagTransWeights = new Matrix<double>(OutputLayerSize, OutputLayerSize);
            for (int i = 0; i < OutputLayerSize; i++)
            {
                for (int j = 0; j < OutputLayerSize; j++)
                {
                    CRFTagTransWeights[i][j] = m[i][j];
                }
            }
        }

        public void UpdateBigramTransition(Sequence seq)
        {
            int OutputLayerSize = OutputLayer.LayerSize;
            int numStates = seq.States.Length;
            Matrix<double> m_DeltaBigramLM = new Matrix<double>(OutputLayerSize, OutputLayerSize);

            for (int timeat = 1; timeat < numStates; timeat++)
            {
                double[] CRFSeqOutput_timeat = CRFSeqOutput[timeat];
                double[] CRFSeqOutput_pre_timeat = CRFSeqOutput[timeat - 1];
                for (int i = 0; i < OutputLayerSize; i++)
                {
                    double CRFSeqOutput_timeat_i = CRFSeqOutput_timeat[i];
                    double[] CRFTagTransWeights_i = CRFTagTransWeights[i];
                    double[] m_DeltaBigramLM_i = m_DeltaBigramLM[i];
                    int j = 0;

                    Vector<double> vecCRFSeqOutput_timeat_i = new Vector<double>(CRFSeqOutput_timeat_i);
                    while (j < OutputLayerSize - Vector<double>.Count)
                    {
                        Vector<double> v1 = new Vector<double>(CRFTagTransWeights_i, j);
                        Vector<double> v2 = new Vector<double>(CRFSeqOutput_pre_timeat, j);
                        Vector<double> v = new Vector<double>(m_DeltaBigramLM_i, j);

                        v -= (v1 * vecCRFSeqOutput_timeat_i * v2);
                        v.CopyTo(m_DeltaBigramLM_i, j);

                        j += Vector<double>.Count;
                    }

                    while (j < OutputLayerSize)
                    {
                        m_DeltaBigramLM_i[j] -= (CRFTagTransWeights_i[j] * CRFSeqOutput_timeat_i * CRFSeqOutput_pre_timeat[j]);
                        j++;
                    }
                }

                int iTagId = seq.States[timeat].Label;
                int iLastTagId = seq.States[timeat - 1].Label;
                m_DeltaBigramLM[iTagId][iLastTagId] += 1;
            }

            //Update tag Bigram LM
            for (int b = 0; b < OutputLayerSize; b++)
            {
                double[] vector_b = CRFTagTransWeights[b];
                double[] vector_delta_b = m_DeltaBigramLM[b];
                int a = 0;

                while (a < OutputLayerSize - Vector<double>.Count)
                {
                    Vector<double> v1 = new Vector<double>(vector_delta_b, a);
                    Vector<double> v = new Vector<double>(vector_b, a);

                    //Normalize delta
                    v1 = RNNHelper.NormalizeGradient(v1);

                    //Update weights
                    v += RNNHelper.vecNormalLearningRate * v1;
                    v.CopyTo(vector_b, a);

                    a += Vector<double>.Count;
                }

                while (a < OutputLayerSize)
                {
                    vector_b[a] += RNNHelper.LearningRate * RNNHelper.NormalizeGradient(vector_delta_b[a]);
                    a++;
                }
            }
        }

        //public int[] GetBestResult(Matrix<double> ys)
        //{
        //    int[] output = new int[ys.Height];

        //    for (int i = 0; i < ys.Height; i++)
        //    {
        //        output[i] = MathUtil.GetMaxProbIndex(ys[i]);
        //    }

        //    return output;
        //}



        public int[] GetBestResult(Matrix<double> ys)
        {
            int[] output = new int[ys.Height];

            Parallel.For(0, ys.Height, parallelOption, i =>
            //            for (int i = 0; i < ys.Height; i++)
            {
                output[i] = MathUtil.GetMaxProbIndex(ys[i]);
            });

            return output;
        }

        public double TrainNet(DataSet<T> trainingSet, int iter)
        {
            DateTime start = DateTime.Now;
            Logger.WriteLine("Iter " + iter + " begins with learning rate alpha = " + RNNHelper.LearningRate + " ...");

            //Initialize varibles
            logp = 0;

            //Shffle training corpus
            trainingSet.Shuffle();

            int numSequence = trainingSet.SequenceList.Count;
            int wordCnt = 0;
            int tknErrCnt = 0;
            int sentErrCnt = 0;
            Logger.WriteLine("Progress = 0/" + numSequence / 1000.0 + "K\r");
            for (int curSequence = 0; curSequence < numSequence; curSequence++)
            {
                T pSequence = trainingSet.SequenceList[curSequence];

                if (pSequence is Sequence)
                {
                    wordCnt += (pSequence as Sequence).States.Length;
                }
                else
                {
                    wordCnt += (pSequence as SequencePair).tgtSequence.States.Length;
                }

                int[] predicted;
                if (IsCRFTraining == true)
                {
                    predicted = ProcessSequenceCRF(pSequence as Sequence, RunningMode.Training);
                }
                else if (ModelType == MODELTYPE.SEQ2SEQ)
                {
                    predicted = ProcessSeq2Seq(pSequence as SequencePair, RunningMode.Training);
                }
                else
                {
                    Matrix<double> m;
                    predicted = ProcessSequence(pSequence as Sequence, RunningMode.Training, false, out m);
                }

                int newTknErrCnt = 0;

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
                    SaveModel(ModelTempFile);
                }
            }

            DateTime now = DateTime.Now;
            TimeSpan duration = now.Subtract(start);

            double entropy = -logp / Math.Log10(2.0) / wordCnt;
            double ppl = exp_10(-logp / wordCnt);
            Logger.WriteLine("Iter " + iter + " completed");
            Logger.WriteLine("Sentences = " + numSequence + ", time escape = " + duration + "s, speed = " + numSequence / duration.TotalSeconds);
            Logger.WriteLine("In training: log probability = " + logp + ", cross-entropy = " + entropy + ", perplexity = " + ppl);

            return ppl;
        }

        public double exp_10(double num) { return Math.Exp(num * 2.302585093); }

        public bool ValidateNet(DataSet<T> validationSet, int iter)
        {
            Logger.WriteLine("Start validation ...");
            int wordcn = 0;
            int tknErrCnt = 0;
            int sentErrCnt = 0;

            //Initialize varibles
            logp = 0;
            int numSequence = validationSet.SequenceList.Count;
            for (int curSequence = 0; curSequence < numSequence; curSequence++)
            {
                T pSequence = validationSet.SequenceList[curSequence];
                if (pSequence is Sequence)
                {
                    wordcn += (pSequence as Sequence).States.Length;
                }
                else
                {
                    wordcn += (pSequence as SequencePair).tgtSequence.States.Length;
                }

                int[] predicted;
                if (IsCRFTraining == true)
                {
                    predicted = ProcessSequenceCRF(pSequence as Sequence, RunningMode.Validate);
                }
                else if (ModelType == MODELTYPE.SEQ2SEQ)
                {
                    predicted = ProcessSeq2Seq(pSequence as SequencePair, RunningMode.Validate);
                }
                else
                {
                    Matrix<double> m;
                    predicted = ProcessSequence(pSequence as Sequence, RunningMode.Validate, false, out m);
                }

                int newTknErrCnt = 0;
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

            double entropy = -logp / Math.Log10(2.0) / wordcn;
            double ppl = exp_10(-logp / wordcn);
            double tknErrRatio = (double)tknErrCnt / (double)wordcn * 100.0;
            double sentErrRatio = (double)sentErrCnt / (double)numSequence * 100.0;

            Logger.WriteLine("In validation: error token ratio = {0}% error sentence ratio = {1}%", tknErrRatio, sentErrRatio);
            Logger.WriteLine("In training: log probability = " + logp + ", cross-entropy = " + entropy + ", perplexity = " + ppl);
            Logger.WriteLine("");

            bool bUpdate = false;
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
            int tknErrCnt = 0;
            int numStates = seq.States.Length;
            for (int curState = 0; curState < numStates; curState++)
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
            Matrix<double> ys;
            ProcessSequence(seq, RunningMode.Test, true, out ys);

            int n = seq.States.Length;
            int K = OutputLayer.LayerSize;
            Matrix<double> STP = CRFTagTransWeights;
            PAIR<int, int>[,,] vPath = new PAIR<int, int>[n, K, N];
            int DUMP_LABEL = -1;
            double[,] vPreAlpha = new double[K, N];
            double[,] vAlpha = new double[K, N];


            int nStartTagIndex = 0;
            int nEndTagIndex = 0;
            double MIN_VALUE = double.MinValue;

            //viterbi algorithm
            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    vPreAlpha[i, j] = MIN_VALUE;
                    vPath[0, i, j] = new PAIR<int, int>(DUMP_LABEL, 0);
                }
            }
            vPreAlpha[nStartTagIndex, 0] = ys[0][nStartTagIndex];
            vPath[0, nStartTagIndex, 0].first = nStartTagIndex;

            AdvUtils.PriorityQueue<double, PAIR<int, int>> q = new AdvUtils.PriorityQueue<double, PAIR<int, int>>();

            for (int t = 1; t < n; t++)
            {
                for (int j = 0; j < K; j++)
                {
                    while (q.Count() > 0)
                        q.Dequeue();
                    double _stp = STP[j][0];
                    double _y = ys[t][j];
                    for (int k = 0; k < N; k++)
                    {
                        double score = vPreAlpha[0, k] + _stp + _y;
                        q.Enqueue(score, new PAIR<int, int>(0, k));
                    }
                    for (int i = 1; i < K; i++)
                    {
                        _stp = STP[j][i];
                        for (int k = 0; k < N; k++)
                        {
                            double score = vPreAlpha[i, k] + _stp + _y;
                            if (score <= q.Peek().Key)
                                break;
                            q.Dequeue();
                            q.Enqueue(score, new PAIR<int, int>(i, k));
                        }
                    }
                    int idx = N - 1;
                    while (q.Count() > 0)
                    {
                        vAlpha[j, idx] = q.Peek().Key;
                        vPath[t, j, idx] = q.Peek().Value;
                        idx--;
                        q.Dequeue();
                    }
                }
                vPreAlpha = vAlpha;
                vAlpha = new double[K, N];
            }


            //backtrace to get the n-best result path
            int[][] vTagOutput = new int[N][];
            for (int i = 0; i < N; i++)
            {
                vTagOutput[i] = new int[n];
            }

            for (int k = 0; k < N; k++)
            {
                vTagOutput[k][n - 1] = nEndTagIndex;
                PAIR<int, int> decision = new PAIR<int, int>(nEndTagIndex, k);
                for (int t = n - 2; t >= 0; t--)
                {
                    vTagOutput[k][t] = vPath[t + 1, decision.first, decision.second].first;
                    decision = vPath[t + 1, decision.first, decision.second];
                }
            }

            return vTagOutput;
        }

        public int[] DecodeNN(Sequence seq)
        {
            Matrix<double> ys; 
            return ProcessSequence(seq, RunningMode.Test, false, out ys);
        }

        public int[] DecodeSeq2Seq(Sentence srcSent, Featurizer feature)
        {
            return TestSeq2Seq(srcSent, feature);
        }

        public int[] DecodeCRF(Sequence seq)
        {
            //ys contains the output of RNN for each word
            Matrix<double> ys;
            ProcessSequence(seq, RunningMode.Test, true, out ys);
            return Viterbi(ys, seq.States.Length);
        }
    }
}
