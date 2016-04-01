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
    public enum MODELTYPE
    {
        SIMPLE = 0,
        LSTM
    }

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

    abstract public class RNN
    {
        public virtual bool IsCRFTraining { get; set; }
        public virtual string ModelFile { get; set; }
        public string ModelTempFile { get { return ModelFile + ".tmp"; } }
        public virtual MODELDIRECTION ModelDirection { get; set; }
        public virtual bool bVQ { get; set; }
        public virtual double GradientCutoff { get; set; }
        public virtual float Dropout { get; set; }
        public virtual float LearningRate { get; set; }
        public virtual int MaxIter { get; set; }
        public virtual long SaveStep { get; set; }
        public virtual int DenseFeatureSize { get; set; }
        public virtual int L0 { get; set; }
        public virtual int L1 { get; set; }
        public virtual int L2 { get; set; }

        public MODELTYPE ModelType { get; set; }
        public Matrix<double> CRFTagTransWeights { get; set; }
        public SimpleLayer OutputLayer { get; set; }
        public Matrix<double> Hidden2OutputWeight;
        public Matrix<double> Hidden2OutputWeightLearningRate;
      
        // CRF result output
        protected Matrix<double> CRFSeqOutput;
        protected double logp;
        protected double minTknErrRatio = double.MaxValue;
        protected ParallelOptions parallelOption = new ParallelOptions();
        protected static Random rand = new Random(DateTime.Now.Millisecond);
        //multiple processor declaration
        protected VectorBase neuFeatures;		//features in input layer
        protected const int MAX_RNN_HIST = 64;

        protected Vector<double> vecMaxGrad;
        protected Vector<double> vecMinGrad;
        protected Vector<double> vecNormalLearningRate;

        public virtual void setTagBigramTransition(List<List<float>> m)
        {
            CRFTagTransWeights = new Matrix<double>(L2, L2);
            for (int i = 0; i < L2; i++)
            {
                for (int j = 0; j < L2; j++)
                {
                    CRFTagTransWeights[i][j] = m[i][j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<double> NormalizeGradient(Vector<double> v)
        {
            v = Vector.Min<double>(v, vecMaxGrad);
            v = Vector.Max<double>(v, vecMinGrad);

            return v;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<double> ComputeLearningRate(Vector<double> vecDelta, ref Vector<double> vecLearningRateWeights)
        {
            vecLearningRateWeights += (vecDelta * vecDelta);
            return vecNormalLearningRate / (Vector<double>.One + Vector.SquareRoot<double>(vecLearningRateWeights));
        }

        public double UpdateLearningRate(Matrix<double> m, int i, int j, double delta)
        {
            double dg = m[i][j] + delta * delta;
            m[i][j] = dg;

            return LearningRate / (1.0 + Math.Sqrt(dg));
        }

        //Save matrix into file as binary format
        protected void SaveMatrix(Matrix<double> mat, BinaryWriter fo)
        {
            //Save the width and height of the matrix
            fo.Write(mat.Width);
            fo.Write(mat.Height);

            if (bVQ == false)
            {
                Logger.WriteLine("Saving matrix without VQ...");
                fo.Write(0); // non-VQ

                //Save the data in matrix
                for (int r = 0; r < mat.Height; r++)
                {
                    for (int c = 0; c < mat.Width; c++)
                    {
                        fo.Write((float)mat[r][c]);
                    }
                }
            }
            else
            {
                //Build vector quantization matrix
                int vqSize = 256;
                VectorQuantization vq = new VectorQuantization();
                Logger.WriteLine("Saving matrix with VQ {0}...", vqSize);

                int valSize = 0;
                for (int i = 0; i < mat.Height; i++)
                {
                    for (int j = 0; j < mat.Width; j++)
                    {
                        vq.Add(mat[i][j]);
                        valSize++;
                    }
                }

                if (vqSize > valSize)
                {
                    vqSize = valSize;
                }

                double distortion = vq.BuildCodebook(vqSize);
                Logger.WriteLine("Distortion: {0}, vqSize: {1}", distortion, vqSize);

                //Save VQ codebook into file
                fo.Write(vqSize);
                for (int j = 0; j < vqSize; j++)
                {
                    fo.Write(vq.CodeBook[j]);
                }

                //Save the data in matrix
                for (int r = 0; r < mat.Height; r++)
                {
                    for (int c = 0; c < mat.Width; c++)
                    {
                        fo.Write((byte)vq.ComputeVQ(mat[r][c]));
                    }
                }
            }
        }

        protected Matrix<double> LoadMatrix(BinaryReader br)
        {
            int width = br.ReadInt32();
            int height = br.ReadInt32();
            int vqSize = br.ReadInt32();
            Logger.WriteLine("Loading matrix. width: {0}, height: {1}, vqSize: {2}", width, height, vqSize);

            Matrix<double> m = new Matrix<double>(height, width);
            if (vqSize == 0)
            {
                for (int r = 0; r < height; r++)
                {
                    for (int c = 0; c < width; c++)
                    {
                        m[r][c] = br.ReadSingle();
                    }
                }
            }
            else
            {
                List<double> codeBook = new List<double>();

                for (int i = 0; i < vqSize; i++)
                {
                    codeBook.Add(br.ReadDouble());
                }


                for (int r = 0; r < height; r++)
                {
                    for (int c = 0; c < width; c++)
                    {
                        int vqIndex = br.ReadByte();
                        m[r][c] = codeBook[vqIndex];
                    }
                }
            }

            return m;
        }

        public void SetInputLayer(State state, int curState, int numStates, int[] predicted, bool forward = true)
        {
            if (predicted != null && state.RuntimeFeatures != null)
            {
                // set runtime feature
                for (int i = 0; i < state.RuntimeFeatures.Length; i++)
                {
                    for (int j = 0; j < L2; j++)
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

            neuFeatures = state.DenseData;
        }

        public double exp_10(double num) { return Math.Exp(num * 2.302585093); }

        public abstract void netReset(bool updateNet = false);
        public abstract void computeHiddenLayer(State state, bool isTrain = true);

        public abstract void computeOutput(double[] doutput);


        public virtual Matrix<double> PredictSentence(Sequence pSequence, RunningMode runningMode)
        {
            int numStates = pSequence.States.Length;
            Matrix<double> m = new Matrix<double>(numStates, L2);
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

            netReset(isTraining);
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.States[curState];
                SetInputLayer(state, curState, numStates, predicted);
                computeHiddenLayer(state, isTraining);

                computeOutput(m[curState]);

                predicted[curState] = GetBestOutputIndex();

                if (runningMode != RunningMode.Test)
                {
                    logp += Math.Log10(OutputLayer.cellOutput[state.Label] + 0.0001);
                }

                if (runningMode == RunningMode.Train)
                {
                    // error propogation
                    ComputeOutputLayerErr(state, curState);
                    ComputeHiddenLayerErr();

                    Parallel.Invoke(() =>
                    {
                        //Update net weights
                        LearnOutputWeight();
                    },
                    () =>
                    {
                        //Update net weights
                        LearnNet(state, numStates, curState);
                    });
                }
            }

            return m;
        }


        public void SoftmaxLayer(SimpleLayer layer)
        {
            double sum = 0;
            for (int c = 0; c < L2; c++)
            {
                double cellOutput = layer.cellOutput[c];
                if (cellOutput > 50) cellOutput = 50;
                if (cellOutput < -50) cellOutput = -50;
                double val = Math.Exp(cellOutput);
                sum += val;
                layer.cellOutput[c] = val;
            }
            int i = 0;
            Vector<double> vecSum = new Vector<double>(sum);
            while (i < L2 - Vector<double>.Count)
            {
                Vector<double> v = new Vector<double>(layer.cellOutput, i);
                v /= vecSum;
                v.CopyTo(layer.cellOutput, i);

                i += Vector<double>.Count;
            }

            while (i < L2)
            {
                layer.cellOutput[i] /= sum;
                i++;
            }
        }

        public int GetBestOutputIndex()
        {
            int imax = 0;
            double dmax = OutputLayer.cellOutput[0];
            for (int k = 1; k < L2; k++)
            {
                if (OutputLayer.cellOutput[k] > dmax)
                {
                    dmax = OutputLayer.cellOutput[k];
                    imax = k;
                }
            }
            return imax;
        }

        public virtual int[] PredictSentenceCRF(Sequence pSequence, RunningMode runningMode)
        {
            int numStates = pSequence.States.Length;

            Matrix<double> nnOutput = PredictSentence(pSequence, RunningMode.Test);
            ForwardBackward(numStates, nnOutput);

            if (runningMode != RunningMode.Test)
            {
                //Get the best result
                for (int i = 0; i < numStates; i++)
                {
                    logp += Math.Log10(CRFSeqOutput[i][pSequence.States[i].Label] + 0.0001);
                }
            }

            int[] predicted = Viterbi(nnOutput, numStates);

            if (runningMode == RunningMode.Train)
            {
                UpdateBigramTransition(pSequence);
                netReset(true);
                for (int curState = 0; curState < numStates; curState++)
                {
                    // error propogation
                    State state = pSequence.States[curState];
                    SetInputLayer(state, curState, numStates, null);
                    computeHiddenLayer(state);      //compute probability distribution

                    ComputeOutputLayerErr(state, curState);
                    ComputeHiddenLayerErr();

                    Parallel.Invoke(() =>
                    {
                        //Update net weights
                        LearnOutputWeight();
                    },
                    () =>
                    {
                        //Update net weights
                        LearnNet(state, numStates, curState);
                    });
                }
            }

            return predicted;
        }

        public void UpdateBigramTransition(Sequence seq)
        {
            int numStates = seq.States.Length;
            Matrix<double> m_DeltaBigramLM = new Matrix<double>(L2, L2);

            for (int timeat = 1; timeat < numStates; timeat++)
            {
                double[] CRFSeqOutput_timeat = CRFSeqOutput[timeat];
                double[] CRFSeqOutput_pre_timeat = CRFSeqOutput[timeat - 1];
                for (int i = 0; i < L2; i++)
                {
                    double CRFSeqOutput_timeat_i = CRFSeqOutput_timeat[i];
                    double[] CRFTagTransWeights_i = CRFTagTransWeights[i];
                    double[] m_DeltaBigramLM_i = m_DeltaBigramLM[i];
                    int j = 0;

                    Vector<double> vecCRFSeqOutput_timeat_i = new Vector<double>(CRFSeqOutput_timeat_i);
                    while (j < L2 - Vector<double>.Count)
                    {
                        Vector<double> v1 = new Vector<double>(CRFTagTransWeights_i, j);
                        Vector<double> v2 = new Vector<double>(CRFSeqOutput_pre_timeat, j);
                        Vector<double> v = new Vector<double>(m_DeltaBigramLM_i, j);

                        v -= (v1 * vecCRFSeqOutput_timeat_i * v2);
                        v.CopyTo(m_DeltaBigramLM_i, j);

                        j += Vector<double>.Count;
                    }

                    while (j < L2)
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
            for (int b = 0;b < L2;b++)
            {
                double[] vector_b = CRFTagTransWeights[b];
                double[] vector_delta_b = m_DeltaBigramLM[b];
                int a = 0;

                while (a < L2 - Vector<double>.Count)
                {
                    Vector<double> v1 = new Vector<double>(vector_delta_b, a);
                    Vector<double> v = new Vector<double>(vector_b, a);

                    //Normalize delta
                    v1 = NormalizeGradient(v1);

                    //Update weights
                    v += vecNormalLearningRate * v1;
                    v.CopyTo(vector_b, a);

                    a += Vector<double>.Count;
                }

                while (a < L2)
                {
                    vector_b[a] += LearningRate * NormalizeGradient(vector_delta_b[a]);
                    a++;
                }
            }
        }

        public void ForwardBackward(int numStates, Matrix<double> m_RawOutput)
        {
            //forward
            double[][] alphaSet = new double[numStates][];
            double[][] betaSet = new double[numStates][];

            Parallel.Invoke(() =>
            {
                for (int i = 0; i < numStates; i++)
                {
                    alphaSet[i] = new double[L2];
                    for (int j = 0; j < L2; j++)
                    {
                        double dscore0 = 0;
                        if (i > 0)
                        {
                            for (int k = 0; k < L2; k++)
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
                    betaSet[i] = new double[L2];
                    for (int j = 0; j < L2; j++)
                    {
                        double dscore0 = 0;
                        if (i < numStates - 1)
                        {
                            for (int k = 0; k < L2; k++)
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
            for (int i = 0; i < L2; i++)
            {
                Z_ = MathUtil.logsumexp(Z_, betaSet_0[i], i == 0);

            }

            //Calculate the output probability of each node
            CRFSeqOutput = new Matrix<double>(numStates, L2);
            for (int i = 0; i < numStates; i++)
            {
                double[] CRFSeqOutput_i = CRFSeqOutput[i];
                double[] alphaSet_i = alphaSet[i];
                double[] betaSet_i = betaSet[i];
                double[] m_RawOutput_i = m_RawOutput[i];
                for (int j = 0; j < L2; j++)
                {
                    CRFSeqOutput_i[j] = Math.Exp(alphaSet_i[j] + betaSet_i[j] - m_RawOutput_i[j] - Z_);
                }
            }

        }



        public abstract void initWeights();

        private double random(double min, double max)
        {
            return rand.NextDouble() * (max - min) + min;
        }

        public float RandInitWeight()
        {
            return (float)(random(-0.1, 0.1) + random(-0.1, 0.1) + random(-0.1, 0.1));
        }


        public abstract void LearnNet(State state, int numStates, int curState);

        public abstract void ComputeHiddenLayerErr();

        public abstract void LearnOutputWeight();

        public virtual double TrainNet(DataSet trainingSet, int iter)
        {
            DateTime start = DateTime.Now;
            Logger.WriteLine("Iter " + iter + " begins with learning rate alpha = " + LearningRate + " ...");

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
                Sequence pSequence = trainingSet.SequenceList[curSequence];
                wordCnt += pSequence.States.Length;

                int[] predicted;
                if (IsCRFTraining == true)
                {
                    predicted = PredictSentenceCRF(pSequence, RunningMode.Train);
                }
                else
                {
                    Matrix<double> m;
                    m = PredictSentence(pSequence, RunningMode.Train);
                    predicted = GetBestResult(m);
                }

                int newTknErrCnt = GetErrorTokenNum(pSequence, predicted);
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

        public abstract void CleanStatus();
        public abstract void InitMem();
        public abstract void SaveModel(string filename);
        public abstract void LoadModel(string filename);

        public abstract SimpleLayer GetHiddenLayer();

        public static void CheckModelFileType(string filename, out MODELTYPE modelType, out MODELDIRECTION modelDir)
        {
            using (StreamReader sr = new StreamReader(filename))
            {
                BinaryReader br = new BinaryReader(sr.BaseStream);
                modelType = (MODELTYPE)br.ReadInt32();
                modelDir = (MODELDIRECTION)br.ReadInt32();
            }

            Logger.WriteLine("Get model type {0} and direction {1}", modelType, modelDir);
        }


        protected double NormalizeGradient(double err)
        {
            if (err > GradientCutoff)
            {
                err = GradientCutoff;
            }
            else if (err < -GradientCutoff)
            {
                err = -GradientCutoff;
            }
            return err;
        }

        public void matrixXvectorADD(double[] dest, double[] srcvec, Matrix<double> srcmatrix, int DestSize, int SrcSize)
        {
            //ac mod
            Parallel.For(0, DestSize, parallelOption, i =>
            {
                double[] vector_i = srcmatrix[i];
                double cellOutput = 0;
                int j = 0;

                while (j < SrcSize - Vector<double>.Count)
                {
                    Vector<double> v1 = new Vector<double>(srcvec, j);
                    Vector<double> v2 = new Vector<double>(vector_i, j);
                    cellOutput += Vector.Dot<double>(v1, v2);

                    j += Vector<double>.Count;
                }

                while (j < SrcSize)
                {
                    cellOutput += srcvec[j] * vector_i[j];
                    j++;
                }

                dest[i] = cellOutput;
            });
        }


        public void matrixXvectorADDErr(double[] dest, double[] srcvec, Matrix<double> srcmatrix, int DestSize, int SrcSize)
        {
            Parallel.For(0, DestSize, parallelOption, i =>
            {
                double er = 0;
                for (int j = 0; j < SrcSize; j++)
                {
                    er += srcvec[j] * srcmatrix[j][i];
                }

                dest[i] = NormalizeGradient(er);
            });
        }

        public int[] GetBestResult(Matrix<double> ys)
        {
            int[] output = new int[ys.Height];

            for (int i = 0; i < ys.Height; i++)
            {
                output[i] = MathUtil.GetMaxProbIndex(ys[i]);
            }

            return output;
        }

        public int[] DecodeNN(Sequence seq)
        {
            Matrix<double> ys = PredictSentence(seq, RunningMode.Test);
            return GetBestResult(ys);
        }


        public int[][] DecodeNBestCRF(Sequence seq, int N)
        {

            //ys contains the output of RNN for each word
            Matrix<double> ys = PredictSentence(seq, RunningMode.Test);

            int n = seq.States.Length;
            int K = L2;
            Matrix<double> STP = CRFTagTransWeights;
            PAIR<int, int>[, ,] vPath = new PAIR<int, int>[n, K, N];
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

        public int[] Viterbi(Matrix<double> ys, int seqLen)
        {
            int[,] vPath = new int[seqLen, L2];

            double[] vPreAlpha = new double[L2];
            double[] vAlpha = new double[L2];


            int nStartTagIndex = 0;
            //viterbi algorithm
            for (int i = 0; i < L2; i++)
            {
                vPreAlpha[i] = ys[0][i];
                if (i != nStartTagIndex)
                    vPreAlpha[i] += double.MinValue;
                vPath[0, i] = nStartTagIndex;
            }
            for (int t = 0; t < seqLen; t++)
            {
                for (int j = 0; j < L2; j++)
                {
                    vPath[t, j] = 0;
                    double[] CRFTagTransWeights_j = CRFTagTransWeights[j];
                    double[] ys_t = ys[t];
                    double maxScore = double.MinValue;
                    for (int i = 0; i < L2; i++)
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
                vAlpha = new double[L2];
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

        public int[] DecodeCRF(Sequence seq)
        {
            //ys contains the output of RNN for each word
            Matrix<double> ys = PredictSentence(seq, RunningMode.Test);
            return Viterbi(ys, seq.States.Length);
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

        public void ComputeOutputLayerErr(State state, int timeat)
        {
            if (IsCRFTraining == true)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                for (int c = 0; c < L2; c++)
                {
                    OutputLayer.er[c] = -CRFSeqOutput[timeat][c];
                }
                OutputLayer.er[state.Label] = 1 - CRFSeqOutput[timeat][state.Label];
            }
            else
            {
                //For standard RNN
                for (int c = 0; c < L2; c++)
                {
                    OutputLayer.er[c] = -OutputLayer.cellOutput[c];
                }
                OutputLayer.er[state.Label] = 1 - OutputLayer.cellOutput[state.Label];
            }

        }

        public virtual bool ValidateNet(DataSet validationSet, int iter)
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
                Sequence pSequence = validationSet.SequenceList[curSequence];
                wordcn += pSequence.States.Length;

                int[] predicted;
                if (IsCRFTraining == true)
                {
                    predicted = PredictSentenceCRF(pSequence, RunningMode.Validate);
                }
                else
                {
                    Matrix<double> m;
                    m = PredictSentence(pSequence, RunningMode.Validate);
                    predicted = GetBestResult(m);
                }

                int newTknErrCnt = GetErrorTokenNum(pSequence, predicted);
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
    }
}
