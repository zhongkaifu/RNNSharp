using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using AdvUtils;

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
        public virtual float GradientCutoff { get; set; }
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
        public neuron[] OutputLayer { get; set; }
        public Matrix<double> Hidden2OutputWeight;
      
        // CRF result output
        protected Matrix<double> CRFSeqOutput;
        protected double logp;
        protected double minTknErrRatio = double.MaxValue;
        protected ParallelOptions parallelOption = new ParallelOptions();
        protected static Random rand = new Random(DateTime.Now.Millisecond);
        //multiple processor declaration
        protected Vector neuFeatures;		//features in input layer
        protected const int MAX_RNN_HIST = 64;

        public virtual void setTagBigramTransition(List<List<float>> m)
        {
            CRFTagTransWeights = new Matrix<double>(L2, L2);

            for (int i = 0; i < L2; i++)
                for (int j = 0; j < L2; j++)
                    CRFTagTransWeights[i][j] = m[i][j];

        }

        //Save matrix into file as binary format
        protected void saveMatrixBin(Matrix<double> mat, BinaryWriter fo, bool BuildVQ = true)
        {
            int width = mat.GetWidth();
            int height = mat.GetHeight();

            //Save the width and height of the matrix
            fo.Write(width);
            fo.Write(height);

            if (BuildVQ == false)
            {
                Logger.WriteLine("Saving matrix without VQ...");
                fo.Write(0); // non-VQ

                //Save the data in matrix
                for (int r = 0; r < height; r++)
                {
                    for (int c = 0; c < width; c++)
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
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
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
                for (int r = 0; r < height; r++)
                {
                    for (int c = 0; c < width; c++)
                    {
                        fo.Write((byte)vq.ComputeVQ(mat[r][c]));
                    }
                }
            }
        }

        protected Matrix<double> loadMatrixBin(BinaryReader br)
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

        public void setInputLayer(State state, int curState, int numStates, int[] predicted, bool forward = true)
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
        public abstract void computeNet(State state, double[] doutput, bool isTrain = true);


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
                setInputLayer(state, curState, numStates, predicted);
                computeNet(state, m[curState], isTraining);
                predicted[curState] = GetBestOutputIndex();

                if (runningMode != RunningMode.Test)
                {
                    logp += Math.Log10(OutputLayer[state.Label].cellOutput);
                }

                if (runningMode == RunningMode.Train)
                {
                    // error propogation
                    learnNet(state, curState);
                    LearnBackTime(state, numStates, curState);
                }
            }

            return m;
        }


        public void SoftmaxLayer(neuron[] layer)
        {
            //activation 2   --softmax on words
            double sum = 0;   //sum is used for normalization: it's better to have larger precision as many numbers are summed together here
            for (int c = 0; c < L2; c++)
            {
                if (layer[c].cellOutput > 50) layer[c].cellOutput = 50;  //for numerical stability
                if (layer[c].cellOutput < -50) layer[c].cellOutput = -50;  //for numerical stability
                double val = Math.Exp(layer[c].cellOutput);
                sum += val;
                layer[c].cellOutput = val;
            }
            for (int c = 0; c < L2; c++)
            {
                layer[c].cellOutput /= sum;
            }
        }

        public int GetBestOutputIndex()
        {
            int imax = 0;
            double dmax = OutputLayer[0].cellOutput;
            for (int k = 1; k < L2; k++)
            {
                if (OutputLayer[k].cellOutput > dmax)
                {
                    dmax = OutputLayer[k].cellOutput;
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
                    logp += Math.Log10(CRFSeqOutput[i][pSequence.States[i].Label]);
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
                    setInputLayer(state, curState, numStates, null);
                    computeNet(state, null);      //compute probability distribution

                    learnNet(state, curState);
                    LearnBackTime(state, numStates, curState);
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
                for (int i = 0; i < L2; i++)
                {
                    for (int j = 0; j < L2; j++)
                    {
                        m_DeltaBigramLM[i][j] -= (CRFTagTransWeights[i][j] * CRFSeqOutput[timeat][i] * CRFSeqOutput[timeat - 1][j]);
                    }
                }

                int iTagId = seq.States[timeat].Label;
                int iLastTagId = seq.States[timeat - 1].Label;
                m_DeltaBigramLM[iTagId][iLastTagId] += 1;
            }

            //Update tag Bigram LM
            for (int b = 0;b < L2;b++)
            {
                for (int a = 0; a < L2; a++)
                {
                    CRFTagTransWeights[b][a] += LearningRate * m_DeltaBigramLM[b][a];
                }
            }
        }

        public void ForwardBackward(int numStates, Matrix<double> m_RawOutput)
        {
            //forward
            double[][] alphaSet = new double[numStates][];
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

            //backward
            double[][] betaSet = new double[numStates][];
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

            //Z_
            double Z_ = 0.0;
            for (int i = 0; i < L2; i++)
            {
                Z_ = MathUtil.logsumexp(Z_, betaSet[0][i], i == 0);

            }

            //Calculate the output probability of each node
            CRFSeqOutput = new Matrix<double>(numStates, L2);
            for (int i = 0; i < numStates; i++)
            {
                for (int j = 0; j < L2; j++)
                {
                    CRFSeqOutput[i][j] = Math.Exp(alphaSet[i][j] + betaSet[i][j] - m_RawOutput[i][j] - Z_);
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


        public abstract void learnNet(State state, int timeat, bool biRNN = false);
        public abstract void LearnBackTime(State state, int numStates, int curState);

        public virtual double TrainNet(DataSet trainingSet, int iter)
        {
            DateTime start = DateTime.Now;
            Logger.WriteLine(Logger.Level.info, "[TRACE] Iter " + iter + " begins with learning rate alpha = " + LearningRate + " ...");

            //Initialize varibles
            logp = 0;

            //Shffle training corpus
            trainingSet.Shuffle();

            int numSequence = trainingSet.SequenceList.Count;
            int wordCnt = 0;
            int tknErrCnt = 0;
            int sentErrCnt = 0;
            Logger.WriteLine(Logger.Level.info, "[TRACE] Progress = 0/" + numSequence / 1000.0 + "K\r");
            for (int curSequence = 0; curSequence < numSequence; curSequence++)
            {
                Sequence pSequence = trainingSet.SequenceList[curSequence];
                int numStates = pSequence.States.Length;
                wordCnt += numStates;

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
                    Logger.WriteLine(Logger.Level.info, "[TRACE] Progress = {0} ", (curSequence + 1) / 1000 + "K/" + numSequence / 1000.0 + "K");
                    Logger.WriteLine(Logger.Level.info, " train cross-entropy = {0} ", -logp / Math.Log10(2.0) / wordCnt);
                    Logger.WriteLine(Logger.Level.info, " Error token ratio = {0}%", (double)tknErrCnt / (double)wordCnt * 100.0);
                    Logger.WriteLine(Logger.Level.info, " Error sentence ratio = {0}%", (double)sentErrCnt / (double)curSequence * 100.0);
                }

                if (SaveStep > 0 && (curSequence + 1) % SaveStep == 0)
                {
                    //After processed every m_SaveStep sentences, save current model into a temporary file
                    Logger.WriteLine(Logger.Level.info, "Saving temporary model into file...");
                    saveNetBin(ModelTempFile);
                }
            }

            DateTime now = DateTime.Now;
            TimeSpan duration = now.Subtract(start);

            double entropy = -logp / Math.Log10(2.0) / wordCnt;
            double ppl = exp_10(-logp / wordCnt);
            Logger.WriteLine(Logger.Level.info, "[TRACE] Iter " + iter + " completed");
            Logger.WriteLine(Logger.Level.info, "[TRACE] Sentences = " + numSequence + ", time escape = " + duration + "s, speed = " + numSequence / duration.TotalSeconds);
            Logger.WriteLine(Logger.Level.info, "[TRACE] In training: log probability = " + logp + ", cross-entropy = " + entropy + ", perplexity = " + ppl);

            return ppl;
        }

        public abstract void initMem();

        public abstract void saveNetBin(string filename);
        public abstract void loadNetBin(string filename);

        public abstract void GetHiddenLayer(Matrix<double> m, int curStatus);

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


        protected double NormalizeErr(double err)
        {
            if (err > GradientCutoff)
                err = GradientCutoff;
            if (err < -GradientCutoff)
                err = -GradientCutoff;

            return err;
        }

        public void matrixXvectorADD(neuron[] dest, neuron[] srcvec, Matrix<double> srcmatrix, int from, int to, int from2, int to2, int type)
        {
            if (type == 0)
            {
                //ac mod
                Parallel.For(0, (to - from), parallelOption, i =>
                {
                    dest[i + from].cellOutput = 0;
                    for (int j = 0; j < to2 - from2; j++)
                    {
                        dest[i + from].cellOutput += srcvec[j + from2].cellOutput * srcmatrix[i][j];
                    }
                });

            }
            else
            {
                Parallel.For(0, (to - from), parallelOption, i =>
                {
                    dest[i + from].er = 0;
                    for (int j = 0; j < to2 - from2; j++)
                    {
                        dest[i + from].er += srcvec[j + from2].er * srcmatrix[j][i];
                    }
                });

                for (int i = from; i < to; i++)
                {
                    dest[i].er = NormalizeErr(dest[i].er);
                }
            }
        }


        public int[] GetBestResult(Matrix<double> ys)
        {
            int[] output = new int[ys.GetHeight()];

            for (int i = 0; i < ys.GetHeight(); i++)
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
                    vAlpha[j] = double.MinValue;

                    for (int i = 0; i < L2; i++)
                    {
                        double score = vPreAlpha[i] + CRFTagTransWeights[j][i] + ys[t][j];
                        if (score > vAlpha[j])
                        {
                            vAlpha[j] = score;
                            vPath[t, j] = i;
                        }
                    }
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

        public void CalculateOutputLayerError(State state, int timeat)
        {
            if (IsCRFTraining == true)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                for (int c = 0; c < L2; c++)
                {
                    OutputLayer[c].er = -CRFSeqOutput[timeat][c];
                }
                OutputLayer[state.Label].er = 1 - CRFSeqOutput[timeat][state.Label];
            }
            else
            {
                //For standard RNN
                for (int c = 0; c < L2; c++)
                {
                    OutputLayer[c].er = -OutputLayer[c].cellOutput;
                }
                OutputLayer[state.Label].er = 1 - OutputLayer[state.Label].cellOutput;
            }

        }

        public virtual bool ValidateNet(DataSet validationSet, int iter)
        {
            Logger.WriteLine(Logger.Level.info, "[TRACE] Start validation ...");
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

            Logger.WriteLine(Logger.Level.info, "[TRACE] In validation: error token ratio = {0}% error sentence ratio = {1}%", tknErrRatio, sentErrRatio);
            Logger.WriteLine(Logger.Level.info, "[TRACE] In training: log probability = " + logp + ", cross-entropy = " + entropy + ", perplexity = " + ppl);         
            Logger.WriteLine(Logger.Level.info, "");

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
