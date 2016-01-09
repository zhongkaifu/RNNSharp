using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.IO;
using AdvUtils;

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
        protected double logp, llogp;
        protected double minTknErrRatio;
        protected long counter;
        protected double dropout;
        protected ParallelOptions parallelOption = new ParallelOptions();
        protected double gradient_cutoff;
        protected bool m_bCRFTraining = false;
        protected MODELTYPE m_modeltype;
        protected MODELDIRECTION m_modeldirection;
        protected string m_strModelFile;

        protected static Random rand = new Random(DateTime.Now.Millisecond);
        //multiple processor declaration
        protected int L0;
        public int L1;
        public int L2;
        protected int fea_size;

        protected double alpha;
        public double Alpha
        {
            get { return alpha; }
            set { alpha = value; }
        }

        protected double[] neuFeatures;		//features in input layer
        public neuron[] neuOutput;		//neurons in output layer
        public Matrix<double> mat_hidden2output;

        protected const int MAX_RNN_HIST = 512;

        protected Matrix<double> m_RawOutput;
        protected int counterTokenForLM;

        // for Viterbi decoding
        public Matrix<double> mat_CRFTagTransWeights;

        /// for sequence training
        public Matrix<double> mat_CRFSeqOutput;

        public virtual void setTagBigramTransition(List<List<double>> m)
        {
            if (null == mat_CRFTagTransWeights)
                mat_CRFTagTransWeights = new Matrix<double>(L2, L2);

            for (int i = 0; i < L2; i++)
                for (int j = 0; j < L2; j++)
                    mat_CRFTagTransWeights[i][j] = m[i][j];

        }

        //Save matrix into file as binary format
        protected void saveMatrixBin(Matrix<double> mat, BinaryWriter fo)
        {
            int width = mat.GetWidth();
            int height = mat.GetHeight();

            //Save the width and height of the matrix
            fo.Write(width);
            fo.Write(height);

            //Save the data in matrix
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    fo.Write((float)(mat[r][c]));
                }
            }
        }

        protected Matrix<double> loadMatrixBin(BinaryReader br)
        {
            int width = br.ReadInt32();
            int height = br.ReadInt32();

            Matrix<double> m = new Matrix<double>(height, width);

            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    m[r][c] = br.ReadSingle();
                }
            }

            return m;
        }

        public void setInputLayer(State state, int curState, int numStates, int[] predicted, bool forward = true)
        {
            if (predicted != null)
            {
                // set runtime feature
                for (int i = 0; i < state.GetNumRuntimeFeature(); i++)
                {
                    for (int j = 0; j < L2; j++)
                    {
                        //Clean up run time feature value and then set a new one
                        state.SetRuntimeFeature(i, j, 0);
                    }

                    int pos = curState + ((forward == true) ? 1 : -1) * state.GetRuntimeFeature(i).OffsetToCurrentState;
                    if (pos >= 0 && pos < numStates)
                    {
                        state.SetRuntimeFeature(i, predicted[pos], 1);
                    }
                }
            }

            var dense = state.GetDenseData();
            for (int i = 0; i < dense.GetDimension(); i++)
            {
                neuFeatures[i] = dense[i];
            }
        }

        public long m_SaveStep;
        public virtual void SetSaveStep(long savestep)
        {
            m_SaveStep = savestep;
        }

        protected int m_MaxIter;
        public int MaxIter { get { return m_MaxIter; } }
        public virtual void SetMaxIter(int _nMaxIter)
        {
            m_MaxIter = _nMaxIter;
        }

        public RNN()
        {
            gradient_cutoff = 15;

            alpha = 0.1;
            dropout = 0;
            logp = 0;
            llogp = -100000000;
            minTknErrRatio = double.MaxValue;
            L1 = 30;

            fea_size = 0;

            neuFeatures = null;
            neuOutput = null;
        }

        public void SetModelDirection(int dir)
        {
            m_modeldirection = (MODELDIRECTION)dir;
        }


        public virtual void SetFeatureDimension(int denseFeatueSize, int sparseFeatureSize, int tagSize)
        {
            fea_size = denseFeatueSize;
            L0 = sparseFeatureSize;
            L2 = tagSize;
        }

        public virtual void SetCRFTraining(bool b) { m_bCRFTraining = b; }
        public virtual void SetGradientCutoff(double newGradient) { gradient_cutoff = newGradient; }
        public virtual void SetLearningRate(double newAlpha) { alpha = newAlpha; }
        public virtual void SetDropout(double newDropout) { dropout = newDropout; }
        public virtual void SetHiddenLayerSize(int newsize) { L1 = newsize;}
        public virtual void SetModelFile(string strModelFile) { m_strModelFile = strModelFile; }

        public bool IsCRFModel()
        {
            return m_bCRFTraining;
        }

        public double exp_10(double num) { return Math.Exp(num * 2.302585093); }

        public abstract void netReset(bool updateNet = false);
        public abstract void computeNet(State state, double[] doutput, bool isTrain = true);


        public virtual Matrix<double> PredictSentence(Sequence pSequence, RunningMode runningMode)
        {
            int numStates = pSequence.GetSize();
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
                State state = pSequence.Get(curState);
                setInputLayer(state, curState, numStates, predicted);
                computeNet(state, m[curState], isTraining);
                predicted[curState] = GetBestOutputIndex();

                if (runningMode != RunningMode.Test)
                {
                    logp += Math.Log10(neuOutput[state.GetLabel()].cellOutput);
                    counter++;
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
            double dmax = neuOutput[0].cellOutput;
            for (int k = 1; k < L2; k++)
            {
                if (neuOutput[k].cellOutput > dmax)
                {
                    dmax = neuOutput[k].cellOutput;
                    imax = k;
                }
            }
            return imax;
        }

        public virtual Matrix<double> learnSentenceForRNNCRF(Sequence pSequence, RunningMode runningMode)
        {
            //Reset the network
            netReset(false);
            int numStates = pSequence.GetSize();

            int[] predicted_nn = new int[numStates];
            m_RawOutput = new Matrix<double>(numStates, L2);// new double[numStates][];
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.Get(curState);

                setInputLayer(state, curState, numStates, predicted_nn);
                computeNet(state, m_RawOutput[curState]);      //compute probability distribution

                predicted_nn[curState] = GetBestOutputIndex();
            }

            ForwardBackward(numStates, m_RawOutput);

            //Get the best result
            int[] predicted = new int[numStates];
            for (int i = 0; i < numStates; i++)
            {
                State state = pSequence.Get(i);
                logp += Math.Log10(mat_CRFSeqOutput[i][state.GetLabel()]);

                predicted[i] = GetBestZIndex(i);
            }

            UpdateBigramTransition(pSequence);

            netReset(true);
            for (int curState = 0; curState < numStates; curState++)
            {
                // error propogation
                State state = pSequence.Get(curState);
                setInputLayer(state, curState, numStates, predicted_nn);
                computeNet(state, m_RawOutput[curState]);      //compute probability distribution

                counter++;

                learnNet(state, curState);
                LearnBackTime(state, numStates, curState);
            }

            return mat_CRFSeqOutput;
        }

        public void UpdateBigramTransition(Sequence seq)
        {
            int numStates = seq.GetSize();
            Matrix<double> m_DeltaBigramLM = new Matrix<double>(L2, L2);

            for (int timeat = 1; timeat < numStates; timeat++)
            {
                for (int i = 0; i < L2; i++)
                {
                    for (int j = 0; j < L2; j++)
                    {
                        m_DeltaBigramLM[i][j] -= (mat_CRFTagTransWeights[i][j] * mat_CRFSeqOutput[timeat][i] * mat_CRFSeqOutput[timeat - 1][j]);
                    }
                }

                int iTagId = seq.Get(timeat).GetLabel();
                int iLastTagId = seq.Get(timeat - 1).GetLabel();
                m_DeltaBigramLM[iTagId][iLastTagId] += 1;
            }

            counterTokenForLM++;

            //Update tag Bigram LM
            for (int b = 0;b < L2;b++)
            {
                for (int a = 0; a < L2; a++)
                {
                    mat_CRFTagTransWeights[b][a] += alpha * m_DeltaBigramLM[b][a];
                }
            }
        }

        public int GetBestZIndex(int currStatus)
        {
            //Get the output tag
            int imax = 0;
            double dmax = mat_CRFSeqOutput[currStatus][0];
            for (int j = 1; j < L2; j++)
            {
                if (mat_CRFSeqOutput[currStatus][j] > dmax)
                {
                    dmax = mat_CRFSeqOutput[currStatus][j];
                    imax = j;
                }
            }
            return imax;
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
                            double fbgm = mat_CRFTagTransWeights[j][k];
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
                            double fbgm = mat_CRFTagTransWeights[k][j];
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
            mat_CRFSeqOutput = new Matrix<double>(numStates, L2);
            for (int i = 0; i < numStates; i++)
            {
                for (int j = 0; j < L2; j++)
                {
                    mat_CRFSeqOutput[i][j] = Math.Exp(alphaSet[i][j] + betaSet[i][j] - m_RawOutput[i][j] - Z_);
                }
            }
        }



        public abstract void initWeights();

        private double random(double min, double max)
        {
            return rand.NextDouble() * (max - min) + min;
        }

        public double RandInitWeight()
        {
            return random(-0.1, 0.1) + random(-0.1, 0.1) + random(-0.1, 0.1);
        }


        public abstract void learnNet(State state, int timeat, bool biRNN = false);
        public abstract void LearnBackTime(State state, int numStates, int curState);

        public virtual double TrainNet(DataSet trainingSet, int iter)
        {
            DateTime start = DateTime.Now;
            int[] predicted;
            Logger.WriteLine(Logger.Level.info, "[TRACE] Iter " + iter + " begins with learning rate alpha = " + alpha + " ...");

            //Initialize varibles
            counter = 0;
            logp = 0;
            counterTokenForLM = 0;

            //Shffle training corpus
            trainingSet.Shuffle();

            int numSequence = trainingSet.GetSize();
            int tknErrCnt = 0;
            int sentErrCnt = 0;
            Logger.WriteLine(Logger.Level.info, "[TRACE] Progress = 0/" + numSequence / 1000.0 + "K\r");
            for (int curSequence = 0; curSequence < numSequence; curSequence++)
            {
                Sequence pSequence = trainingSet.Get(curSequence);
                int numStates = pSequence.GetSize();

                if (numStates < 3)
                    continue;

                Matrix<double> m;
                if (m_bCRFTraining == true)
                {
                    m = learnSentenceForRNNCRF(pSequence, RunningMode.Train);
                }
                else
                {
                    m = PredictSentence(pSequence, RunningMode.Train);
                }

                predicted = new int[pSequence.GetSize()];
                for (int i = 0; i < pSequence.GetSize(); i++)
                {
                    predicted[i] = MathUtil.GetMaxProbIndex(m[i]);
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
                    Logger.WriteLine(Logger.Level.info, " train cross-entropy = {0} ", -logp / Math.Log10(2.0) / counter);
                    Logger.WriteLine(Logger.Level.info, " Error token ratio = {0}%", (double)tknErrCnt / (double)counter * 100);
                    Logger.WriteLine(Logger.Level.info, " Error sentence ratio = {0}%", (double)sentErrCnt / (double)curSequence * 100);
                }

                if (m_SaveStep > 0 && (curSequence + 1) % m_SaveStep == 0)
                {
                    //After processed every m_SaveStep sentences, save current model into a temporary file
                    Logger.WriteLine(Logger.Level.info, "Saving temporary model into file...");
                    saveNetBin(m_strModelFile + ".tmp");
                }
            }

            DateTime now = DateTime.Now;
            TimeSpan duration = now.Subtract(start);

            double entropy = -logp / Math.Log10(2.0) / counter;
            double ppl = exp_10(-logp / counter);
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
        }


        protected double NormalizeErr(double err)
        {
            if (err > gradient_cutoff)
                err = gradient_cutoff;
            if (err < -gradient_cutoff)
                err = -gradient_cutoff;

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

        public int[] DecodeNN(Sequence seq)
        {
            Matrix<double> ys = PredictSentence(seq, RunningMode.Test);
            int n = seq.GetSize();
            int[] output = new int[n];

            for (int i = 0; i < n; i++)
            {
                output[i] = MathUtil.GetMaxProbIndex(ys[i]);
            }

            return output;
        }


        public int[][] DecodeNBestCRF(Sequence seq, int N)
        {

            //ys contains the output of RNN for each word
            Matrix<double> ys = PredictSentence(seq, RunningMode.Test);

            int n = seq.GetSize();
            int K = L2;
            Matrix<double> STP = mat_CRFTagTransWeights;
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

        public int[] DecodeCRF(Sequence seq)
        {
            //ys contains the output of RNN for each word
            Matrix<double> ys = PredictSentence(seq, RunningMode.Test);

            int n = seq.GetSize();
            int K = L2;
            Matrix<double> STP = mat_CRFTagTransWeights;
            int[,] vPath = new int[n, K];

            double[] vPreAlpha = new double[K];
            double[] vAlpha = new double[K];


            int nStartTagIndex = 0;
            double MIN_VALUE = double.MinValue;
            //viterbi algorithm
            for (int i = 0; i < K; i++)
            {
                vPreAlpha[i] = ys[0][i];
                if (i != nStartTagIndex)
                    vPreAlpha[i] += MIN_VALUE;
                vPath[0, i] = nStartTagIndex;
            }
            for (int t = 1; t < n; t++)
            {
                for (int j = 0; j < K; j++)
                {
                    vPath[t, j] = 0;
                    vAlpha[j] = MIN_VALUE;

                    for (int i = 0; i < K; i++)
                    {
                        double score = vPreAlpha[i] + STP[j][i] + ys[t][j];

                        if (score > vAlpha[j])
                        {
                            vAlpha[j] = score;
                            vPath[t, j] = i;
                        }
                    }
                }
                vPreAlpha = vAlpha;
                vAlpha = new double[K];
            }

            //backtrace to get the best result path
            int[] tagOutputs = new int[n];
            tagOutputs[n - 1] = nStartTagIndex;
            int nNextTag = tagOutputs[n - 1];
            for (int t = n - 2; t >= 0; t--)
            {
                tagOutputs[t] = vPath[t + 1, nNextTag];
                nNextTag = tagOutputs[t];
            }

            return tagOutputs;
        }

        private int GetErrorTokenNum(Sequence seq, int[] predicted)
        {
            int tknErrCnt = 0;
            int numStates = seq.GetSize();
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = seq.Get(curState);
                if (predicted[curState] != state.GetLabel())
                {
                    tknErrCnt++;
                }
            }

            return tknErrCnt;
        }

        public void CalculateOutputLayerError(State state, int timeat)
        {
            if (m_bCRFTraining == true)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                for (int c = 0; c < L2; c++)
                {
                    neuOutput[c].er = -mat_CRFSeqOutput[timeat][c];
                }
                neuOutput[state.GetLabel()].er = 1 - mat_CRFSeqOutput[timeat][state.GetLabel()];
            }
            else
            {
                //For standard RNN
                for (int c = 0; c < L2; c++)
                {
                    neuOutput[c].er = -neuOutput[c].cellOutput;
                }
                neuOutput[state.GetLabel()].er = 1 - neuOutput[state.GetLabel()].cellOutput;
            }

        }


        public virtual bool ValidateNet(DataSet validationSet)
        {
            Logger.WriteLine(Logger.Level.info, "[TRACE] Start validation ...");
            int wordcn = 0;
            int[] predicted;
            int tknErrCnt = 0;
            int sentErrCnt = 0;

            //Initialize varibles
            counter = 0;
            logp = 0;
            counterTokenForLM = 0;
          
            int numSequence = validationSet.GetSize();
            for (int curSequence = 0; curSequence < numSequence; curSequence++)
            {
                Sequence pSequence = validationSet.Get(curSequence);
                wordcn += pSequence.GetSize();

                Matrix<double> m;
                if (m_bCRFTraining == true)
                {
                    m = learnSentenceForRNNCRF(pSequence, RunningMode.Validate);
                }
                else
                {
                    m = PredictSentence(pSequence, RunningMode.Validate);
                }

                predicted = new int[pSequence.GetSize()];
                for (int i = 0; i < pSequence.GetSize(); i++)
                {
                    predicted[i] = MathUtil.GetMaxProbIndex(m[i]);
                }

                int newTknErrCnt = GetErrorTokenNum(pSequence, predicted);
                tknErrCnt += newTknErrCnt;
                if (newTknErrCnt > 0)
                {
                    sentErrCnt++;
                }
            }

            double entropy = -logp / Math.Log10(2.0) / counter;
            double ppl = exp_10(-logp / counter);
            double tknErrRatio = (double)tknErrCnt / (double)wordcn * 100;
            double sentErrRatio = (double)sentErrCnt / (double)numSequence * 100;

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
