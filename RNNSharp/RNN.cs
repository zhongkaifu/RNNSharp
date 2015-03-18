using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.IO;

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
        protected int iter;
        protected double alpha;
        protected double logp, llogp;
        protected double minTknErrRatio;
        protected double lastTknErrRatio;
        protected long counter;
        protected double beta;
        protected ParallelOptions parallelOption = new ParallelOptions();
        protected double gradient_cutoff;
        protected bool m_bCRFTraining = false;
        protected MODELTYPE m_modeltype;
        protected MODELDIRECTION m_modeldirection;
        protected string m_strModelFile;
        protected bool m_bDynAlpha = true;

        static Random rand = new Random(DateTime.Now.Millisecond);

        //multiple processor declaration
        protected int L0;
        protected int L1;
        public int L2;
        protected int fea_size;

        protected float randrng;

        protected int final_stop;

        protected neuron[] neuInput;		//neurons in input layer
        protected neuron[] neuFeatures;		//features in input layer
        protected neuron[] neuHidden;		//neurons in hidden layer
        public neuron[] neuOutput;		//neurons in output layer


        protected const int MAX_RNN_HIST = 128;

        protected Matrix m_RawOutput;
        protected int counterTokenForLM;

        protected Matrix mat_input2hidden = new Matrix();
        protected Matrix mat_input2hidden_alpha = new Matrix();
        
        protected Matrix mat_hidden2output = new Matrix();
        protected Matrix mat_hidden2output_alpha = new Matrix();

        protected Matrix mat_feature2hidden = new Matrix();
        protected Matrix mat_feature2hidden_alpha = new Matrix();

        protected Matrix mat_feature2output = new Matrix();
        protected Matrix mat_feature2output_alpha = new Matrix();


        protected double md_beta = 1.0;

        protected DataSet m_TrainingSet;
        protected DataSet m_ValidationSet;

        // for Viterbi decoding
        public Matrix m_tagBigramTransition;
        public Matrix m_tagBigramTransition_alpha;

        /// for sequence training
        public Matrix m_DeltaBigramLM; // different of tag output tag transition probability, saving p(u|v) in a sparse matrix
        public double[][] m_Diff = new double[MAX_RNN_HIST][];
        public double m_dTagBigramTransitionWeight = 1.0; // tag bigram transition probability weight

        public virtual void SetTagBigramTransitionWeight(double w)
        {
            m_dTagBigramTransitionWeight = w;
        }

        public virtual void setTagBigramTransition(List<List<double>> m)
        {
            if (null == m_tagBigramTransition)
                m_tagBigramTransition = new Matrix(L2, L2);

            for (int i = 0; i < L2; i++)
                for (int j = 0; j < L2; j++)
                    m_tagBigramTransition[i][j] = m[i][j];

            m_tagBigramTransition_alpha = new Matrix(L2, L2);
        }

        //Save matrix into file as binary format
        protected void saveMatrixBin(Matrix mat, BinaryWriter fo)
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

        protected Matrix loadMatrixBin(BinaryReader br)
        {
            int width = br.ReadInt32();
            int height = br.ReadInt32();

            Matrix m = new Matrix(height, width);

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
            // set runtime feature
            for (int i = 0; i < state.GetNumRuntimeFeature(); i++)
            {
                int pos = curState + ((forward == true) ? 1 : -1) * state.GetRuntimeFeature(i).OffsetToCurrentState;
                if (pos >= 0 && pos < numStates)
                {
                    state.SetRuntimeFeature(i, predicted[pos], 1);
                }
            }

            var dense = state.GetDenseData();
            for (int i = 0; i < dense.GetDimension(); i++)
            {
                neuFeatures[i].ac = dense[i];
            }
        }

        public long m_SaveStep;
        public virtual void SetSaveStep(long savestep)
        {
            m_SaveStep = savestep;
        }

        public int m_MaxIter;
        public virtual void SetMaxIter(int _nMaxIter)
        {
            m_MaxIter = _nMaxIter;
        }

        public RNN()
        {
            randrng = 0.1f;
            gradient_cutoff = 15;

            alpha = 0.1;
            beta = 0.0000001;
            logp = 0;
            llogp = -100000000;
            minTknErrRatio = 1000000;
            lastTknErrRatio = 1000000;

            iter = 0;
            final_stop = 0;

            L1 = 30;

            fea_size = 0;

            neuInput = null;
            neuFeatures = null;
            neuHidden = null;
            neuOutput = null;
        }

        public void SetModelDirection(int dir)
        {
            m_modeldirection = (MODELDIRECTION)dir;
        }

        public virtual void SetTrainingSet(DataSet train)
        {
            m_TrainingSet = train;
            fea_size = m_TrainingSet.GetDenseDimension();
            L0 = m_TrainingSet.GetSparseDimension() + L1;
            L2 = m_TrainingSet.GetTagSize();
        }

        public bool ShouldTrainingStop()
        {
            return (final_stop == 1) ? true : false;
        }

        public virtual void SetCRFTraining(bool b) { m_bCRFTraining = b; }
        public virtual void SetValidationSet(DataSet validation) { m_ValidationSet = validation; }
        public virtual void SetGradientCutoff(double newGradient) { gradient_cutoff = newGradient; }
        public virtual void SetLearningRate(double newAlpha) { alpha = newAlpha; }
        public virtual void SetRegularization(double newBeta) { beta = newBeta; }
        public virtual void SetHiddenLayerSize(int newsize) { L1 = newsize; if (null != m_TrainingSet) L0 = (int)m_TrainingSet.GetSparseDimension() + L1; }
        public virtual void SetModelFile(string strModelFile) { m_strModelFile = strModelFile; }
        public virtual void SetDynAlpha(bool b) { m_bDynAlpha = b; }

        public bool IsCRFModel()
        {
            return m_bCRFTraining;
        }

        public double exp_10(double num) { return Math.Exp(num * 2.302585093); }

        public abstract void netReset();
        public abstract void copyHiddenLayerToInput();
        public abstract void computeNet(State state, double[] doutput);


        public virtual int[] PredictSentence(Sequence pSequence)
        {
            int numStates = pSequence.GetSize();
            int[] predicted = new int[numStates];
            netReset();
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.Get(curState);
                setInputLayer(state, curState, numStates, predicted);

                computeNet(state, null);      //compute probability distribution neu2[...].ac
                logp += Math.Log10(neuOutput[state.GetLabel()].ac);

                predicted[curState] = GetBestOutputIndex();

                counter++;

                // error propogation
                learnNet(state, curState);
                LearnBackTime(state, numStates, curState);


                copyHiddenLayerToInput();

            }

            return predicted;
        }

        public int GetBestOutputIndex()
        {
            int imax = 0;
            double dmax = neuOutput[0].ac;
            for (int k = 1; k < L2; k++)
            {
                if (neuOutput[k].ac > dmax)
                {
                    dmax = neuOutput[k].ac;
                    imax = k;
                }
            }
            return imax;
        }

        public virtual int[] learnSentenceForRNNCRF(Sequence pSequence)
        {
            //Reset the network
            netReset();

            int numStates = pSequence.GetSize();

            int[] predicted_nn = new int[numStates];
            m_RawOutput = new Matrix(numStates, L2);// new double[numStates][];
            for (int curState = 0; curState < numStates; curState++)
            {
         //       m_RawOutput[curState] = new double[L2];
                State state = pSequence.Get(curState);

                setInputLayer(state, curState, numStates, predicted_nn);
                computeNet(state, m_RawOutput[curState]);      //compute probability distribution

                predicted_nn[curState] = GetBestOutputIndex();
                copyHiddenLayerToInput();
            }

            ForwardBackward(numStates, m_RawOutput);

            //Get the best result
            int []predicted = new int[numStates];
            for (int i = 0; i < numStates; i++)
            {
                State state = pSequence.Get(i);
                logp += Math.Log10(m_Diff[i][state.GetLabel()]);
                counter++;

                predicted[i] = GetBestZIndex(i);
            }

            UpdateBigramTransition(pSequence);

            netReset();
            for (int curState = 0; curState < numStates;curState++)
            {
                // error propogation
                State state = pSequence.Get(curState);
                setInputLayer(state, curState, numStates, predicted_nn);
                computeNet(state, m_RawOutput[curState]);      //compute probability distribution

                learnNet(state, curState);
                LearnBackTime(state, numStates, curState);
                copyHiddenLayerToInput();
            }

                return predicted;
        }

        // update learning rate per element
        public double calcAlpha(Matrix mg, int i, int j, double fg)
        {
            if (m_bDynAlpha == true)
            {
                double dg = mg[i][j] + fg * fg;
                mg[i][j] = dg;
                return alpha / (md_beta + Math.Sqrt(dg));
            }
            else
            {
                return alpha;
            }
        }

        public void UpdateWeights(Matrix weights, Matrix delta, Matrix weights_alpha)
        {
            Parallel.For(0, weights.GetHeight(), parallelOption, b =>
            {
                for (int a = 0; a < weights.GetWidth(); a++)
                {
                    double dlr = calcAlpha(weights_alpha, b, a, delta[b][a]);
                    if ((counterTokenForLM % 10) == 0)
                    {
                        weights[b][a] += dlr * (delta[b][a] - weights[b][a] * beta);
                    }
                    else
                    {
                        weights[b][a] += dlr * delta[b][a];
                    }

                    delta[b][a] = 0;
                }
            });
        }

        public void UpdateBigramTransition(Sequence seq)
        {
            //Clean up the detla data for tag bigram LM
            m_DeltaBigramLM.Reset();
            int numStates = seq.GetSize();

            for (int timeat = 1; timeat < numStates; timeat++)
            {
                for (int i = 0; i < L2; i++)
                {
                    for (int j = 0; j < L2; j++)
                    {
                        m_DeltaBigramLM[i][j] -= (m_tagBigramTransition[i][j] * m_Diff[timeat][i] * m_Diff[timeat - 1][j]);
                    }
                }

                int iTagId = seq.Get(timeat).GetLabel();
                int iLastTagId = seq.Get(timeat - 1).GetLabel();
                m_DeltaBigramLM[iTagId][iLastTagId] += 1;
            }

            counterTokenForLM++;

            //Update tag Bigram LM
            UpdateWeights(m_tagBigramTransition, m_DeltaBigramLM, m_tagBigramTransition_alpha);
        }

        public int GetBestZIndex(int currStatus)
        {
            //Get the output tag
            int imax = 0;
            double dmax = m_Diff[currStatus][0];
            for (int j = 1; j < L2; j++)
            {
                if (m_Diff[currStatus][j] > dmax)
                {
                    dmax = m_Diff[currStatus][j];
                    imax = j;
                }
            }
            return imax;
        }

        public void ForwardBackward(int numStates, Matrix m_RawOutput)
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
                            double fbgm = m_tagBigramTransition[j][k];
                            double finit = alphaSet[i - 1][k];
                            double ftmp = m_dTagBigramTransitionWeight * fbgm + finit;

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
                            double fbgm = m_tagBigramTransition[k][j];
                            double finit = betaSet[i + 1][k];
                            double ftmp = m_dTagBigramTransitionWeight * fbgm + finit;

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
            for (int i = 0; i < numStates; i++)
            {
                for (int j = 0; j < L2; j++)
                {
                    m_Diff[i][j] = Math.Exp(alphaSet[i][j] + betaSet[i][j] - m_RawOutput[i][j] - Z_);
                }
            }
        }

        public void computeHiddenActivity()
        {
            for (int a = 0; a < L1; a++)
            {
                if (neuHidden[a].ac > 50) neuHidden[a].ac = 50;  //for numerical stability
                if (neuHidden[a].ac < -50) neuHidden[a].ac = -50;  //for numerical stability
                neuHidden[a].ac = 1.0 / (1.0 + Math.Exp(-neuHidden[a].ac));
            }
        }

        public virtual void initWeights()
        {
            int b, a;
            for (b = 0; b < L1; b++)
            {
                for (a = 0; a < L0 - L1; a++)
                {
                    mat_input2hidden[b][a] = random(-randrng, randrng) + random(-randrng, randrng) + random(-randrng, randrng);
                }
            }


            for (b = 0; b < L1; b++)
            {
                for (a = 0; a < fea_size; a++)
                {
                    mat_feature2hidden[b][a] = random(-randrng, randrng) + random(-randrng, randrng) + random(-randrng, randrng);

                }
            }

            for (b = 0; b < L2; b++)
            {
                for (a = 0; a < fea_size; a++)
                {
                    mat_feature2output[b][a] = random(-randrng, randrng) + random(-randrng, randrng) + random(-randrng, randrng);

                }
            }

            for (b = 0; b < L2; b++)
            {
                for (a = 0; a < L1; a++)
                {
                    mat_hidden2output[b][a] = random(-randrng, randrng) + random(-randrng, randrng) + random(-randrng, randrng);
                }
            }
        }

        public double random(double min, double max)
        {
            return rand.NextDouble() * (max - min) + min;
        }

        public abstract void learnNet(State state, int timeat);
        public abstract void LearnBackTime(State state, int numStates, int curState);

        public virtual void TrainNet()
        {
            DateTime start = DateTime.Now;
            int[] predicted;
            Console.WriteLine("[TRACE] Iter " + iter + " begins with learning rate alpha = " + alpha + " ...");

            //Initialize varibles
            counter = 0;
            logp = 0;
            counterTokenForLM = 0;

            netFlush();

            //Shffle training corpus
            m_TrainingSet.Shuffle();

            int numSequence = m_TrainingSet.GetSize();
            int tknErrCnt = 0;
            int sentErrCnt = 0;
            Console.WriteLine("[TRACE] Progress = 0/" + numSequence / 1000.0 + "K\r");
            for (int curSequence = 0; curSequence < numSequence; curSequence++)
            {
                Sequence pSequence = m_TrainingSet.Get(curSequence);
                int numStates = pSequence.GetSize();

                if (numStates < 3)
                    continue;

                if (m_bCRFTraining == true)
                {
                    predicted = learnSentenceForRNNCRF(pSequence);
                }
                else
                {
                    predicted = PredictSentence(pSequence);
                }

                int newTknErrCnt = GetErrorTokenNum(pSequence, predicted);
                tknErrCnt += newTknErrCnt;
                if (newTknErrCnt > 0)
                {
                    sentErrCnt++;
                }

                if ((curSequence + 1) % 1000 == 0)
                {
                    Console.WriteLine("[TRACE] Progress = {0} ", (curSequence + 1) / 1000 + "K/" + numSequence / 1000.0 + "K");
                    Console.WriteLine(" train cross-entropy = {0} ", -logp / Math.Log10(2.0) / counter);
                    Console.WriteLine(" Error token ratio = {0}%", (double)tknErrCnt / (double)counter * 100);
                    Console.WriteLine(" Error sentence ratio = {0}%", (double)sentErrCnt / (double)curSequence * 100);
                }

                if (m_SaveStep > 0 && (curSequence + 1) % m_SaveStep == 0)
                {
                    //After processed every m_SaveStep sentences, save current model into a temporary file
                    Console.Write("Saving temporary model into file...");
                    saveNetBin(m_strModelFile + ".tmp");
                    Console.WriteLine("Done.");
                }
            }

            DateTime now = DateTime.Now;
            TimeSpan duration = now.Subtract(start);

            double entropy = -logp / Math.Log10(2.0) / counter;
            double ppl = exp_10(-logp / counter);
            Console.WriteLine("[TRACE] Iter " + iter + " completed");
            Console.WriteLine("[TRACE] Sentences = " + numSequence + ", time escape = " + duration + "s, speed = " + numSequence / duration.TotalSeconds);
            Console.WriteLine("[TRACE] In training: log probability = " + logp + ", cross-entropy = " + entropy + ", perplexity = " + ppl);
        }

        public virtual void initMem()
        {
            CreateCells();

            mat_input2hidden = new Matrix(L1, L0 - L1);
            mat_input2hidden_alpha = new Matrix(L1, L0 - L1);
            mat_feature2hidden = new Matrix(L1, fea_size);
            mat_feature2hidden_alpha = new Matrix(L1, fea_size);
            mat_feature2output = new Matrix(L2, fea_size);
            mat_feature2output_alpha = new Matrix(L2, fea_size);
            mat_hidden2output = new Matrix(L2, L1);
            mat_hidden2output_alpha = new Matrix(L2, L1);

            Console.WriteLine("[TRACE] Initializing weights, random value is {0}", random(-1.0, 1.0));// yy debug
            initWeights();

            for (int i = 0; i < MAX_RNN_HIST; i++)
            {
                m_Diff[i] = new double[L2];
            }

            m_tagBigramTransition = new Matrix(L2, L2);
            m_tagBigramTransition_alpha = new Matrix(L2, L2);
            m_DeltaBigramLM = new Matrix(L2, L2);

        }

        protected void CreateCells()
        {

            neuInput = new neuron[L0];
            neuFeatures = new neuron[fea_size];
            neuHidden = new neuron[L1];
            neuOutput = new neuron[L2];

            for (int a = 0; a < L0; a++)
            {
                neuInput[a].ac = 0;
                neuInput[a].er = 0;
            }

            for (int a = 0; a < fea_size; a++)
            {
                neuFeatures[a].ac = 0;
                neuFeatures[a].er = 0;
            }

            for (int a = 0; a < L1; a++)
            {
                neuHidden[a].ac = 0;
                neuHidden[a].er = 0;
            }

            for (int a = 0; a < L2; a++)
            {
                neuOutput[a].ac = 0;
                neuOutput[a].er = 0;
            }
        }


        public abstract void saveNetBin(string filename);
        public abstract void loadNetBin(string filename);

        public static MODELTYPE CheckModelFileType(string filename)
        {
            StreamReader sr = new StreamReader(filename);
            BinaryReader br = new BinaryReader(sr.BaseStream);

            MODELTYPE type = (MODELTYPE)br.ReadInt32();

            return type;
        }

        public void matrixXvectorADD(neuron[] dest, neuron[] srcvec, Matrix srcmatrix, int from, int to, int from2, int to2, int type)
        {
            if (type == 0)
            {		
                //ac mod
                Parallel.For(0, (to - from), parallelOption, i =>
                {
                    for (int j = 0; j < to2 - from2; j++)
                    {
                        dest[i + from].ac += srcvec[j + from2].ac * srcmatrix[i][j];
                    }
                });

            }
            else
            {
                Parallel.For(0, (to2 - from2), parallelOption, i =>
                {
                    for (int j = 0; j < to - from; j++)
                    {
                        dest[i + from2].er += srcvec[j + from].er * srcmatrix[j][i];
                    }
                });

                if (gradient_cutoff > 0)
                {
                    for (int i = from2; i < to2; i++)
                    {
                        if (dest[i].er > gradient_cutoff)
                            dest[i].er = gradient_cutoff;
                        if (dest[i].er < -gradient_cutoff)
                            dest[i].er = -gradient_cutoff;
                    }
                }
            }
        }

        public virtual void netFlush()   //cleans all activations and error vectors
        {
            int a;
            for (a = 0; a < L0 - L1; a++)
            {
                neuInput[a].ac = 0;
                neuInput[a].er = 0;
            }

            for (a = L0 - L1; a < L0; a++)
            {   
                //last hidden layer is initialized to vector of 0.1 values to prevent unstability
                neuInput[a].ac = 0.1;
                neuInput[a].er = 0;
            }

            for (a = 0; a < L1; a++)
            {
                neuHidden[a].ac = 0;
                neuHidden[a].er = 0;
            }

            for (a = 0; a < L2; a++)
            {
                neuOutput[a].ac = 0;
                neuOutput[a].er = 0;
            }
        }

        public int[] DecodeNN(Sequence seq)
        {
            Matrix ys = InnerDecode(seq);
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
            Matrix ys = InnerDecode(seq);

            int n = seq.GetSize();
            int K = L2;
            Matrix STP = m_tagBigramTransition;
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
            Matrix ys = InnerDecode(seq);

            int n = seq.GetSize();
            int K = L2;
            Matrix STP = m_tagBigramTransition;
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


        public virtual Matrix InnerDecode(Sequence pSequence)
        {
            //Reset the network
            netReset();
            int numStates = pSequence.GetSize();

            Matrix m = new Matrix(numStates, L2);
            int[] predicted = new int[numStates];
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.Get(curState);
                setInputLayer(state, curState, numStates, predicted);
                computeNet(state, m[curState]);      //compute probability distribution

                copyHiddenLayerToInput();
                predicted[curState] = GetBestOutputIndex();
            }

            return m;
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

        public virtual bool ValidateNet()
        {
            Console.WriteLine("[TRACE] Start validation ...");
            int wordcn = 0;
            int[] predicted;
            int tknErrCnt = 0;
            int sentErrCnt = 0;

            netFlush();            
            int numSequence = m_ValidationSet.GetSize();
            for (int curSequence = 0; curSequence < numSequence; curSequence++)
            {
                Sequence pSequence = m_ValidationSet.Get(curSequence);
                wordcn += pSequence.GetSize();

                if (m_bCRFTraining == true)
                {
                    predicted = DecodeCRF(pSequence);
                }
                else
                {
                    predicted = DecodeNN(pSequence);
                }

                int newTknErrCnt = GetErrorTokenNum(pSequence, predicted);
                tknErrCnt += newTknErrCnt;
                if (newTknErrCnt > 0)
                {
                    sentErrCnt++;
                }
            }

            Console.WriteLine("[TRACE] In validation: error token ratio = {0}% error sentence ratio = {1}%", (double)tknErrCnt / (double)wordcn * 100, (double)sentErrCnt / (double)numSequence * 100);
            Console.WriteLine();

            bool bUpdate = false;
            double TknErrRatio = (double)tknErrCnt / (double)wordcn * 100;
            if (TknErrRatio < minTknErrRatio)
            {
                bUpdate = true;
                minTknErrRatio = TknErrRatio;
            }

            if (TknErrRatio > lastTknErrRatio)
            {
                //Reduce the learning rate as half than before
                alpha /= 2;
            }

            lastTknErrRatio = TknErrRatio;

            iter++;
            if (m_MaxIter > 0 && iter >= m_MaxIter)
            {
                final_stop = 1;
            }

            return bUpdate;
        }      
    }
}
