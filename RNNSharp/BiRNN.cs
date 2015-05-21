using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    class BiRNN : RNN
    {
        private RNN forwardRNN;
        private RNN backwardRNN;

        public BiRNN(int modeltype)
        {
            if (modeltype == 0)
            {
                SimpleRNN s_forwardRNN = new SimpleRNN();
                SimpleRNN s_backwardRNN = new SimpleRNN();

                s_forwardRNN.setBPTT(4 + 1);
                s_forwardRNN.setBPTTBlock(30);

                s_backwardRNN.setBPTT(4 + 1);
                s_backwardRNN.setBPTTBlock(30);

                forwardRNN = s_forwardRNN;
                backwardRNN = s_backwardRNN;
            }
            else
            {
                forwardRNN = new LSTMRNN();
                backwardRNN = new LSTMRNN();
            }

            m_modeldirection = MODELDIRECTION.BI_DIRECTIONAL;
        }
        public override void SetTrainingSet(DataSet train)
        {
            m_TrainingSet = train;
            fea_size = m_TrainingSet.GetDenseDimension();
            L0 = m_TrainingSet.GetSparseDimension() + L1;
            L2 = m_TrainingSet.GetTagSize();

            forwardRNN.SetTrainingSet(train);
            backwardRNN.SetTrainingSet(train);
        }

        public override void SetValidationSet(DataSet validation)
        {
            m_ValidationSet = validation;

            forwardRNN.SetValidationSet(validation);
            backwardRNN.SetValidationSet(validation);
        }

        public override void SetModelFile(string strModelFile)
        {
            m_strModelFile = strModelFile;

            forwardRNN.SetModelFile(strModelFile);
            backwardRNN.SetModelFile(strModelFile);
        }

        public override void SetSaveStep(long savestep)
        {
            m_SaveStep = savestep;

            forwardRNN.SetSaveStep(savestep);
            backwardRNN.SetSaveStep(savestep);
        }

        public override void SetMaxIter(int _nMaxIter)
        {
            m_MaxIter = _nMaxIter;

            forwardRNN.SetMaxIter(_nMaxIter);
            backwardRNN.SetMaxIter(_nMaxIter);
        }


        public override void SetCRFTraining(bool b)
        {
            m_bCRFTraining = b;

            forwardRNN.SetCRFTraining(b);
            backwardRNN.SetCRFTraining(b);
        }

        public override void SetLearningRate(double newAlpha)
        {
            alpha = newAlpha;

            forwardRNN.SetLearningRate(newAlpha);
            backwardRNN.SetLearningRate(newAlpha);
        }

        public override void SetGradientCutoff(double newGradient)
        {
            gradient_cutoff = newGradient;

            forwardRNN.SetGradientCutoff(newGradient);
            backwardRNN.SetGradientCutoff(newGradient);
        }

        public override void SetRegularization(double newBeta)
        {
            beta = newBeta;

            forwardRNN.SetRegularization(newBeta);
            backwardRNN.SetRegularization(newBeta);
        }

        public override void SetHiddenLayerSize(int newsize)
        {
            L1 = newsize; if (null != m_TrainingSet) L0 = (int)m_TrainingSet.GetSparseDimension() + L1;

            forwardRNN.SetHiddenLayerSize(newsize);
            backwardRNN.SetHiddenLayerSize(newsize);
        }

        public override void SetTagBigramTransitionWeight(double w)
        {
            m_dTagBigramTransitionWeight = w;

            //forwardRNN.SetTagBigramTransitionWeight(w);
            //backwardRNN.SetTagBigramTransitionWeight(w);
        }

        public override void initMem()
        {
            for (int i = 0; i < MAX_RNN_HIST; i++)
            {
                m_Diff[i] = new double[L2];
            }

            m_tagBigramTransition = new Matrix(L2, L2);
            m_DeltaBigramLM = new Matrix(L2, L2);

            forwardRNN.initMem();
            backwardRNN.initMem();
        }

        //public virtual void setTagBigramTransition(List<List<double>> m)
        //{
        //    forwardRNN.setTagBigramTransition(m);
        //    backwardRNN.setTagBigramTransition(m);
        //}

        int[] predicted_fnn;
        int[] predicted_bnn;
        public override Matrix InnerDecode(Sequence pSequence)
        {
            //Reset the network
            netReset();
            int numStates = pSequence.GetSize();
            predicted_fnn = new int[numStates];
            predicted_bnn = new int[numStates];
            Matrix mForward = new Matrix(numStates, forwardRNN.L2);
            Matrix mBackward = new Matrix(numStates, backwardRNN.L2);



            Parallel.Invoke(() =>
            {
                //Computing forward RNN
                for (int curState = 0; curState < numStates; curState++)
                {
                    State state = pSequence.Get(curState);
                    forwardRNN.setInputLayer(state, curState, numStates, predicted_fnn);
                    forwardRNN.computeNet(state, mForward[curState]);      //compute probability distribution

                    predicted_fnn[curState] = forwardRNN.GetBestOutputIndex();

                    forwardRNN.copyHiddenLayerToInput();
                }
            },
             () =>
             {
                 //Computing backward RNN
                 for (int curState = numStates - 1; curState >= 0; curState--)
                 {
                     State state = pSequence.Get(curState);
                     backwardRNN.setInputLayer(state, curState, numStates, predicted_bnn, false);
                     backwardRNN.computeNet(state, mBackward[curState]);      //compute probability distribution

                     predicted_bnn[curState] = backwardRNN.GetBestOutputIndex();

                     backwardRNN.copyHiddenLayerToInput();
                 }
             });

            //Merge forward and backward
            Matrix m = new Matrix(numStates, forwardRNN.L2);
            for (int curState = 0; curState < numStates; curState++)
            {
                for (int i = 0; i < forwardRNN.L2; i++)
                {
                    m[curState][i] = (mForward[curState][i] + mBackward[curState][i]) / 2.0;
                }
            }

            return m;
        }

        public override void netFlush()
        {
            forwardRNN.netFlush();
            backwardRNN.netFlush();
        }

        public override int[] learnSentenceForRNNCRF(Sequence pSequence)
        {
            //Reset the network
            int numStates = pSequence.GetSize();
            int[] predicted = new int[numStates];

            //Predict output
            Matrix m = InnerDecode(pSequence);

            ForwardBackward(numStates, m);
            //Get the best result
            predicted = new int[numStates];
            for (int i = 0; i < numStates; i++)
            {
                State state = pSequence.Get(i);
                logp += Math.Log10(m_Diff[i][state.GetLabel()]);
                counter++;

                predicted[i] = GetBestZIndex(i);
            }

            UpdateBigramTransition(pSequence);

            netReset();

            forwardRNN.m_Diff = m_Diff;
            backwardRNN.m_Diff = m_Diff;

            double[] output_fnn = new double[L2];
            double[] output_bnn = new double[L2];


            Parallel.Invoke(() =>
            {
                //Learn forward network
                for (int curState = 0; curState < numStates; curState++)
                {
                    // error propogation
                    State state = pSequence.Get(curState);
                    forwardRNN.setInputLayer(state, curState, numStates, predicted_fnn);
                    forwardRNN.computeNet(state, output_fnn);      //compute probability distribution

                    forwardRNN.learnNet(state, curState);
                    forwardRNN.LearnBackTime(state, numStates, curState);
                    forwardRNN.copyHiddenLayerToInput();
                }
            },
             () =>
             {
                 for (int curState = numStates - 1; curState >= 0; curState--)
                 {
                     // error propogation
                     State state = pSequence.Get(curState);
                     backwardRNN.setInputLayer(state, curState, numStates, predicted_bnn, false);
                     backwardRNN.computeNet(state, output_bnn);      //compute probability distribution

                     backwardRNN.learnNet(state, curState);
                     backwardRNN.LearnBackTime(state, numStates, curState);
                     backwardRNN.copyHiddenLayerToInput();
                 }
             });

            return predicted;
        }

        public override int[] PredictSentence(Sequence pSequence)
        {

            //Reset the network
            int numStates = pSequence.GetSize();
            int[] predicted = new int[numStates];

            //Predict output
            Matrix m = InnerDecode(pSequence);

            //Merge forward and backward
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.Get(curState);
                //activation 2   --softmax on words
                double sum = 0;   //sum is used for normalization: it's better to have larger precision as many numbers are summed together here
                for (int c = 0; c < forwardRNN.L2; c++)
                {
                    if (m[curState][c] > 50) m[curState][c] = 50;  //for numerical stability
                    if (m[curState][c] < -50) m[curState][c] = -50;  //for numerical stability
                    double val = Math.Exp(m[curState][c]);
                    sum += val;
                    m[curState][c] = val;
                }

                for (int c = 0; c < forwardRNN.L2; c++)
                {
                    m[curState][c] /= sum;
                }

                logp += Math.Log10(m[curState][state.GetLabel()]);
                counter++;

                predicted[curState] = GetBestOutputIndex(m, curState);
            }

            netReset();

            double[] output = new double[L2];
            //Learn forward network
            for (int curState = 0; curState < numStates; curState++)
            {
                // error propogation
                State state = pSequence.Get(curState);
                forwardRNN.setInputLayer(state, curState, numStates, predicted_fnn);
                forwardRNN.computeNet(state, output);      //compute probability distribution

                //Copy output result to forward net work's output
                for (int i = 0; i < forwardRNN.L2; i++)
                {
                    forwardRNN.neuOutput[i].ac = m[curState][i];
                }

                forwardRNN.learnNet(state, curState);
                forwardRNN.LearnBackTime(state, numStates, curState);
                forwardRNN.copyHiddenLayerToInput();
            }

            for (int curState = numStates - 1; curState >= 0; curState--)
            {
                // error propogation
                State state = pSequence.Get(curState);
                backwardRNN.setInputLayer(state, curState, numStates, predicted_bnn, false);
                backwardRNN.computeNet(state, output);      //compute probability distribution

                //Copy output result to forward net work's output
                for (int i = 0; i < backwardRNN.L2; i++)
                {
                    backwardRNN.neuOutput[i].ac = m[curState][i];
                }

                backwardRNN.learnNet(state, curState);
                backwardRNN.LearnBackTime(state, numStates, curState);
                backwardRNN.copyHiddenLayerToInput();
            }

            return predicted;
        }

        public int GetBestOutputIndex(Matrix m, int curState)
        {
            int imax = 0;
            double dmax = m[curState][0];
            for (int k = 1; k < m.GetWidth(); k++)
            {
                if (m[curState][k] > dmax)
                {
                    dmax = m[curState][k];
                    imax = k;
                }
            }
            return imax;
        }

        public override void LearnBackTime(State state, int numStates, int curState)
        {
        }

        public override void learnNet(State state, int timeat)
        {

        }

        public override void computeNet(State state, double[] doutput)
        {

        }

        public override void copyHiddenLayerToInput()
        {

        }

        public override void netReset()
        {
            forwardRNN.netReset();
            backwardRNN.netReset();
        }

        public override void saveNetBin(string filename)
        {
            forwardRNN.saveNetBin(filename + ".forward");
            backwardRNN.saveNetBin(filename + ".backward");
        }

        public override void loadNetBin(string filename)
        {
            forwardRNN.loadNetBin(filename + ".forward");
            backwardRNN.loadNetBin(filename + ".backward");
        }
    }
}
