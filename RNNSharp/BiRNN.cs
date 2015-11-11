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

            forwardRNN.mat_hidden2output = mat_hidden2output;
            backwardRNN.mat_hidden2output = mat_hidden2output;

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
        }

        public override void GetHiddenLayer(Matrix m, int curStatus)
        {
            throw new NotImplementedException("Not implement GetHiddenLayer");
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


            //Create and intialise the weights from hidden to output layer, these are just normal weights
            double hiddenOutputRand = 1 / Math.Sqrt((double)L1);
            mat_hidden2output = new Matrix(L2, L1 + 1);

            for (int i = 0; i < mat_hidden2output.GetHeight(); i++)
            {
                for (int j = 0; j < mat_hidden2output.GetWidth(); j++)
                {
                    mat_hidden2output[i][j] = (((double)((randNext() % 100) + 1) / 100) * 2 * hiddenOutputRand) - hiddenOutputRand;
                }
            }
        }



        public override Matrix InnerDecode(Sequence pSequence)
        {
            Matrix mHiddenLayer = null;
            Matrix mRawOutputLayer = null;
            neuron[][] outputLayer = InnerDecode(pSequence, out mHiddenLayer, out mRawOutputLayer);
            int numStates = pSequence.GetSize();

            Matrix m = new Matrix(numStates, L2);
            for (int currState = 0; currState < numStates; currState++)
            {
                for (int i = 0; i < L2; i++)
                {
                    m[currState][i] = outputLayer[currState][i].cellOutput;
                }
            }

            return m;
        }

        int[] predicted_fnn;
        int[] predicted_bnn;
        public neuron[][] InnerDecode(Sequence pSequence, out Matrix outputHiddenLayer, out Matrix rawOutputLayer)
        {
            //Reset the network
            netReset();
            int numStates = pSequence.GetSize();
            predicted_fnn = new int[numStates];
            predicted_bnn = new int[numStates];
            Matrix mForward = new Matrix(numStates, forwardRNN.L1 + 1);
            Matrix mBackward = new Matrix(numStates, backwardRNN.L1 + 1);

            Parallel.Invoke(() =>
            {
                //Computing forward RNN
                for (int curState = 0; curState < numStates; curState++)
                {
                    State state = pSequence.Get(curState);
                    forwardRNN.setInputLayer(state, curState, numStates, predicted_fnn);
                    forwardRNN.computeNet(state, mForward[curState]);      //compute probability distribution

                    forwardRNN.GetHiddenLayer(mForward, curState);

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

                     backwardRNN.GetHiddenLayer(mBackward, curState);

                     predicted_bnn[curState] = backwardRNN.GetBestOutputIndex();

                     backwardRNN.copyHiddenLayerToInput();
                 }
             });

            //Merge forward and backward
            Matrix mergedHiddenLayer = new Matrix(numStates, forwardRNN.L1 + 1);
            for (int curState = 0; curState < numStates; curState++)
            {
                for (int i = 0; i <= forwardRNN.L1; i++)
                {
                    mergedHiddenLayer[curState][i] = mForward[curState][i] + mBackward[curState][i];
                }
            }

            rawOutputLayer = new Matrix(numStates, L2);

            neuron[][] seqOutput = new neuron[numStates][];
            for (int curStatus = 0; curStatus < numStates; curStatus++)
            {
                seqOutput[curStatus] = new neuron[L2];

                neuron[] tempHiddenLayer = new neuron[L1];
                for (int c = 0; c < L1; c++)
                {
                    tempHiddenLayer[c].cellOutput = mergedHiddenLayer[curStatus][c];
                }

                matrixXvectorADD(seqOutput[curStatus], tempHiddenLayer, mat_hidden2output, 0, L2, 0, L1, 0);

                for (int i = 0; i < L2; i++)
                {
                    rawOutputLayer[curStatus][i] = seqOutput[curStatus][i].cellOutput;
                }

                //activation 2   --softmax on words
                SoftmaxLayer(seqOutput[curStatus]);
            }

            outputHiddenLayer = mergedHiddenLayer;

            return seqOutput;
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
            Matrix mergedHiddenLayer = null;
            Matrix rawOutputLayer = null;
            neuron[][] seqOutput = InnerDecode(pSequence, out mergedHiddenLayer, out rawOutputLayer);

            ForwardBackward(numStates, rawOutputLayer);

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

            //Update hidden-output layer weights
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.Get(curState);
                //For standard RNN
                for (int c = 0; c < L2; c++)
                {
                    seqOutput[curState][c].er = -m_Diff[curState][c];
                }
                seqOutput[curState][state.GetLabel()].er = 1 - m_Diff[curState][state.GetLabel()];
            }

            double[] output = new double[L2];
            //Learn forward network
            for (int curState = 0; curState < numStates; curState++)
            {
                counter++;

                int curState2 = numStates - 1 - curState;

                // error propogation
                State state = pSequence.Get(curState);
                forwardRNN.setInputLayer(state, curState, numStates, predicted_fnn);
                forwardRNN.computeNet(state, output);      //compute probability distribution

                //Copy output result to forward net work's output
                forwardRNN.neuOutput = seqOutput[curState];

                forwardRNN.learnNet(state, curState, true);
                forwardRNN.LearnBackTime(state, numStates, curState);
                forwardRNN.copyHiddenLayerToInput();

                // error propogation
                State state2 = pSequence.Get(curState2);
                backwardRNN.setInputLayer(state2, curState2, numStates, predicted_bnn, false);
                backwardRNN.computeNet(state2, output);      //compute probability distribution

                //Copy output result to forward net work's output
                backwardRNN.neuOutput = seqOutput[curState2];

                backwardRNN.learnNet(state2, curState2, true);
                backwardRNN.LearnBackTime(state2, numStates, curState2);
                backwardRNN.copyHiddenLayerToInput();

                for (int i = 0; i < mat_hidden2output.GetHeight(); i++)
                {
                    //update weights for hidden to output layer

                    for (int k = 0; k < mat_hidden2output.GetWidth(); k++)
                    {
                        if ((counter % 10) == 0)	//regularization is done every 10. step
                        {
                            mat_hidden2output[i][k] += alpha * (mergedHiddenLayer[curState][k] * seqOutput[curState][i].er - mat_hidden2output[i][k] * beta);
                        }
                        else
                        {
                            mat_hidden2output[i][k] += alpha * mergedHiddenLayer[curState][k] * seqOutput[curState][i].er;
                        }
                    }
                }

            }

            return predicted;
        }

        public override int[] PredictSentence(Sequence pSequence)
        {
            //Reset the network
            int numStates = pSequence.GetSize();
            int[] predicted = new int[numStates];

            //Predict output
            Matrix mergedHiddenLayer = null;
            Matrix rawOutputLayer = null;
            neuron[][] seqOutput = InnerDecode(pSequence, out mergedHiddenLayer, out rawOutputLayer);

            //Merge forward and backward
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.Get(curState);
                logp += Math.Log10(seqOutput[curState][state.GetLabel()].cellOutput);

                predicted[curState] = GetBestOutputIndex(seqOutput, curState, L2);
            }

            netReset();

            //Update hidden-output layer weights
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.Get(curState);
                //For standard RNN
                for (int c = 0; c < L2; c++)
                {
                    seqOutput[curState][c].er = -seqOutput[curState][c].cellOutput;
                }
                seqOutput[curState][state.GetLabel()].er = 1 - seqOutput[curState][state.GetLabel()].cellOutput;
            }

            double[] output = new double[L2];
            //Learn forward network
            for (int curState = 0; curState < numStates; curState++)
            {
                counter++;

                int curState2 = numStates - 1 - curState;

                // error propogation
                State state = pSequence.Get(curState);
                forwardRNN.setInputLayer(state, curState, numStates, predicted_fnn);
                forwardRNN.computeNet(state, output);      //compute probability distribution

                //Copy output result to forward net work's output
                forwardRNN.neuOutput = seqOutput[curState];

                forwardRNN.learnNet(state, curState, true);
                forwardRNN.LearnBackTime(state, numStates, curState);
                forwardRNN.copyHiddenLayerToInput();

                // error propogation
                State state2 = pSequence.Get(curState2);
                backwardRNN.setInputLayer(state2, curState2, numStates, predicted_bnn, false);
                backwardRNN.computeNet(state2, output);      //compute probability distribution

                //Copy output result to forward net work's output
                backwardRNN.neuOutput = seqOutput[curState2];

                backwardRNN.learnNet(state2, curState2, true);
                backwardRNN.LearnBackTime(state2, numStates, curState2);
                backwardRNN.copyHiddenLayerToInput();

                for (int i = 0; i < mat_hidden2output.GetHeight(); i++)
                {
                    //update weights for hidden to output layer

                    for (int k = 0; k < mat_hidden2output.GetWidth(); k++)
                    {
                        if ((counter % 10) == 0)	//regularization is done every 10. step
                        {
                            mat_hidden2output[i][k] += alpha * (mergedHiddenLayer[curState][k] * seqOutput[curState][i].er - mat_hidden2output[i][k] * beta);
                        }
                        else
                        {
                            mat_hidden2output[i][k] += alpha * mergedHiddenLayer[curState][k] * seqOutput[curState][i].er;
                        }
                    }
                }

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


        public int GetBestOutputIndex(neuron[][] m, int curState, int L2)
        {
            int imax = 0;
            double dmax = m[curState][0].cellOutput;
            for (int k = 1; k < L2; k++)
            {
                if (m[curState][k].cellOutput > dmax)
                {
                    dmax = m[curState][k].cellOutput;
                    imax = k;
                }
            }
            return imax;
        }

        public override void LearnBackTime(State state, int numStates, int curState)
        {
        }

        public override void learnNet(State state, int timeat, bool biRNN = false)
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
            forwardRNN.mat_hidden2output = mat_hidden2output;
            backwardRNN.mat_hidden2output = mat_hidden2output;

            forwardRNN.saveNetBin(filename + ".forward");
            backwardRNN.saveNetBin(filename + ".backward");
        }

        public override void loadNetBin(string filename)
        {
            forwardRNN.loadNetBin(filename + ".forward");
            backwardRNN.loadNetBin(filename + ".backward");

            mat_hidden2output = forwardRNN.mat_hidden2output;
        }
    }
}
