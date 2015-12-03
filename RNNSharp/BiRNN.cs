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
                s_forwardRNN.setBPTTBlock(10);

                s_backwardRNN.setBPTT(4 + 1);
                s_backwardRNN.setBPTTBlock(10);

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
            L0 = m_TrainingSet.GetSparseDimension();
            L2 = m_TrainingSet.GetTagSize();

            forwardRNN.SetTrainingSet(train);
            backwardRNN.SetTrainingSet(train);
        }

        public override void initWeights()
        {
            forwardRNN.initWeights();
            backwardRNN.initWeights();

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

        public override void SetDropout(double newDropout)
        {
            dropout = newDropout;

            forwardRNN.SetDropout(newDropout);
            backwardRNN.SetDropout(newDropout);
        }

        public override void SetHiddenLayerSize(int newsize)
        {
            L1 = newsize;

            forwardRNN.SetHiddenLayerSize(newsize);
            backwardRNN.SetHiddenLayerSize(newsize);
        }

        public override void SetTagBigramTransitionWeight(double w)
        {
            m_dTagBigramTransitionWeight = w;
        }

        public override void GetHiddenLayer(Matrix<double> m, int curStatus)
        {
            throw new NotImplementedException("Not implement GetHiddenLayer");
        }

        public override void initMem()
        {
            for (int i = 0; i < MAX_RNN_HIST; i++)
            {
                m_Diff[i] = new double[L2];
            }

            m_tagBigramTransition = new Matrix<double>(L2, L2);
            m_DeltaBigramLM = new Matrix<double>(L2, L2);

            forwardRNN.initMem();
            backwardRNN.initMem();


            //Create and intialise the weights from hidden to output layer, these are just normal weights
            double hiddenOutputRand = 1 / Math.Sqrt((double)L1);
            mat_hidden2output = new Matrix<double>(L2, L1 + 1);

            for (int i = 0; i < mat_hidden2output.GetHeight(); i++)
            {
                for (int j = 0; j < mat_hidden2output.GetWidth(); j++)
                {
                    mat_hidden2output[i][j] = (((double)((randNext() % 100) + 1) / 100) * 2 * hiddenOutputRand) - hiddenOutputRand;
                }
            }
        }



        public override Matrix<double> InnerDecode(Sequence pSequence)
        {
            Matrix<neuron> mHiddenLayer = null;
            Matrix<double> mRawOutputLayer = null;
            neuron[][] outputLayer = InnerDecode(pSequence, out mHiddenLayer, out mRawOutputLayer);
            int numStates = pSequence.GetSize();

            Matrix<double> m = new Matrix<double>(numStates, L2);
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
        public neuron[][] InnerDecode(Sequence pSequence, out Matrix<neuron> outputHiddenLayer, out Matrix<double> rawOutputLayer)
        {
            //Reset the network
            netReset(false);
            int numStates = pSequence.GetSize();
            predicted_fnn = new int[numStates];
            predicted_bnn = new int[numStates];
            Matrix<double> mForward = new Matrix<double>(numStates, forwardRNN.L1 + 1);
            Matrix<double> mBackward = new Matrix<double>(numStates, backwardRNN.L1 + 1);

            Parallel.Invoke(() =>
            {
                //Computing forward RNN
                for (int curState = 0; curState < numStates; curState++)
                {
                    State state = pSequence.Get(curState);
                    forwardRNN.setInputLayer(state, curState, numStates, predicted_fnn);
                    forwardRNN.computeNet(state, null);      //compute probability distribution

                    forwardRNN.GetHiddenLayer(mForward, curState);

                    predicted_fnn[curState] = forwardRNN.GetBestOutputIndex();
                }
            },
             () =>
             {
                 //Computing backward RNN
                 for (int curState = numStates - 1; curState >= 0; curState--)
                 {
                     State state = pSequence.Get(curState);
                     backwardRNN.setInputLayer(state, curState, numStates, predicted_bnn, false);
                     backwardRNN.computeNet(state, null);      //compute probability distribution

                     backwardRNN.GetHiddenLayer(mBackward, curState);

                     predicted_bnn[curState] = backwardRNN.GetBestOutputIndex();
                 }
             });

            //Merge forward and backward
            Matrix<neuron> mergedHiddenLayer = new Matrix<neuron>(numStates, forwardRNN.L1 + 1);
            for (int curState = 0; curState < numStates; curState++)
            {
                for (int i = 0; i <= forwardRNN.L1; i++)
                {
                    mergedHiddenLayer[curState][i].cellOutput = mForward[curState][i] + mBackward[curState][i];
                }
            }

            Matrix<double> tmp_rawOutputLayer = new Matrix<double>(numStates, L2);

            neuron[][] seqOutput = new neuron[numStates][];
            Parallel.For(0, numStates, parallelOption, curStatus =>
            {
                seqOutput[curStatus] = new neuron[L2];
                matrixXvectorADD(seqOutput[curStatus], mergedHiddenLayer[curStatus], mat_hidden2output, 0, L2, 0, L1, 0);

                for (int i = 0; i < L2; i++)
                {
                    tmp_rawOutputLayer[curStatus][i] = seqOutput[curStatus][i].cellOutput;
                }

                //activation 2   --softmax on words
                SoftmaxLayer(seqOutput[curStatus]);
            });

            outputHiddenLayer = mergedHiddenLayer;
            rawOutputLayer = tmp_rawOutputLayer;

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
            Matrix<neuron> mergedHiddenLayer = null;
            Matrix<double> rawOutputLayer = null;
            neuron[][] seqOutput = InnerDecode(pSequence, out mergedHiddenLayer, out rawOutputLayer);

            ForwardBackward(numStates, rawOutputLayer);

            //Get the best result
            predicted = new int[numStates];
            for (int i = 0; i < numStates; i++)
            {
                State state = pSequence.Get(i);
                logp += Math.Log10(m_Diff[i][state.GetLabel()]);
                predicted[i] = GetBestZIndex(i);
            }

            UpdateBigramTransition(pSequence);

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

            LearnTwoRNN(pSequence, mergedHiddenLayer, seqOutput);

            return predicted;
        }

        public override int[] PredictSentence(Sequence pSequence)
        {
            //Reset the network
            int numStates = pSequence.GetSize();
            int[] predicted = new int[numStates];

            //Predict output
            Matrix<neuron> mergedHiddenLayer = null;
            Matrix<double> rawOutputLayer = null;
            neuron[][] seqOutput = InnerDecode(pSequence, out mergedHiddenLayer, out rawOutputLayer);

            //Merge forward and backward
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.Get(curState);
                logp += Math.Log10(seqOutput[curState][state.GetLabel()].cellOutput);

                predicted[curState] = GetBestOutputIndex(seqOutput, curState, L2);
            }

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

            LearnTwoRNN(pSequence, mergedHiddenLayer, seqOutput);

            return predicted;
        }

        private void LearnTwoRNN(Sequence pSequence, Matrix<neuron> mergedHiddenLayer, neuron[][] seqOutput)
        {
            netReset(true);

            int numStates = pSequence.GetSize();
            forwardRNN.mat_hidden2output = mat_hidden2output.CopyTo();
            backwardRNN.mat_hidden2output = mat_hidden2output.CopyTo();



            Parallel.Invoke(() =>
                {
                    for (int curState = 0; curState < numStates; curState++)
                    {
                        for (int i = 0; i < mat_hidden2output.GetHeight(); i++)
                        {
                            //update weights for hidden to output layer

                            for (int k = 0; k < mat_hidden2output.GetWidth(); k++)
                            {
                                mat_hidden2output[i][k] += alpha * mergedHiddenLayer[curState][k].cellOutput * seqOutput[curState][i].er;
                            }
                        }
                    }

                },
                ()=>
            {

                //Learn forward network
                for (int curState = 0; curState < numStates; curState++)
                {
                    System.Threading.Interlocked.Increment(ref counter);

                    // error propogation
                    State state = pSequence.Get(curState);
                    forwardRNN.setInputLayer(state, curState, numStates, predicted_fnn);
                    forwardRNN.computeNet(state, null);      //compute probability distribution

                    //Copy output result to forward net work's output
                    forwardRNN.neuOutput = seqOutput[curState];

                    forwardRNN.learnNet(state, curState, true);
                    forwardRNN.LearnBackTime(state, numStates, curState);
                }
            },
            () =>
            {

                for (int curState = 0; curState < numStates; curState++)
                {
                    System.Threading.Interlocked.Increment(ref counter);

                    int curState2 = numStates - 1 - curState;

                    // error propogation
                    State state2 = pSequence.Get(curState2);
                    backwardRNN.setInputLayer(state2, curState2, numStates, predicted_bnn, false);
                    backwardRNN.computeNet(state2, null);      //compute probability distribution

                    //Copy output result to forward net work's output
                    backwardRNN.neuOutput = seqOutput[curState2];

                    backwardRNN.learnNet(state2, curState2, true);
                    backwardRNN.LearnBackTime(state2, numStates, curState2);
                }
            });
        }

        public int GetBestOutputIndex(Matrix<double> m, int curState)
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

        public override void computeNet(State state, double[] doutput, bool isTrain = true)
        {

        }

        public override void netReset(bool updateNet = false)
        {
            forwardRNN.netReset(updateNet);
            backwardRNN.netReset(updateNet);
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
