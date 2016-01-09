using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AdvUtils;

namespace RNNSharp
{
    class BiRNN : RNN
    {
        private RNN forwardRNN;
        private RNN backwardRNN;

        public BiRNN(RNN s_forwardRNN, RNN s_backwardRNN)
        {
            forwardRNN = s_forwardRNN;
            backwardRNN = s_backwardRNN;

            m_modeldirection = MODELDIRECTION.BI_DIRECTIONAL;
        }

        public override void SetFeatureDimension(int denseFeatueSize, int sparseFeatureSize, int tagSize)
        {
            fea_size = denseFeatueSize;
            L0 = sparseFeatureSize;
            L2 = tagSize;

            forwardRNN.SetFeatureDimension(denseFeatueSize, sparseFeatureSize, tagSize);
            backwardRNN.SetFeatureDimension(denseFeatueSize, sparseFeatureSize, tagSize);
        }


        public override void initWeights()
        {
            forwardRNN.initWeights();
            backwardRNN.initWeights();

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

        public override void GetHiddenLayer(Matrix<double> m, int curStatus)
        {
            throw new NotImplementedException("Not implement GetHiddenLayer");
        }

        public override void initMem()
        {
            forwardRNN.initMem();
            backwardRNN.initMem();

            //Create and intialise the weights from hidden to output layer, these are just normal weights
            mat_hidden2output = new Matrix<double>(L2, L1);

            for (int i = 0; i < mat_hidden2output.GetHeight(); i++)
            {
                for (int j = 0; j < mat_hidden2output.GetWidth(); j++)
                {
                    mat_hidden2output[i][j] = RandInitWeight();
                }
            }
        }

        public neuron[][] InnerDecode(Sequence pSequence, out Matrix<neuron> outputHiddenLayer, out Matrix<double> rawOutputLayer)
        {
            int numStates = pSequence.GetSize();
            Matrix<double> mForward = null;
            Matrix<double> mBackward = null;

            //Reset the network
            netReset(false);

            Parallel.Invoke(() =>
            {
                //Computing forward RNN
                mForward = new Matrix<double>(numStates, forwardRNN.L1);
                for (int curState = 0; curState < numStates; curState++)
                {
                    State state = pSequence.Get(curState);
                    forwardRNN.setInputLayer(state, curState, numStates, null);
                    forwardRNN.computeNet(state, null);      //compute probability distribution

                    forwardRNN.GetHiddenLayer(mForward, curState);
                }
            },
             () =>
             {
                 //Computing backward RNN
                 mBackward = new Matrix<double>(numStates, backwardRNN.L1);
                 for (int curState = numStates - 1; curState >= 0; curState--)
                 {
                     State state = pSequence.Get(curState);
                     backwardRNN.setInputLayer(state, curState, numStates, null, false);
                     backwardRNN.computeNet(state, null);      //compute probability distribution

                     backwardRNN.GetHiddenLayer(mBackward, curState);
                 }
             });

            //Merge forward and backward
            Matrix<neuron> mergedHiddenLayer = new Matrix<neuron>(numStates, forwardRNN.L1);
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                for (int i = 0; i < forwardRNN.L1; i++)
                {
                    mergedHiddenLayer[curState][i].cellOutput = mForward[curState][i] + mBackward[curState][i];
                }
            });

            //Calculate output layer
            Matrix<double> tmp_rawOutputLayer = new Matrix<double>(numStates, L2);
            neuron[][] seqOutput = new neuron[numStates][];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                seqOutput[curState] = new neuron[L2];
                matrixXvectorADD(seqOutput[curState], mergedHiddenLayer[curState], mat_hidden2output, 0, L2, 0, L1, 0);

                for (int i = 0; i < L2; i++)
                {
                    tmp_rawOutputLayer[curState][i] = seqOutput[curState][i].cellOutput;
                }

                //Activation on output layer
                SoftmaxLayer(seqOutput[curState]);
            });

            outputHiddenLayer = mergedHiddenLayer;
            rawOutputLayer = tmp_rawOutputLayer;

            return seqOutput;
        }

        public override Matrix<double> learnSentenceForRNNCRF(Sequence pSequence, RunningMode runningMode)
        {
            //Reset the network
            int numStates = pSequence.GetSize();
            //Predict output
            Matrix<neuron> mergedHiddenLayer = null;
            Matrix<double> rawOutputLayer = null;
            neuron[][] seqOutput = InnerDecode(pSequence, out mergedHiddenLayer, out rawOutputLayer);

            ForwardBackward(numStates, rawOutputLayer);

            //Get the best result
            for (int i = 0; i < numStates; i++)
            {
                State state = pSequence.Get(i);
                logp += Math.Log10(mat_CRFSeqOutput[i][state.GetLabel()]);
                counter++;
            }

            UpdateBigramTransition(pSequence);

            //Update hidden-output layer weights
            for (int curState = 0; curState < numStates; curState++)
            {
                State state = pSequence.Get(curState);
                //For standard RNN
                for (int c = 0; c < L2; c++)
                {
                    seqOutput[curState][c].er = -mat_CRFSeqOutput[curState][c];
                }
                seqOutput[curState][state.GetLabel()].er = 1 - mat_CRFSeqOutput[curState][state.GetLabel()];
            }

            LearnTwoRNN(pSequence, mergedHiddenLayer, seqOutput);

            return mat_CRFSeqOutput;
        }

        public override Matrix<double> PredictSentence(Sequence pSequence, RunningMode runningMode)
        {
            //Reset the network
            int numStates = pSequence.GetSize();

            //Predict output
            Matrix<neuron> mergedHiddenLayer = null;
            Matrix<double> rawOutputLayer = null;
            neuron[][] seqOutput = InnerDecode(pSequence, out mergedHiddenLayer, out rawOutputLayer);

            if (runningMode != RunningMode.Test)
            {
                //Merge forward and backward
                for (int curState = 0; curState < numStates; curState++)
                {
                    State state = pSequence.Get(curState);
                    logp += Math.Log10(seqOutput[curState][state.GetLabel()].cellOutput);
                    counter++;
                }
            }

            if (runningMode == RunningMode.Train)
            {
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
            }

            return rawOutputLayer;
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
                    // error propogation
                    State state = pSequence.Get(curState);
                    forwardRNN.setInputLayer(state, curState, numStates, null);
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
                    int curState2 = numStates - 1 - curState;

                    // error propogation
                    State state2 = pSequence.Get(curState2);
                    backwardRNN.setInputLayer(state2, curState2, numStates, null, false);
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
            //Save bi-directional model
            forwardRNN.mat_hidden2output = mat_hidden2output;
            backwardRNN.mat_hidden2output = mat_hidden2output;

            forwardRNN.mat_CRFTagTransWeights = mat_CRFTagTransWeights;
            backwardRNN.mat_CRFTagTransWeights = mat_CRFTagTransWeights;

            forwardRNN.saveNetBin(filename + ".forward");
            backwardRNN.saveNetBin(filename + ".backward");

            //Save meta data
            using (StreamWriter sw = new StreamWriter(filename))
            {
                BinaryWriter fo = new BinaryWriter(sw.BaseStream);
                fo.Write((int)m_modeltype);
                fo.Write((int)m_modeldirection);

                // Signiture , 0 is for RNN or 1 is for RNN-CRF
                int iflag = 0;
                if (m_bCRFTraining == true)
                {
                    iflag = 1;
                }
                fo.Write(iflag);

                fo.Write(L0);
                fo.Write(L1);
                fo.Write(L2);
                fo.Write(fea_size);
            }
        }

        public override void loadNetBin(string filename)
        {
            Logger.WriteLine(Logger.Level.info, "Loading bi-directional model: {0}", filename);

            forwardRNN.loadNetBin(filename + ".forward");
            backwardRNN.loadNetBin(filename + ".backward");

            mat_hidden2output = forwardRNN.mat_hidden2output;
            mat_CRFTagTransWeights = forwardRNN.mat_CRFTagTransWeights;

            using (StreamReader sr = new StreamReader(filename))
            {
                BinaryReader br = new BinaryReader(sr.BaseStream);

                m_modeltype = (MODELTYPE)br.ReadInt32();
                m_modeldirection = (MODELDIRECTION)br.ReadInt32();

                int iflag = br.ReadInt32();
                if (iflag == 1)
                {
                    m_bCRFTraining = true;
                }
                else
                {
                    m_bCRFTraining = false;
                }

                //Load basic parameters
                L0 = br.ReadInt32();
                L1 = br.ReadInt32();
                L2 = br.ReadInt32();
                fea_size = br.ReadInt32();
            }
        }
    }
}
