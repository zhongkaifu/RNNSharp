using System;
using System.IO;
using System.Threading.Tasks;
using AdvUtils;
using System.Collections.Generic;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
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

            ModelType = forwardRNN.ModelType;
            ModelDirection = MODELDIRECTION.BI_DIRECTIONAL;
        }

        public override int L0
        {
            get
            {
                return forwardRNN.L0;
            }

            set
            {
                forwardRNN.L0 = value;
                backwardRNN.L0 = value;
            }
        }

        public override int L2
        {
            get
            {
                return forwardRNN.L2;
            }

            set
            {
                forwardRNN.L2 = value;
                backwardRNN.L2 = value;
            }
        }

        public override void initWeights()
        {
            forwardRNN.initWeights();
            backwardRNN.initWeights();

        }

        public override string ModelFile
        {
            get { return forwardRNN.ModelFile; }
            set
            {
                forwardRNN.ModelFile = value;
                backwardRNN.ModelFile = value;
            }
        }

        public override long SaveStep
        {
            get
            {
                return forwardRNN.SaveStep;
            }

            set
            {
                forwardRNN.SaveStep = value;
                backwardRNN.SaveStep = value;
            }
        }

        public override int MaxIter
        {
            get
            {
                return forwardRNN.MaxIter;
            }

            set
            {
                forwardRNN.MaxIter = value;
                backwardRNN.MaxIter = value;
            }
        }

        public override bool IsCRFTraining
        {
            get { return forwardRNN.IsCRFTraining; }

            set
            {
                forwardRNN.IsCRFTraining = value;
                backwardRNN.IsCRFTraining = value;
            }
        }

        public override float LearningRate
        {
            get
            {
                return forwardRNN.LearningRate;
            }

            set
            {
                forwardRNN.LearningRate = value;
                backwardRNN.LearningRate = value;
            }
        }

        public override float GradientCutoff
        {
            get
            {
                return forwardRNN.GradientCutoff;
            }

            set
            {
                forwardRNN.GradientCutoff = value;
                backwardRNN.GradientCutoff = value;
            }
        }

        public override float Dropout
        {
            get
            {
                return forwardRNN.Dropout;
            }

            set
            {
                forwardRNN.Dropout = value;
                backwardRNN.Dropout = value;
            }
        }

        public override int L1
        {
            get
            {
                return forwardRNN.L1;
            }

            set
            {
                forwardRNN.L1 = value;
                backwardRNN.L1 = value;
            }
        }

        public override int DenseFeatureSize
        {
            get
            {
                return forwardRNN.DenseFeatureSize;
            }

            set
            {
                forwardRNN.DenseFeatureSize = value;
                backwardRNN.DenseFeatureSize = value;
            }
        }

        public override void SetHiddenLayer(SimpleCell[] cells)
        {
            throw new NotImplementedException("SetHiddenLayer is not implemented in BiRNN");
        }

        public override SimpleCell[] GetHiddenLayer()
        {
            throw new NotImplementedException("GetHiddenLayer is not implemented in BiRNN");
        }

        public override void initMem()
        {
            forwardRNN.initMem();
            backwardRNN.initMem();

            //Create and intialise the weights from hidden to output layer, these are just normal weights
            Hidden2OutputWeight = new Matrix<double>(L2, L1);

            for (int i = 0; i < Hidden2OutputWeight.Height; i++)
            {
                for (int j = 0; j < Hidden2OutputWeight.Width; j++)
                {
                    Hidden2OutputWeight[i][j] = RandInitWeight();
                }
            }
        }

        public SimpleCell[][] InnerDecode(Sequence pSequence, out SimpleCell[][] outputHiddenLayer, out Matrix<double> rawOutputLayer, out SimpleCell[][] forwardHidden, out SimpleCell[][] backwardHidden)
        {
            int numStates = pSequence.States.Length;
            SimpleCell[][] mForward = null;
            SimpleCell[][] mBackward = null;

            Parallel.Invoke(() =>
            {
                //Computing forward RNN      
                forwardRNN.netReset(false);
                mForward = new SimpleCell[numStates][];
                for (int curState = 0; curState < numStates; curState++)
                {
                    State state = pSequence.States[curState];
                    forwardRNN.setInputLayer(state, curState, numStates, null);
                    forwardRNN.computeHiddenLayer(state);      //compute probability distribution

                    mForward[curState] = forwardRNN.GetHiddenLayer();
                }
            },
             () =>
             {
                 //Computing backward RNN
                 backwardRNN.netReset(false);
                 mBackward = new SimpleCell[numStates][];
                 for (int curState = numStates - 1; curState >= 0; curState--)
                 {
                     State state = pSequence.States[curState];
                     backwardRNN.setInputLayer(state, curState, numStates, null, false);
                     backwardRNN.computeHiddenLayer(state);      //compute probability distribution

                     mBackward[curState] = backwardRNN.GetHiddenLayer();
                 }
             });

            //Merge forward and backward
            SimpleCell[][] mergedHiddenLayer = new SimpleCell[numStates][];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                mergedHiddenLayer[curState] = InitSimpleCell(L1);
                SimpleCell[] cells = mergedHiddenLayer[curState];
                SimpleCell[] forwardCells = mForward[curState];
                SimpleCell[] backwardCells = mBackward[curState];

                for (int i = 0; i < forwardRNN.L1; i++)
                {
                    cells[i].cellOutput = forwardCells[i].cellOutput + backwardCells[i].cellOutput;
                }
            });

            //Calculate output layer
            Matrix<double> tmp_rawOutputLayer = new Matrix<double>(numStates, L2);
            SimpleCell[][] seqOutput = new SimpleCell[numStates][];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                seqOutput[curState] = InitSimpleCell(L2);
                SimpleCell[] outputCells = seqOutput[curState];

                matrixXvectorADD(outputCells, mergedHiddenLayer[curState], Hidden2OutputWeight, 0, L2, 0, L1, 0);

                for (int i = 0; i < L2; i++)
                {
                    tmp_rawOutputLayer[curState][i] = outputCells[i].cellOutput;
                }

                //Activation on output layer
                SoftmaxLayer(outputCells);
            });

            outputHiddenLayer = mergedHiddenLayer;
            rawOutputLayer = tmp_rawOutputLayer;
            forwardHidden = mForward;
            backwardHidden = mBackward;

            return seqOutput;
        }

        public override int[] PredictSentenceCRF(Sequence pSequence, RunningMode runningMode)
        {
            //Reset the network
            int numStates = pSequence.States.Length;
            //Predict output
            SimpleCell[][] mergedHiddenLayer = null;
            Matrix<double> rawOutputLayer = null;
            SimpleCell[][] forwardHidden = null;
            SimpleCell[][] backwardHidden = null;
            SimpleCell[][] seqOutput = InnerDecode(pSequence, out mergedHiddenLayer, out rawOutputLayer, out forwardHidden, out backwardHidden);

            ForwardBackward(numStates, rawOutputLayer);

            if (runningMode != RunningMode.Test)
            {
                //Get the best result
                for (int i = 0; i < numStates; i++)
                {
                    logp += Math.Log10(CRFSeqOutput[i][pSequence.States[i].Label] + 0.0001);
                }
            }

            int[] predict = Viterbi(rawOutputLayer, numStates);

            if (runningMode == RunningMode.Train)
            {
                UpdateBigramTransition(pSequence);

                //Update hidden-output layer weights
                for (int curState = 0; curState < numStates; curState++)
                {
                    int label = pSequence.States[curState].Label;
                    SimpleCell[] layer = seqOutput[curState];
                    double[] CRFOutputLayer = CRFSeqOutput[curState];

                    //For standard RNN
                    for (int c = 0; c < L2; c++)
                    {
                        layer[c].er = -CRFOutputLayer[c];
                    }
                    layer[label].er = 1 - CRFOutputLayer[label];
                }

                LearnTwoRNN(pSequence, mergedHiddenLayer, seqOutput, forwardHidden, backwardHidden);
            }

            return predict;
        }

        public override Matrix<double> PredictSentence(Sequence pSequence, RunningMode runningMode)
        {
            //Reset the network
            int numStates = pSequence.States.Length;

            //Predict output
            SimpleCell[][] mergedHiddenLayer = null;
            Matrix<double> rawOutputLayer = null;
            SimpleCell[][] forwardHidden = null;
            SimpleCell[][] backwardHidden = null;
            SimpleCell[][] seqOutput = InnerDecode(pSequence, out mergedHiddenLayer, out rawOutputLayer, out forwardHidden, out backwardHidden);

            if (runningMode != RunningMode.Test)
            {
                //Merge forward and backward
                for (int curState = 0; curState < numStates; curState++)
                {
                    logp += Math.Log10(seqOutput[curState][pSequence.States[curState].Label].cellOutput + 0.0001);
                }
            }

            if (runningMode == RunningMode.Train)
            {
                //Update hidden-output layer weights
                for (int curState = 0; curState < numStates; curState++)
                {
                    int label = pSequence.States[curState].Label;
                    SimpleCell[] layer = seqOutput[curState];

                    //For standard RNN
                    for (int c = 0; c < L2; c++)
                    {
                        layer[c].er = -layer[c].cellOutput;
                    }
                    layer[label].er = 1.0 - layer[label].cellOutput;
                }

                LearnTwoRNN(pSequence, mergedHiddenLayer, seqOutput, forwardHidden, backwardHidden);
            }

            return rawOutputLayer;
        }

        private void LearnTwoRNN(Sequence pSequence, SimpleCell[][] mergedHiddenLayer, SimpleCell[][] seqOutput, SimpleCell[][] forwardHidden, SimpleCell[][] backwardHidden)
        {
            int numStates = pSequence.States.Length;

            Parallel.Invoke(() =>
            {
                forwardRNN.netReset(true);
                forwardRNN.Hidden2OutputWeight = Hidden2OutputWeight.CopyTo();
            },
            () =>
            {
                backwardRNN.netReset(true);
                backwardRNN.Hidden2OutputWeight = Hidden2OutputWeight.CopyTo();
            });

            Parallel.Invoke(() =>
            {
                for (int curState = 0; curState < numStates; curState++)
                {
                    SimpleCell[] outputCells = seqOutput[curState];
                    SimpleCell[] mergedHiddenCells = mergedHiddenLayer[curState];
                    for (int i = 0; i < Hidden2OutputWeight.Height; i++)
                    {
                        //update weights for hidden to output layer

                        double er = outputCells[i].er;
                        double[] vector_i = Hidden2OutputWeight[i];
                        for (int k = 0; k < Hidden2OutputWeight.Width; k++)
                        {
                            vector_i[k] += LearningRate * mergedHiddenCells[k].cellOutput * er;
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
                    State state = pSequence.States[curState];

                    forwardRNN.setInputLayer(state, curState, numStates, null);
                    forwardRNN.SetHiddenLayer(forwardHidden[curState]);
                    //Copy output result to forward net work's output
                    forwardRNN.OutputLayer = seqOutput[curState];

                    forwardRNN.ComputeHiddenLayerErr();

                    forwardRNN.learnNet(state);
                    forwardRNN.LearnBackTime(state, numStates, curState);
                }
            },
            () =>
            {

                for (int curState = 0; curState < numStates; curState++)
                {
                    int curState2 = numStates - 1 - curState;

                    // error propogation
                    State state2 = pSequence.States[curState2];

                    backwardRNN.setInputLayer(state2, curState2, numStates, null, false);
                    backwardRNN.SetHiddenLayer(backwardHidden[curState2]);
                    //Copy output result to forward net work's output
                    backwardRNN.OutputLayer = seqOutput[curState2];

                    backwardRNN.ComputeHiddenLayerErr();

                    backwardRNN.learnNet(state2);
                    backwardRNN.LearnBackTime(state2, numStates, curState2);
                }
            });
        }

        public override void ComputeHiddenLayerErr()
        {
            throw new NotImplementedException("ComputeHiddenLayerErr is not implemented in BiRNN");
        }

        public override void LearnOutputWeight()
        {
            throw new NotImplementedException("LearnOutputWeight is not implemented in BiRNN");
        }

        public override void LearnBackTime(State state, int numStates, int curState)
        {
            throw new NotImplementedException("LearnBackTime is not implemented in BiRNN");
        }

        public override void learnNet(State state)
        {
            throw new NotImplementedException("learnNet is not implemented in BiRNN");
        }

        public override void computeHiddenLayer(State state, bool isTrain = true)
        {
            throw new NotImplementedException("computeHiddenLayer is not implemented in BiRNN");
        }

        public override void computeOutput(double[] doutput)
        {
            throw new NotImplementedException("computeOutput is not implemented in BiRNN");
        }

        public override void netReset(bool updateNet = false)
        {
            throw new NotImplementedException("netReset is not implemented in BiRNN");
        }

        public override void saveNetBin(string filename)
        {
            //Save bi-directional model
            forwardRNN.Hidden2OutputWeight = Hidden2OutputWeight;
            backwardRNN.Hidden2OutputWeight = Hidden2OutputWeight;

            forwardRNN.CRFTagTransWeights = CRFTagTransWeights;
            backwardRNN.CRFTagTransWeights = CRFTagTransWeights;

            forwardRNN.saveNetBin(filename + ".forward");
            backwardRNN.saveNetBin(filename + ".backward");

            //Save meta data
            using (StreamWriter sw = new StreamWriter(filename))
            {
                BinaryWriter fo = new BinaryWriter(sw.BaseStream);
                fo.Write((int)ModelType);
                fo.Write((int)ModelDirection);

                // Signiture , 0 is for RNN or 1 is for RNN-CRF
                int iflag = 0;
                if (IsCRFTraining == true)
                {
                    iflag = 1;
                }
                fo.Write(iflag);

                fo.Write(L0);
                fo.Write(L1);
                fo.Write(L2);
                fo.Write(DenseFeatureSize);
            }
        }

        public override void loadNetBin(string filename)
        {
            Logger.WriteLine(Logger.Level.info, "Loading bi-directional model: {0}", filename);

            forwardRNN.loadNetBin(filename + ".forward");
            backwardRNN.loadNetBin(filename + ".backward");

            Hidden2OutputWeight = forwardRNN.Hidden2OutputWeight;
            CRFTagTransWeights = forwardRNN.CRFTagTransWeights;

            using (StreamReader sr = new StreamReader(filename))
            {
                BinaryReader br = new BinaryReader(sr.BaseStream);

                ModelType = (MODELTYPE)br.ReadInt32();
                ModelDirection = (MODELDIRECTION)br.ReadInt32();

                int iflag = br.ReadInt32();
                if (iflag == 1)
                {
                    IsCRFTraining = true;
                }
                else
                {
                    IsCRFTraining = false;
                }

                //Load basic parameters
                L0 = br.ReadInt32();
                L1 = br.ReadInt32();
                L2 = br.ReadInt32();
                DenseFeatureSize = br.ReadInt32();
            }
        }
    }
}
