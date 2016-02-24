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

        public override void CleanStatus()
        {
            forwardRNN.CleanStatus();
            backwardRNN.CleanStatus();

            Hidden2OutputWeightLearningRate = new Matrix<float>(L2, L1);
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

        public override double GradientCutoff
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

        public override bool bVQ
        {
            get
            {
                return forwardRNN.bVQ;
            }

            set
            {
                forwardRNN.bVQ = value;
                backwardRNN.bVQ = value;
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

        public override SimpleLayer GetHiddenLayer()
        {
            throw new NotImplementedException("GetHiddenLayer is not implemented in BiRNN");
        }

        public override void InitMem()
        {
            forwardRNN.InitMem();
            backwardRNN.InitMem();

            //Create and intialise the weights from hidden to output layer, these are just normal weights
            Hidden2OutputWeight = new Matrix<double>(L2, L1);

            for (int i = 0; i < Hidden2OutputWeight.Height; i++)
            {
                for (int j = 0; j < Hidden2OutputWeight.Width; j++)
                {
                    Hidden2OutputWeight[i][j] = RandInitWeight();
                }
            }

            Hidden2OutputWeightLearningRate = new Matrix<float>(L2, L1);
        }

        public SimpleLayer[] InnerDecode(Sequence pSequence, out SimpleLayer[] outputHiddenLayer, out Matrix<double> rawOutputLayer)
        {
            int numStates = pSequence.States.Length;
            SimpleLayer[] mForward = null;
            SimpleLayer[] mBackward = null;

            Parallel.Invoke(() =>
            {
                //Computing forward RNN      
                forwardRNN.netReset(false);
                mForward = new SimpleLayer[numStates];
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
                 mBackward = new SimpleLayer[numStates];
                 for (int curState = numStates - 1; curState >= 0; curState--)
                 {
                     State state = pSequence.States[curState];
                     backwardRNN.setInputLayer(state, curState, numStates, null, false);
                     backwardRNN.computeHiddenLayer(state);      //compute probability distribution

                     mBackward[curState] = backwardRNN.GetHiddenLayer();
                 }
             });

            //Merge forward and backward
            SimpleLayer[] mergedHiddenLayer = new SimpleLayer[numStates];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                mergedHiddenLayer[curState] = new SimpleLayer(L1);
                SimpleLayer cells = mergedHiddenLayer[curState];
                SimpleLayer forwardCells = mForward[curState];
                SimpleLayer backwardCells = mBackward[curState];

                for (int i = 0; i < forwardRNN.L1; i++)
                {
                    cells.cellOutput[i] = (forwardCells.cellOutput[i] + backwardCells.cellOutput[i]) / 2.0;
                }
            });

            //Calculate output layer
            Matrix<double> tmp_rawOutputLayer = new Matrix<double>(numStates, L2);
            SimpleLayer[] seqOutput = new SimpleLayer[numStates];
            Parallel.For(0, numStates, parallelOption, curState =>
            {
                seqOutput[curState] = new SimpleLayer(L2);
                SimpleLayer outputCells = seqOutput[curState];

                matrixXvectorADD(outputCells, mergedHiddenLayer[curState], Hidden2OutputWeight, L2, L1, 0);

                double[] tmp_vector = tmp_rawOutputLayer[curState];
                outputCells.cellOutput.CopyTo(tmp_vector, 0);

                //Activation on output layer
                SoftmaxLayer(outputCells);
            });

            outputHiddenLayer = mergedHiddenLayer;
            rawOutputLayer = tmp_rawOutputLayer;

            return seqOutput;
        }

        public override int[] PredictSentenceCRF(Sequence pSequence, RunningMode runningMode)
        {
            //Reset the network
            int numStates = pSequence.States.Length;
            //Predict output
            SimpleLayer[] mergedHiddenLayer = null;
            Matrix<double> rawOutputLayer = null;
            SimpleLayer[] seqOutput = InnerDecode(pSequence, out mergedHiddenLayer, out rawOutputLayer);

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
                    SimpleLayer layer = seqOutput[curState];
                    double[] CRFOutputLayer = CRFSeqOutput[curState];

                    //For standard RNN
                    for (int c = 0; c < L2; c++)
                    {
                        layer.er[c] = -CRFOutputLayer[c];
                    }
                    layer.er[label] = 1 - CRFOutputLayer[label];
                }

                LearnTwoRNN(pSequence, mergedHiddenLayer, seqOutput);
            }

            return predict;
        }

        public override Matrix<double> PredictSentence(Sequence pSequence, RunningMode runningMode)
        {
            //Reset the network
            int numStates = pSequence.States.Length;

            //Predict output
            SimpleLayer[] mergedHiddenLayer = null;
            Matrix<double> rawOutputLayer = null;
            SimpleLayer[] seqOutput = InnerDecode(pSequence, out mergedHiddenLayer, out rawOutputLayer);

            if (runningMode != RunningMode.Test)
            {
                //Merge forward and backward
                for (int curState = 0; curState < numStates; curState++)
                {
                    logp += Math.Log10(seqOutput[curState].cellOutput[pSequence.States[curState].Label] + 0.0001);
                }
            }

            if (runningMode == RunningMode.Train)
            {
                //Update hidden-output layer weights
                for (int curState = 0; curState < numStates; curState++)
                {
                    int label = pSequence.States[curState].Label;
                    SimpleLayer layer = seqOutput[curState];

                    //For standard RNN
                    for (int c = 0; c < L2; c++)
                    {
                        layer.er[c] = -layer.cellOutput[c];
                    }
                    layer.er[label] = 1.0 - layer.cellOutput[label];
                }

                LearnTwoRNN(pSequence, mergedHiddenLayer, seqOutput);
            }

            return rawOutputLayer;
        }

        private void LearnTwoRNN(Sequence pSequence, SimpleLayer[] mergedHiddenLayer, SimpleLayer[] seqOutput)
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
                    SimpleLayer outputCells = seqOutput[curState];
                    SimpleLayer mergedHiddenCells = mergedHiddenLayer[curState];
                    for (int i = 0; i < Hidden2OutputWeight.Height; i++)
                    {
                        //update weights for hidden to output layer
                        double er = outputCells.er[i];
                        double[] vector_i = Hidden2OutputWeight[i];
                        for (int k = 0; k < Hidden2OutputWeight.Width; k++)
                        {
                            double delta = NormalizeGradient(mergedHiddenCells.cellOutput[k] * er);
                            double newLearningRate = UpdateLearningRate(Hidden2OutputWeightLearningRate, i, k, delta);

                            vector_i[k] += newLearningRate * delta;
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

                    forwardRNN.computeHiddenLayer(state, true);

                    //Copy output result to forward net work's output
                    forwardRNN.OutputLayer = seqOutput[curState];
                    forwardRNN.ComputeHiddenLayerErr();

                    //Update net weights
                    forwardRNN.LearnNet(state, numStates, curState);
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

                    backwardRNN.computeHiddenLayer(state2, true);

                    //Copy output result to forward net work's output
                    backwardRNN.OutputLayer = seqOutput[curState2];
                    backwardRNN.ComputeHiddenLayerErr();

                    //Update net weights
                    backwardRNN.LearnNet(state2, numStates, curState);
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

        public override void LearnNet(State state, int numStates, int curState)
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

        public override void SaveModel(string filename)
        {
            //Save bi-directional model
            forwardRNN.Hidden2OutputWeight = Hidden2OutputWeight;
            backwardRNN.Hidden2OutputWeight = Hidden2OutputWeight;

            forwardRNN.CRFTagTransWeights = CRFTagTransWeights;
            backwardRNN.CRFTagTransWeights = CRFTagTransWeights;

            forwardRNN.SaveModel(filename + ".forward");
            backwardRNN.SaveModel(filename + ".backward");

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

        public override void LoadModel(string filename)
        {
            Logger.WriteLine(Logger.Level.info, "Loading bi-directional model: {0}", filename);

            forwardRNN.LoadModel(filename + ".forward");
            backwardRNN.LoadModel(filename + ".backward");

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
