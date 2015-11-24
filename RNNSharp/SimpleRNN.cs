using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace RNNSharp
{
    public class SimpleRNN : RNN
    {
        protected int bptt;
        protected int bptt_block;
        protected neuron[] bptt_hidden;
        protected neuron[] bptt_fea;
        protected SparseVector[] bptt_inputs = new SparseVector[MAX_RNN_HIST];    // TODO: add const constraint

        protected Matrix<double> mat_bptt_syn0_w = new Matrix<double>();
        protected Matrix<double> mat_bptt_syn0_ph = new Matrix<double>();

        protected Matrix<double> mat_bptt_synf = new Matrix<double>();
        protected Matrix<double> mat_hiddenBpttWeight = new Matrix<double>();

        protected neuron[] neuLastHidden;		//neurons in input layer
        protected neuron[] neuHidden;		//neurons in hidden layer
        protected Matrix<double> mat_input2hidden = new Matrix<double>();
        protected Matrix<double> mat_feature2hidden = new Matrix<double>();

        public SimpleRNN()
        {
            m_modeltype = MODELTYPE.SIMPLE;
            gradient_cutoff = 15;
            beta = 0.0000001;
            llogp = -100000000;
            iter = 0;

            L1 = 30;
            bptt = 0;
            bptt_block = 10;
            bptt_hidden = null;
            bptt_fea = null;


            fea_size = 0;

            neuLastHidden = null;
            neuFeatures = null;
            neuHidden = null;
            neuOutput = null;
        }

        public void setBPTT(int newval) { bptt = newval; }
        public void setBPTTBlock(int newval) { bptt_block = newval; }


        public override void initWeights()
        {
            int b, a;
            for (b = 0; b < L1; b++)
            {
                for (a = 0; a < L0; a++)
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

            for (b = 0; b < mat_hidden2output.GetHeight(); b++)
            {
                for (a = 0; a < L1; a++)
                {
                    mat_hidden2output[b][a] = random(-randrng, randrng) + random(-randrng, randrng) + random(-randrng, randrng);
                }
            }

            for (b = 0; b < L1; b++)
            {
                for (a = 0; a < L1; a++)
                {
                    mat_hiddenBpttWeight[b][a] = random(-randrng, randrng) + random(-randrng, randrng) + random(-randrng, randrng);
                }
            }
        }

        public override void GetHiddenLayer(Matrix<double> m, int curStatus)
        {
            for (int i = 0; i < L1; i++)
            {
                m[curStatus][i] = neuHidden[i].cellOutput;
            }
        }

        public void computeHiddenActivity()
        {
            for (int a = 0; a < L1; a++)
            {
                if (neuHidden[a].cellOutput > 50) neuHidden[a].cellOutput = 50;  //for numerical stability
                if (neuHidden[a].cellOutput < -50) neuHidden[a].cellOutput = -50;  //for numerical stability
                neuHidden[a].cellOutput = 1.0 / (1.0 + Math.Exp(-neuHidden[a].cellOutput));
            }
        }

        // forward process. output layer consists of tag value
        public override void computeNet(State state, double[] doutput)
        {
            //keep last hidden layer and erase activations
            neuLastHidden = new neuron[L1];
            for (int a = 0; a < L1; a++)
            {
                neuLastHidden[a].cellOutput = neuHidden[a].cellOutput;
            }

            //hidden(t-1) -> hidden(t)
            neuHidden = new neuron[L1];
            matrixXvectorADD(neuHidden, neuLastHidden, mat_hiddenBpttWeight, 0, L1, 0, L1, 0);

            //inputs(t) -> hidden(t)
            //Get sparse feature and apply it into hidden layer
            var sparse = state.GetSparseData();
            int n = sparse.GetNumberOfEntries();

            for (int i = 0; i < n; i++)
            {
                var entry = sparse.GetEntry(i);
                for (int b = 0; b < L1; b++)
                {
                    neuHidden[b].cellOutput += entry.Value * mat_input2hidden[b][entry.Key];
                }
            }

            //fea(t) -> hidden(t) 
            if (fea_size > 0)
            {
                matrixXvectorADD(neuHidden, neuFeatures, mat_feature2hidden, 0, L1, 0, fea_size);
            }

            //activate 1      --sigmoid
            computeHiddenActivity();

            //initialize output nodes
            for (int c = 0; c < L2; c++)
            {
                neuOutput[c].cellOutput = 0;
            }

            matrixXvectorADD(neuOutput, neuHidden, mat_hidden2output, 0, L2, 0, L1, 0);
            if (doutput != null)
            {
                for (int i = 0; i < L2; i++)
                {
                    doutput[i] = neuOutput[i].cellOutput;
                }
            }

            //activation 2   --softmax on words
            SoftmaxLayer(neuOutput);
        }

     

        public override void learnNet(State state, int timeat, bool biRNN = false)
        {
            if (biRNN == false)
            {
                CalculateOutputLayerError(state, timeat);
            }

            for (int a = 0; a < L1; a++)
            {
                neuHidden[a].er = 0;
            }
            matrixXvectorADD(neuHidden, neuOutput, mat_hidden2output, 0, L1, 0, L2, 1);	//error output->hidden for words from specific class    	

            if (biRNN == false)
            {
                for (int a = 0; a < L1; a++)
                {
                    for (int c = 0; c < L2; c++)
                    {
                        double dg = neuOutput[c].er * neuHidden[a].cellOutput;

                        if ((counter % 10) == 0)	//regularization is done every 10. step
                        {
                            mat_hidden2output[c][a] += alpha * (dg - mat_hidden2output[c][a] * beta);
                        }
                        else
                        {
                            mat_hidden2output[c][a] += alpha * dg;
                        }
                    }
                }
            }
        }

        void learnBptt(State state)
        {
            for (int step = 0; step < bptt + bptt_block - 2; step++)
            {
                if (null == bptt_inputs[step])
                    break;

                // compute hidden layter gradient
                for (int a = 0; a < L1; a++)
                {
                    neuHidden[a].er *= neuHidden[a].cellOutput * (1 - neuHidden[a].cellOutput);
                }

                //weight update fea->0
                if (fea_size > 0)
                {
                    Parallel.For(0, L1, parallelOption, b =>
                    {
                        for (int a = 0; a < fea_size; a++)
                        {
                            mat_bptt_synf[b][a] += neuHidden[b].er * bptt_fea[a + step * fea_size].cellOutput;
                        }
                    });
                }

                //weight update hidden->input
                var sparse = bptt_inputs[step];
                Parallel.For(0, L1, parallelOption, b =>
                {
                    for (int i = 0; i < sparse.GetNumberOfEntries(); i++)
                    {
                        mat_bptt_syn0_w[b][sparse.GetEntry(i).Key] += neuHidden[b].er * sparse.GetEntry(i).Value;

                    }
                });

                for (int a = 0; a < L1; a++)
                {
                    neuLastHidden[a].er = 0;
                }

                matrixXvectorADD(neuLastHidden, neuHidden, mat_hiddenBpttWeight, 0, L1, 0, L1, 1);		//propagates errors hidden->input to the recurrent part

                Parallel.For(0, L1, parallelOption, b =>
                {
                    for (int a = 0; a < L1; a++)
                    {
                        mat_bptt_syn0_ph[b][a] += neuHidden[b].er * neuLastHidden[a].cellOutput;
                    }
                });

                for (int a = 0; a < L1; a++)
                {
                    //propagate error from time T-n to T-n-1
                    neuHidden[a].er = neuLastHidden[a].er + bptt_hidden[(step + 1) * L1 + a].er;
                }

                if (step < bptt + bptt_block - 3)
                {
                    for (int a = 0; a < L1; a++)
                    {
                        neuHidden[a].cellOutput = bptt_hidden[(step + 1) * L1 + a].cellOutput;
                        neuLastHidden[a].cellOutput = bptt_hidden[(step + 2) * L1 + a].cellOutput;
                    }
                }
            }

            for (int a = 0; a < (bptt + bptt_block) * L1; a++)
            {
                bptt_hidden[a].er = 0;
            }

            for (int b = 0; b < L1; b++)
            {
                neuHidden[b].cellOutput = bptt_hidden[b].cellOutput;		//restore hidden layer after bptt
            }


            UpdateWeights(mat_hiddenBpttWeight, mat_bptt_syn0_ph);

            if (fea_size > 0)
            {
                UpdateWeights(mat_feature2hidden, mat_bptt_synf);
            }

            Parallel.For(0, L1, parallelOption, b =>
            {
                for (int step = 0; step < bptt + bptt_block - 2; step++)
                {
                    if (null == bptt_inputs[step])
                        break;

                    var sparse = bptt_inputs[step];
                    for (int i = 0; i < sparse.GetNumberOfEntries(); i++)
                    {
                        int pos = sparse.GetEntry(i).Key;
                        if ((counter % 10) == 0)
                        {
                            mat_input2hidden[b][pos] += alpha * (mat_bptt_syn0_w[b][pos] - mat_input2hidden[b][pos] * beta);
                        }
                        else
                        {
                            mat_input2hidden[b][pos] += alpha * mat_bptt_syn0_w[b][pos];
                        }

                        mat_bptt_syn0_w[b][pos] = 0;
                    }
                }
            });
        }


        public void resetBpttMem()
        {
            if (null != bptt_hidden)
            {
                bptt_hidden = null;
            }
            if (null != bptt_fea)
            {
                bptt_fea = null;
            }

            for (int a = 0; a < MAX_RNN_HIST; a++)
            {
                bptt_inputs[a] = null;
            }

            bptt_hidden = new neuron[(bptt + bptt_block + 1) * L1];
            for (int a = 0; a < (bptt + bptt_block) * L1; a++)
            {
                bptt_hidden[a].cellOutput = 0;
                bptt_hidden[a].er = 0;
            }

            bptt_fea = new neuron[(bptt + bptt_block + 2) * fea_size];
            for (int a = 0; a < (bptt + bptt_block) * fea_size; a++)
                bptt_fea[a].cellOutput = 0;

            mat_bptt_syn0_w = new Matrix<double>(L1, L0);
            mat_bptt_syn0_ph = new Matrix<double>(L1, L1);
            mat_bptt_synf = new Matrix<double>(L1, fea_size);
        }

        public override void initMem()
        {
            CreateCells();

            mat_hidden2output = new Matrix<double>(L2, L1);

            for (int i = 0; i < MAX_RNN_HIST; i++)
            {
                m_Diff[i] = new double[L2];
            }

            m_tagBigramTransition = new Matrix<double>(L2, L2);
            m_DeltaBigramLM = new Matrix<double>(L2, L2);


            mat_input2hidden = new Matrix<double>(L1, L0);
            mat_feature2hidden = new Matrix<double>(L1, fea_size);

            mat_hiddenBpttWeight = new Matrix<double>(L1, L1);


            Console.WriteLine("[TRACE] Initializing weights, random value is {0}", random(-1.0, 1.0));// yy debug
            initWeights();

            //Initialize BPTT
            if (bptt > 0)
            {
                resetBpttMem();
            }
        }

        public override void netReset(bool updateNet = false)   //cleans hidden layer activation + bptt history
        {
            for (int a = 0; a < L1; a++)
                neuHidden[a].cellOutput = 0.1;

            if (bptt > 0)
            {
                for (int a = 2; a < bptt + bptt_block; a++)
                {
                    for (int b = 0; b < L1; b++)
                    {
                        bptt_hidden[a * L1 + b].cellOutput = 0;
                        bptt_hidden[a * L1 + b].er = 0;
                    }
                }

                for (int a = 0; a < bptt + bptt_block; a++)
                {
                    for (int b = 0; b < fea_size; b++)
                        bptt_fea[a * fea_size + b].cellOutput = 0;
                }
            }
        }


        public override void LearnBackTime(State state, int numStates, int curState)
        {
            if (bptt > 0)
            {
                //shift memory needed for bptt to next time step
                for (int a = bptt + bptt_block - 1; a > 0; a--)
                    bptt_inputs[a] = bptt_inputs[a - 1];
                bptt_inputs[0] = state.GetSparseData();

                for (int a = bptt + bptt_block - 1; a > 0; a--)
                {
                    for (int b = 0; b < L1; b++)
                    {
                        bptt_hidden[a * L1 + b] = bptt_hidden[(a - 1) * L1 + b];
                    }
                }

                for (int a = bptt + bptt_block - 1; a > 0; a--)
                {
                    for (int b = 0; b < fea_size; b++)
                    {
                        bptt_fea[a * fea_size + b].cellOutput = bptt_fea[(a - 1) * fea_size + b].cellOutput;
                    }
                }
            }

            //Save hidden and feature layer nodes values for bptt
            for (int b = 0; b < L1; b++)
            {
                bptt_hidden[b] = neuHidden[b];
            }
            for (int b = 0; b < fea_size; b++)
            {
                bptt_fea[b].cellOutput = neuFeatures[b];
            }

            // time to learn bptt
            if (((counter % bptt_block) == 0) || (curState == numStates - 1))
            {
                learnBptt(state);
            }
        }


        public override void netFlush()   //cleans all activations and error vectors
        {
            int a;

            for (a = 0; a < L1; a++)
            {
                neuHidden[a].cellOutput = 0;
                neuHidden[a].er = 0;
            }

            for (a = 0; a < L2; a++)
            {
                neuOutput[a].cellOutput = 0;
                neuOutput[a].er = 0;
            }
        }

        public override void loadNetBin(string filename)
        {
            StreamReader sr = new StreamReader(filename);
            BinaryReader br = new BinaryReader(sr.BaseStream);

            m_modeltype = (MODELTYPE)br.ReadInt32();
            if (m_modeltype != MODELTYPE.SIMPLE)
            {
                throw new Exception("Invalidated model format: must be simple RNN");
            }

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

            //Create cells of each layer
            CreateCells();

            //Load weight matrix between each two layer pairs
            mat_input2hidden = loadMatrixBin(br);
            mat_hiddenBpttWeight = loadMatrixBin(br);

            mat_feature2hidden = loadMatrixBin(br);
            mat_hidden2output = loadMatrixBin(br);


            if (iflag == 1)
            {
                m_tagBigramTransition = loadMatrixBin(br);

                for (int i = 0; i < MAX_RNN_HIST; i++)
                {
                    m_Diff[i] = new double[L2];
                }
                m_DeltaBigramLM = new Matrix<double>(L2, L2);
            }

            sr.Close();
        }

        private void CreateCells()
        {
            neuFeatures = new double[fea_size];

            neuOutput = new neuron[L2];
            for (int a = 0; a < L2; a++)
            {
                neuOutput[a].cellOutput = 0;
                neuOutput[a].er = 0;
            }

            neuHidden = new neuron[L1];
            for (int a = 0; a < L1; a++)
            {
                neuHidden[a].cellOutput = 0;
                neuHidden[a].er = 0;
            }
        }

        // save model as binary format
        public override void saveNetBin(string filename) 
        {
            StreamWriter sw = new StreamWriter(filename);
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


            //weight input->hidden
            saveMatrixBin(mat_input2hidden, fo);
            saveMatrixBin(mat_hiddenBpttWeight, fo);

            //weight fea->hidden
            saveMatrixBin(mat_feature2hidden, fo);

            //weight hidden->output
            saveMatrixBin(mat_hidden2output, fo);

            if (iflag == 1)
            {
                // Save Bigram
                saveMatrixBin(m_tagBigramTransition, fo);
            }

            fo.Close();
        }
    }
}
