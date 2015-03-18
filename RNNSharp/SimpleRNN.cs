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

        protected Matrix mat_bptt_syn0_w = new Matrix();
        protected Matrix mat_bptt_syn0_ph = new Matrix();

        protected Matrix mat_bptt_synf = new Matrix();
        protected Matrix mat_hiddenBpttWeight = new Matrix();
        protected Matrix mat_hiddenBpttWeight_alpha = new Matrix();

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

            neuInput = null;
            neuFeatures = null;
            neuHidden = null;
            neuOutput = null;
        }

        public void setBPTT(int newval) { bptt = newval; }
        public void setBPTTBlock(int newval) { bptt_block = newval; }


        // forward process. output layer consists of tag value
        public override void computeNet(State state, double[] doutput)
        {
            //erase activations
            for (int a = 0; a < L1; a++)
                neuHidden[a].ac = 0;

            //hidden(t-1) -> hidden(t)
            matrixXvectorADD(neuHidden, neuInput, mat_hiddenBpttWeight, 0, L1, L0 - L1, L0, 0);

            //inputs(t) -> hidden(t)
            //Get sparse feature and apply it into hidden layer
            var sparse = state.GetSparseData();
            int n = sparse.GetNumberOfEntries();

            for (int i = 0; i < n; i++)
            {
                var entry = sparse.GetEntry(i);
                for (int b = 0; b < L1; b++)
                {
                    neuHidden[b].ac += entry.Value * mat_input2hidden[b][entry.Key];
                }
            }

            //fea(t) -> hidden(t) 
            if (fea_size > 0)
            {
                matrixXvectorADD(neuHidden, neuFeatures, mat_feature2hidden, 0, L1, 0, fea_size, 0);
            }

            //activate 1      --sigmoid
            computeHiddenActivity();

            //initialize output nodes
            for (int c = 0; c < L2; c++)
            {
                neuOutput[c].ac = 0;
            }

            matrixXvectorADD(neuOutput, neuHidden, mat_hidden2output, 0, L2, 0, L1, 0);

            //fea to out word
            if (fea_size > 0)
            {
                matrixXvectorADD(neuOutput, neuFeatures, mat_feature2output, 0, L2, 0, fea_size, 0);
            }


            if (doutput != null)
            {
                for (int i = 0; i < L2; i++)
                {
                    doutput[i] = neuOutput[i].ac;
                }
            }

            //activation 2   --softmax on words
            double sum = 0;   //sum is used for normalization: it's better to have larger precision as many numbers are summed together here
            for (int c = 0; c < L2; c++)
            {
                if (neuOutput[c].ac > 50) neuOutput[c].ac = 50;  //for numerical stability
                if (neuOutput[c].ac < -50) neuOutput[c].ac = -50;  //for numerical stability
                double val = Math.Exp(neuOutput[c].ac);
                sum += val;
                neuOutput[c].ac = val;
            }

            for (int c = 0; c < L2; c++)
            {
                neuOutput[c].ac /= sum;
            }
        }

     

        public override void learnNet(State state, int timeat)
        {
            if (m_bCRFTraining == true)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                for (int c = 0; c < L2; c++)
                {
                    neuOutput[c].er = -m_Diff[timeat][c];
                }
                neuOutput[state.GetLabel()].er = 1 - m_Diff[timeat][state.GetLabel()];
            }
            else
            {
                //For standard RNN
                for (int c = 0; c < L2; c++)
                {
                    neuOutput[c].er = -neuOutput[c].ac;
                }
                neuOutput[state.GetLabel()].er = 1 - neuOutput[state.GetLabel()].ac;
            }

            for (int a = 0; a < L1; a++)
            {
                neuHidden[a].er = 0;
            }
            matrixXvectorADD(neuHidden, neuOutput, mat_hidden2output, 0, L2, 0, L1, 1);	//error output->hidden for words from specific class    	

            Parallel.For(0, L2, parallelOption, c =>
            {
                for (int a = 0; a < L1; a++)
                {
                    double dg = neuOutput[c].er * neuHidden[a].ac;
                    double dlr = calcAlpha(mat_hidden2output_alpha, c, a, dg);

                    if ((counter % 10) == 0)	//regularization is done every 10. step
                    {
                        mat_hidden2output[c][a] += dlr * (dg - mat_hidden2output[c][a] * beta);
                    }
                    else
                    {
                        mat_hidden2output[c][a] += dlr * dg;
                    }
                }
            });

            //direct fea size weights update
            if (fea_size > 0)
            {
                Parallel.For(0, L2, parallelOption, c =>
                {
                    for (int a = 0; a < fea_size; a++)
                    {
                        double dg = neuOutput[c].er * neuFeatures[a].ac;
                        double dlr = calcAlpha(mat_feature2output_alpha, c, a, dg);

                        if ((counter % 10) == 0)	//regularization is done every 10. step
                        {
                            mat_feature2output[c][a] += dlr * (dg - mat_feature2output[c][a] * beta);
                        }
                        else
                        {
                            mat_feature2output[c][a] += dlr * dg;
                        }
                    }
                    // probably also need to do regularization
                });
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
                    neuHidden[a].er *= neuHidden[a].ac * (1 - neuHidden[a].ac);
                }

                //weight update fea->0
                if (fea_size > 0)
                {
                    Parallel.For(0, L1, parallelOption, b =>
                    {
                        for (int a = 0; a < fea_size; a++)
                        {
                            mat_bptt_synf[b][a] += neuHidden[b].er * bptt_fea[a + step * fea_size].ac;
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

                for (int a = L0 - L1; a < L0; a++)
                {
                    neuInput[a].er = 0;
                }

                matrixXvectorADD(neuInput, neuHidden, mat_hiddenBpttWeight, 0, L1, L0 - L1, L0, 1);		//propagates errors hidden->input to the recurrent part

                Parallel.For(0, L1, parallelOption, b =>
                {
                    for (int a = 0; a < L1; a++)
                    {
                        mat_bptt_syn0_ph[b][a] += neuHidden[b].er * neuInput[L0 - L1 + a].ac;
                    }
                });

                for (int a = 0; a < L1; a++)
                {
                    //propagate error from time T-n to T-n-1
                    neuHidden[a].er = neuInput[a + L0 - L1].er + bptt_hidden[(step + 1) * L1 + a].er;
                }

                if (step < bptt + bptt_block - 3)
                {
                    for (int a = 0; a < L1; a++)
                    {
                        neuHidden[a].ac = bptt_hidden[(step + 1) * L1 + a].ac;
                        neuInput[a + L0 - L1].ac = bptt_hidden[(step + 2) * L1 + a].ac;
                    }
                }
            }

            for (int a = 0; a < (bptt + bptt_block) * L1; a++)
            {
                bptt_hidden[a].er = 0;
            }

            for (int b = 0; b < L1; b++)
            {
                neuHidden[b].ac = bptt_hidden[b].ac;		//restore hidden layer after bptt
            }


            UpdateWeights(mat_hiddenBpttWeight, mat_bptt_syn0_ph, mat_hiddenBpttWeight_alpha);

            if (fea_size > 0)
            {
                UpdateWeights(mat_feature2hidden, mat_bptt_synf, mat_feature2hidden_alpha);
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

                        double dlr = calcAlpha(mat_input2hidden_alpha, b, pos, mat_bptt_syn0_w[b][pos]);

                        if ((counter % 10) == 0)
                        {
                            mat_input2hidden[b][pos] += dlr * (mat_bptt_syn0_w[b][pos] - mat_input2hidden[b][pos] * beta);
                        }
                        else
                        {
                            mat_input2hidden[b][pos] += dlr * mat_bptt_syn0_w[b][pos];
                        }

                        mat_bptt_syn0_w[b][pos] = 0;
                    }
                }
            });
        }

        public override void copyHiddenLayerToInput()
        {
            for (int a = 0; a < L1; a++)
                neuInput[a + L0 - L1].ac = neuHidden[a].ac;
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
                bptt_hidden[a].ac = 0;
                bptt_hidden[a].er = 0;
            }

            bptt_fea = new neuron[(bptt + bptt_block + 2) * fea_size];
            for (int a = 0; a < (bptt + bptt_block) * fea_size; a++)
                bptt_fea[a].ac = 0;

            mat_bptt_syn0_w = new Matrix(L1, L0 - L1);
            mat_bptt_syn0_ph = new Matrix(L1, L1);
            mat_bptt_synf = new Matrix(L1, fea_size);
        }

        public override void initMem()
        {
            base.initMem();

            //Initialize BPTT

            mat_hiddenBpttWeight = new Matrix(L1, L1);
            int b, a;
            for (b = 0; b < L1; b++)
            {
                for (a = 0; a < L1; a++)
                {
                    mat_hiddenBpttWeight[b][a] = random(-randrng, randrng) + random(-randrng, randrng) + random(-randrng, randrng);
                }
            }

            mat_hiddenBpttWeight_alpha = new Matrix(L1, L1);

            if (bptt > 0)
            {
                resetBpttMem();
            }
        }

        public override void netReset()   //cleans hidden layer activation + bptt history
        {
            for (int a = 0; a < L1; a++)
                neuHidden[a].ac = 1.0;

            copyHiddenLayerToInput();

            if (bptt > 0)
            {
                for (int a = 2; a < bptt + bptt_block; a++)
                {
                    for (int b = 0; b < L1; b++)
                    {
                        bptt_hidden[a * L1 + b].ac = 0;
                        bptt_hidden[a * L1 + b].er = 0;
                    }
                }

                for (int a = 0; a < bptt + bptt_block; a++)
                {
                    for (int b = 0; b < fea_size; b++)
                        bptt_fea[a * fea_size + b].ac = 0;
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
                        bptt_fea[a * fea_size + b].ac = bptt_fea[(a - 1) * fea_size + b].ac;
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
                bptt_fea[b].ac = neuFeatures[b].ac;
            }

            // time to learn bptt
            if (((counter % bptt_block) == 0) || (curState == numStates - 1))
            {
                learnBptt(state);
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
            mat_feature2output = loadMatrixBin(br);
            mat_hidden2output = loadMatrixBin(br);


            if (iflag == 1)
            {
                m_tagBigramTransition = loadMatrixBin(br);

                for (int i = 0; i < MAX_RNN_HIST; i++)
                {
                    m_Diff[i] = new double[L2];
                }
                m_DeltaBigramLM = new Matrix(L2, L2);
            }

            sr.Close();
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

            //weight fea->output
            saveMatrixBin(mat_feature2output, fo);

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
