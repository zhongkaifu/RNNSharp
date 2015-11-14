using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace RNNSharp
{
    public class LSTMCell
    {
        //input gate
        public double netIn;
        public double yIn;

        //forget gate
        public double netForget;
        public double yForget;

        //cell state
        public double netCellState;
        public double previousCellState;
        public double cellState;

        //internal weights and deltas
        public double wCellIn;
        public double wCellForget;
        public double wCellOut;

        //partial derivatives
        public double dSWCellIn;
        public double dSWCellForget;
        //double dSWCellState;

        //output gate
        public double netOut;
        public double yOut;

        //cell output
        public double cellOutput;
    }

    public struct LSTMWeight
    {
        //variables
        public double wInputCell;
        public double wInputInputGate;
        public double wInputForgetGate;
        public double wInputOutputGate;

        //partial derivatives. dont need partial derivative for output gate as it uses BP not RTRL
        public double dSInputCell;
        public double dSInputInputGate;
        public double dSInputForgetGate;
    }


    public class LSTMRNN : RNN
    {
        public LSTMCell[] neuHidden;		//neurons in hidden layer
        protected LSTMWeight[][] mat_input2hidden;
        protected LSTMWeight[][] mat_feature2hidden;

        //for LSTM layer
        const bool NORMAL = true;
        const bool BIAS = false;

        public LSTMRNN()
        {
            m_modeltype = MODELTYPE.LSTM;
        }


        public override void GetHiddenLayer(Matrix m, int curStatus)
        {
            for (int i = 0; i <= L1; i++)
            {
                m[curStatus][i] = neuHidden[i].cellOutput;
            }
        }

        public LSTMWeight[][] loadLSTMWeight(BinaryReader br)
        {
            int w = br.ReadInt32();
            int h = br.ReadInt32();
            LSTMWeight[][] m = new LSTMWeight[w][];

            for (int i = 0; i < w; i++)
            {
                m[i] = new LSTMWeight[h];
                for (int j = 0; j < h; j++)
                {
                    m[i][j].wInputCell = br.ReadSingle();
                    m[i][j].wInputForgetGate = br.ReadSingle();
                    m[i][j].wInputInputGate = br.ReadSingle();
                    m[i][j].wInputOutputGate = br.ReadSingle();
                }
            }

            return m;
        }

        private void saveLSTMWeight(LSTMWeight[][] weight, BinaryWriter fo)
        {
            if (weight == null || weight.Length == 0)
            {
                fo.Write(0);
                fo.Write(0);
            }

            fo.Write(weight.Length);
            fo.Write(weight[0].Length);

            int w = weight.Length;
            int h = weight[0].Length;

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    fo.Write((float)weight[i][j].wInputCell);
                    fo.Write((float)weight[i][j].wInputForgetGate);
                    fo.Write((float)weight[i][j].wInputInputGate);
                    fo.Write((float)weight[i][j].wInputOutputGate);
                }
            }

        }

        public override void loadNetBin(string filename)
        {
            StreamReader sr = new StreamReader(filename);
            BinaryReader br = new BinaryReader(sr.BaseStream);

            m_modeltype = (MODELTYPE)br.ReadInt32();
            if (m_modeltype != MODELTYPE.LSTM)
            {
                throw new Exception("Invalidated model format: must be LSTM-RNN format");
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
            CreateCell(br);

            //Load weight matrix between each two layer pairs
            //weight input->hidden
            mat_input2hidden = loadLSTMWeight(br);

            if (fea_size > 0)
            {
                //weight fea->hidden
                mat_feature2hidden = loadLSTMWeight(br);
            }

            //weight hidden->output
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


        public void SaveHiddenLayerWeights(BinaryWriter fo)
        {
            for (int i = 0; i < L1 + 1; i++)
            {
                fo.Write(neuHidden[i].wCellIn);
                fo.Write(neuHidden[i].wCellForget);
                fo.Write(neuHidden[i].wCellOut);
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

            //Save hidden layer weights
            SaveHiddenLayerWeights(fo);

            //weight input->hidden
            saveLSTMWeight(mat_input2hidden, fo);
  
            if (fea_size > 0)
            {
                //weight fea->hidden
                saveLSTMWeight(mat_feature2hidden, fo);
            }

            //weight hidden->output
            saveMatrixBin(mat_hidden2output, fo);

            if (iflag == 1)
            {
                // Save Bigram
                saveMatrixBin(m_tagBigramTransition, fo);
            }

            fo.Close();
        }


        double TanH(double x)
        {
            return Math.Tanh(x);
        }

        double gPrime(double x)
        {
            double tmp = Math.Tanh(x);
            return 1 - tmp * tmp;
        }

        double activationFunctionF(double x)
        {
            return (1 / (1 + Math.Exp(-x)));
        }

        double fPrime(double x)
        {
            return activationFunctionF(x) * (1 - activationFunctionF(x));
        }


        public LSTMWeight LSTMWeightInit(int iL)
        {
            LSTMWeight w;
            //range of random values
            double inputHiddenRand = 1 / Math.Sqrt((double)iL);

            //initialise each weight to random value
            w.wInputCell = (((double)((randNext() % 100) + 1) / 100) * 2 * inputHiddenRand) - inputHiddenRand;
            w.wInputInputGate = (((double)((randNext() % 100) + 1) / 100) * 2 * inputHiddenRand) - inputHiddenRand;
            w.wInputForgetGate = (((double)((randNext() % 100) + 1) / 100) * 2 * inputHiddenRand) - inputHiddenRand;
            w.wInputOutputGate = (((double)((randNext() % 100) + 1) / 100) * 2 * inputHiddenRand) - inputHiddenRand;

            //partial derivatives
            w.dSInputCell = 0;
            w.dSInputInputGate = 0;
            w.dSInputForgetGate = 0;

            return w;
        }

        public override void initWeights()
        {
            int INPUT = m_TrainingSet.GetSparseDimension();
            //create and initialise the weights from input to hidden layer
            mat_input2hidden = new LSTMWeight[L1][];
            for (int i = 0; i < L1; i++)
            {
                mat_input2hidden[i] = new LSTMWeight[INPUT + 1];
                for (int j = 0; j <= INPUT; j++)
                {
                    mat_input2hidden[i][j] = LSTMWeightInit(INPUT);
                }
            }

            if (fea_size > 0)
            {
                mat_feature2hidden = new LSTMWeight[L1][];
                for (int i = 0; i < L1; i++)
                {
                    mat_feature2hidden[i] = new LSTMWeight[fea_size];
                    for (int j = 0; j < fea_size; j++)
                    {
                        mat_feature2hidden[i][j] = LSTMWeightInit(INPUT);
                    }
                }
            }

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

        public void LSTMCellInit(bool type, LSTMCell c)
        {
            //input gate
            c.netIn = 0;
            c.yIn = 0;

            //forget gate
            c.netForget = 0;
            c.yForget = 0;

            //cell state
            c.netCellState = 0;
            c.previousCellState = 0; //this is important
            c.cellState = 0;

            //partial derivatives
            c.dSWCellIn = 0;
            c.dSWCellForget = 0;

            //output gate
            c.netOut = 0;
            c.yOut = 0;

            //cell output
            c.cellOutput = (type == true) ? 0 : -1;
        }

        public override void initMem()
        {
            CreateCell(null);

            mat_hidden2output = new Matrix(L2, L1);

            for (int i = 0; i < MAX_RNN_HIST; i++)
            {
                m_Diff[i] = new double[L2];
            }

            m_tagBigramTransition = new Matrix(L2, L2);
            m_DeltaBigramLM = new Matrix(L2, L2);

            Console.WriteLine("[TRACE] Initializing weights, random value is {0}", random(-1.0, 1.0));// yy debug
            initWeights();
        }

        private void CreateCell(BinaryReader br)
        {
            neuFeatures = new double[fea_size];
            neuOutput = new neuron[L2];

            for (int a = 0; a < L2; a++)
            {
                neuOutput[a].cellOutput = 0;
                neuOutput[a].er = 0;
            }

            neuHidden = new LSTMCell[L1 + 1];
            for (int i = 0; i < L1; i++)
            {
                neuHidden[i] = new LSTMCell();
                LSTMCellInit(NORMAL, neuHidden[i]);
            }
            neuHidden[L1] = new LSTMCell();
            LSTMCellInit(BIAS, neuHidden[L1]);

            if (br != null)
            {
                //Load weight from input file
                for (int i = 0; i < L1 + 1; i++)
                {
                    neuHidden[i].wCellIn = br.ReadDouble();
                    neuHidden[i].wCellForget = br.ReadDouble();
                    neuHidden[i].wCellOut = br.ReadDouble();
                }
            }
            else
            {
                //Initialize weight by random number
                double internalRand = 1 / Math.Sqrt(3);
                for (int i = 0; i < L1; i++)
                {
                    //internal weights, also important
                    neuHidden[i].wCellIn = (((double)((randNext() % 100) + 1) / 100) * 2 * internalRand) - internalRand;
                    neuHidden[i].wCellForget = (((double)((randNext() % 100) + 1) / 100) * 2 * internalRand) - internalRand;
                    neuHidden[i].wCellOut = (((double)((randNext() % 100) + 1) / 100) * 2 * internalRand) - internalRand;
                }

                //internal weights
                neuHidden[L1].wCellIn = 0;
                neuHidden[L1].wCellForget = 0;
                neuHidden[L1].wCellOut = 0;
            }
        }

        public void matrixXvectorADD(neuron[] dest, LSTMCell[] srcvec, Matrix srcmatrix, int from, int to, int from2, int to2)
        {
            //ac mod
            Parallel.For(0, (to - from), parallelOption, i =>
            {
                for (int j = 0; j < to2 - from2; j++)
                {
                    dest[i + from].cellOutput += srcvec[j + from2].cellOutput * srcmatrix[i][j];
                }
            });
        }

        public void matrixXvectorADD(LSTMCell[] dest, double[] srcvec, LSTMWeight[][] srcmatrix, int from, int to, int from2, int to2)
        {
            //ac mod
            Parallel.For(0, (to - from), parallelOption, i =>
            {
                for (int j = 0; j < to2 - from2; j++)
                {
                    dest[i + from].netIn += srcvec[j + from2] * srcmatrix[i][j].wInputInputGate;
                }
            });
        }


        public override void LearnBackTime(State state, int numStates, int curState)
        {
        }


    


        public override void learnNet(State state, int timeat, bool biRNN = false)
        {
            //create delta list
            double beta2 = beta * alpha;

            if (biRNN == false)
            {
                CalculateOutputLayerError(state, timeat);
            }

            //Get sparse feature and apply it into hidden layer
            var sparse = state.GetSparseData();
            int sparseFeatureSize = sparse.GetNumberOfEntries();

            //put variables for derivaties in weight class and cell class
            Parallel.For(0, L1, parallelOption, i =>
            {
                LSTMWeight[] w_i = mat_input2hidden[i];
                LSTMCell c = neuHidden[i];
                for (int k = 0; k < sparseFeatureSize; k++)
                {
                    var entry = sparse.GetEntry(k);
                    LSTMWeight w = w_i[entry.Key];
                    w_i[entry.Key].dSInputCell = w.dSInputCell * c.yForget + gPrime(c.netCellState) * c.yIn * entry.Value;
                    w_i[entry.Key].dSInputInputGate = w.dSInputInputGate * c.yForget + TanH(c.netCellState) * fPrime(c.netIn) * entry.Value;
                    w_i[entry.Key].dSInputForgetGate = w.dSInputForgetGate * c.yForget + c.previousCellState * fPrime(c.netForget) * entry.Value;

                }

                if (fea_size > 0)
                {
                    w_i = mat_feature2hidden[i];
                    for (int j = 0; j < fea_size; j++)
                    {
                        LSTMWeight w = w_i[j];
                        w_i[j].dSInputCell = w.dSInputCell * c.yForget + gPrime(c.netCellState) * c.yIn * neuFeatures[j];
                        w_i[j].dSInputInputGate = w.dSInputInputGate * c.yForget + TanH(c.netCellState) * fPrime(c.netIn) * neuFeatures[j];
                        w_i[j].dSInputForgetGate = w.dSInputForgetGate * c.yForget + c.previousCellState * fPrime(c.netForget) * neuFeatures[j];

                    }
                }

                //partial derivatives for internal connections
                c.dSWCellIn = c.dSWCellIn * c.yForget + TanH(c.netCellState) * fPrime(c.netIn) * c.cellState;

                //partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
                c.dSWCellForget = c.dSWCellForget * c.yForget + c.previousCellState * fPrime(c.netForget) * c.previousCellState;

                neuHidden[i] = c;
            });

            //for each hidden neuron
            Parallel.For(0, L1, parallelOption, i =>
          {
              LSTMCell c = neuHidden[i];

              //find the error by find the product of the output errors and their weight connection.
              double weightedSum = 0;
              for (int k = 0; k < L2; k++)
              {
                  weightedSum += neuOutput[k].er * mat_hidden2output[k][i];
              }

              //using the error find the gradient of the output gate
              double gradientOutputGate = fPrime(c.netOut) * TanH(c.cellState) * weightedSum;

              //internal cell state error
              double cellStateError = c.yOut * weightedSum * gPrime(c.cellState);

              //weight updates

              //already done the deltas for the hidden-output connections

              //output gates. for each connection to the hidden layer
              //to the input layer
              LSTMWeight[] w_i = mat_input2hidden[i];
              for (int k = 0; k < sparseFeatureSize; k++)
              {
                  var entry = sparse.GetEntry(k);
                  //updates weights for input to hidden layer
                  if ((counter % 10) == 0)	//regularization is done every 10. step
                  {
                      w_i[entry.Key].wInputCell += alpha * cellStateError * w_i[entry.Key].dSInputCell - w_i[entry.Key].wInputCell * beta2;
                      w_i[entry.Key].wInputInputGate += alpha * cellStateError * w_i[entry.Key].dSInputInputGate - w_i[entry.Key].wInputInputGate * beta2;
                      w_i[entry.Key].wInputForgetGate += alpha * cellStateError * w_i[entry.Key].dSInputForgetGate - w_i[entry.Key].wInputForgetGate * beta2;
                      w_i[entry.Key].wInputOutputGate += alpha * gradientOutputGate * entry.Value - w_i[entry.Key].wInputOutputGate * beta2;
                  }
                  else
                  {
                      w_i[entry.Key].wInputCell += alpha * cellStateError * w_i[entry.Key].dSInputCell;
                      w_i[entry.Key].wInputInputGate += alpha * cellStateError * w_i[entry.Key].dSInputInputGate;
                      w_i[entry.Key].wInputForgetGate += alpha * cellStateError * w_i[entry.Key].dSInputForgetGate;
                      w_i[entry.Key].wInputOutputGate += alpha * gradientOutputGate * entry.Value;
                  }
              }


              if (fea_size > 0)
              {
                  w_i = mat_feature2hidden[i];
                  for (int j = 0; j < fea_size; j++)
                  {
                      //make the delta equal to the learning rate multiplied by the gradient multipled by the input for the connection
                      //update connection weights
                      if ((counter % 10) == 0)	//regularization is done every 10. step
                      {
                          w_i[j].wInputCell += alpha * cellStateError * w_i[j].dSInputCell - w_i[j].wInputCell * beta2;
                          w_i[j].wInputInputGate += alpha * cellStateError * w_i[j].dSInputInputGate - w_i[j].wInputInputGate * beta2;
                          w_i[j].wInputForgetGate += alpha * cellStateError * w_i[j].dSInputForgetGate - w_i[j].wInputForgetGate * beta2;
                          w_i[j].wInputOutputGate += alpha * gradientOutputGate * neuFeatures[j] - w_i[j].wInputOutputGate * beta2;
                      }
                      else
                      {
                          w_i[j].wInputCell += alpha * cellStateError * w_i[j].dSInputCell;
                          w_i[j].wInputInputGate += alpha * cellStateError * w_i[j].dSInputInputGate;
                          w_i[j].wInputForgetGate += alpha * cellStateError * w_i[j].dSInputForgetGate;
                          w_i[j].wInputOutputGate += alpha * gradientOutputGate * neuFeatures[j];
                      }

                  }
              }

              //for the internal connection
              double deltaOutputGateCell = alpha * gradientOutputGate * c.cellState;

              //using internal partial derivative
              double deltaInputGateCell = alpha * cellStateError * c.dSWCellIn;

              double deltaForgetGateCell = alpha * cellStateError * c.dSWCellForget;

              //update internal weights
              if ((counter % 10) == 0)	//regularization is done every 10. step
              {
                  c.wCellIn += deltaInputGateCell - c.wCellIn * beta2;
                  c.wCellForget += deltaForgetGateCell - c.wCellForget * beta2;
                  c.wCellOut += deltaOutputGateCell - c.wCellOut * beta2;
              }
              else
              {
                  c.wCellIn += deltaInputGateCell;
                  c.wCellForget += deltaForgetGateCell;
                  c.wCellOut += deltaOutputGateCell;
              }

              neuHidden[i] = c;
          });

            if (biRNN == false)
            {
                //update weights for hidden to output layer
                for (int i = 0; i <= L1; i++)
                {
                    for (int k = 0; k < L2; k++)
                    {
                        if ((counter % 10) == 0)	//regularization is done every 10. step
                        {
                            mat_hidden2output[k][i] += alpha * neuHidden[i].cellOutput * neuOutput[k].er - mat_hidden2output[k][i] * beta2;
                        }
                        else
                        {
                            mat_hidden2output[k][i] += alpha * neuHidden[i].cellOutput * neuOutput[k].er;
                        }
                    }
                }
            }

        }


        // forward process. output layer consists of tag value
        public override void computeNet(State state, double[] doutput)
        {
            //inputs(t) -> hidden(t)
            //Get sparse feature and apply it into hidden layer
            var sparse = state.GetSparseData();
            int sparseFeatureSize = sparse.GetNumberOfEntries();

            //loop through all input gates in hidden layer
            //for each hidden neuron
            Parallel.For(0, L1, parallelOption, j =>
          {
              //rest the value of the net input to zero
              neuHidden[j].netIn = 0;

              //hidden(t-1) -> hidden(t)
              neuHidden[j].previousCellState = neuHidden[j].cellState;

              //for each input neuron
              for (int i = 0; i < sparseFeatureSize; i++)
              {
                  var entry = sparse.GetEntry(i);
                  neuHidden[j].netIn += entry.Value * mat_input2hidden[j][entry.Key].wInputInputGate;
              }

          });

            //fea(t) -> hidden(t) 
            if (fea_size > 0)
            {
                matrixXvectorADD(neuHidden, neuFeatures, mat_feature2hidden, 0, L1, 0, fea_size);
            }

            Parallel.For(0, L1, parallelOption, j =>
            {
                LSTMCell cell_j = neuHidden[j];

                cell_j.netForget = 0;
                //reset each netCell state to zero
                cell_j.netCellState = 0;
                //reset each netOut to zero
                cell_j.netOut = 0;
                for (int i = 0; i < sparseFeatureSize; i++)
                {
                    var entry = sparse.GetEntry(i);
                    LSTMWeight w = mat_input2hidden[j][entry.Key];
                    //loop through all forget gates in hiddden layer
                    cell_j.netForget += entry.Value * w.wInputForgetGate;
                    cell_j.netCellState += entry.Value * w.wInputCell;
                    cell_j.netOut += entry.Value * w.wInputOutputGate;
                }

                if (fea_size > 0)
                {
                    for (int i = 0; i < fea_size; i++)
                    {
                        LSTMWeight w = mat_feature2hidden[j][i];
                        cell_j.netForget += neuFeatures[i] * w.wInputForgetGate;
                        cell_j.netCellState += neuFeatures[i] * w.wInputCell;
                        cell_j.netOut += neuFeatures[i] * w.wInputOutputGate;
                    }
                }

                //include internal connection multiplied by the previous cell state
                cell_j.netIn += cell_j.previousCellState * cell_j.wCellIn;
                //include internal connection multiplied by the previous cell state
                cell_j.netForget += cell_j.previousCellState * cell_j.wCellForget;
                cell_j.netOut += cell_j.previousCellState * cell_j.wCellOut;

                //squash input
                cell_j.yIn = activationFunctionF(cell_j.netIn);
                cell_j.yForget = activationFunctionF(cell_j.netForget);
                //squash output gate 
                cell_j.yOut = activationFunctionF(cell_j.netOut);

                //cell state is equal to the previous cell state multipled by the forget gate and the cell inputs multiplied by the input gate
                cell_j.cellState = cell_j.yForget * cell_j.previousCellState + cell_j.yIn * TanH(cell_j.netCellState);

                ////include the internal connection multiplied by the CURRENT cell state
                //cell_j.netOut += cell_j.cellState * cell_j.wCellOut;



                cell_j.cellOutput = TanH(cell_j.cellState) * cell_j.yOut;


                neuHidden[j] = cell_j;
            });


            //initialize output nodes
            for (int c = 0; c < L2; c++)
            {
                neuOutput[c].cellOutput = 0;
            }

            matrixXvectorADD(neuOutput, neuHidden, mat_hidden2output, 0, L2, 0, L1);
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

        public override void netFlush()   //cleans all activations and error vectors
        {
            int a;
            for (a = 0; a < fea_size; a++)
            {
                neuFeatures[a] = 0;
            }

            for (int i = 0; i < L1; i++)
            {
                LSTMCellInit(NORMAL, neuHidden[i]);
            }
            LSTMCellInit(BIAS, neuHidden[L1]);

            for (a = 0; a < L2; a++)
            {
                neuOutput[a].cellOutput = 0;
                neuOutput[a].er = 0;
            }
        }

        public override void netReset()   //cleans hidden layer activation + bptt history
        {
            for (int i = 0; i < L1; i++)
            {
                LSTMCellInit(NORMAL, neuHidden[i]);
            }
            LSTMCellInit(BIAS, neuHidden[L1]);
        }

        public override void copyHiddenLayerToInput()
        {

        }
    }

}
