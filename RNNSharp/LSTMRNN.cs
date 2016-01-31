using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using AdvUtils;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class LSTMCell : SimpleCell
    {
        //input gate
        public float netIn;
        public float yIn;

        //forget gate
        public float netForget;
        public float yForget;

        //cell state
        public float netCellState;
        public float previousCellState;
        public float cellState;

        //internal weights and deltas
        public float wCellIn;
        public float wCellForget;
        public float wCellOut;

        //partial derivatives
        public float dSWCellIn;
        public float dSWCellForget;
        //double dSWCellState;

        //output gate
        public float netOut;
        public float yOut;
    }

    public struct LSTMWeight
    {
        //variables
        public float wInputCell;
        public float wInputInputGate;
        public float wInputForgetGate;
        public float wInputOutputGate;

    }

    public struct LSTMWeightDerivative
    {
        //partial derivatives. dont need partial derivative for output gate as it uses BP not RTRL
        public float dSInputCell;
        public float dSInputInputGate;
        public float dSInputForgetGate;
    }

    public class LSTMRNN : RNN
    {
        public LSTMCell[] neuHidden;		//neurons in hidden layer
        protected LSTMWeight[][] input2hidden;
        protected LSTMWeight[][] feature2hidden;

        protected LSTMWeightDerivative[][] input2hiddenDeri;
        protected LSTMWeightDerivative[][] feature2hiddenDeri;

        public LSTMRNN()
        {
            ModelType = MODELTYPE.LSTM;
        }

        public override void SetHiddenLayer(SimpleCell[] cells)
        {
            neuHidden = (LSTMCell[])cells;
        }

        public override SimpleCell[] GetHiddenLayer()
        {
            LSTMCell[] m = new LSTMCell[L1];
            for (int i = 0; i < L1; i++)
            {
                m[i] = new LSTMCell();
                m[i].cellOutput = neuHidden[i].cellOutput;
                m[i].cellState = neuHidden[i].cellState;
                m[i].dSWCellForget = neuHidden[i].dSWCellForget;
                m[i].dSWCellIn = neuHidden[i].dSWCellIn;
                m[i].er = neuHidden[i].er;
                m[i].mask = neuHidden[i].mask;
                m[i].netCellState = neuHidden[i].netCellState;
                m[i].netForget = neuHidden[i].netForget;
                m[i].netIn = neuHidden[i].netIn;
                m[i].netOut = neuHidden[i].netOut;
                m[i].previousCellState = neuHidden[i].previousCellState;
                m[i].wCellForget = neuHidden[i].wCellForget;
                m[i].wCellIn = neuHidden[i].wCellIn;
                m[i].wCellOut = neuHidden[i].wCellOut;
                m[i].yForget = neuHidden[i].yForget;
                m[i].yIn = neuHidden[i].yIn;
                m[i].yOut = neuHidden[i].yOut;
            }

            return m;
        }

        public LSTMWeight[][] loadLSTMWeight(BinaryReader br)
        {
            int w = br.ReadInt32();
            int h = br.ReadInt32();
            int vqSize = br.ReadInt32();

            Logger.WriteLine("Loading LSTM-Weight: width:{0}, height:{1}, vqSize:{2}...", w, h, vqSize);

            List<double> codeBook = new List<double>();
            for (int i = 0; i < vqSize; i++)
            {
                codeBook.Add(br.ReadDouble());
            }


            LSTMWeight[][] m = new LSTMWeight[w][];

            for (int i = 0; i < w; i++)
            {
                m[i] = new LSTMWeight[h];
                for (int j = 0; j < h; j++)
                {
                    int vqIdx = br.ReadByte();
                    m[i][j].wInputCell = (float)codeBook[vqIdx];

                    vqIdx = br.ReadByte();
                    m[i][j].wInputForgetGate = (float)codeBook[vqIdx];

                    vqIdx = br.ReadByte();
                    m[i][j].wInputInputGate = (float)codeBook[vqIdx];

                    vqIdx = br.ReadByte();
                    m[i][j].wInputOutputGate = (float)codeBook[vqIdx];
                }
            }

            return m;
        }

        private void saveLSTMWeight(LSTMWeight[][] weight, BinaryWriter fo)
        {
            int w = weight.Length;
            int h = weight[0].Length;
            int vqSize = 256;

            Logger.WriteLine("Saving LSTM weight matrix. width:{0}, height:{1}, vqSize:{2}", w, h, vqSize);

            fo.Write(weight.Length);
            fo.Write(weight[0].Length);

            //Build vector quantization model
            VectorQuantization vq = new VectorQuantization();
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    vq.Add(weight[i][j].wInputCell);
                    vq.Add(weight[i][j].wInputForgetGate);
                    vq.Add(weight[i][j].wInputInputGate);
                    vq.Add(weight[i][j].wInputOutputGate);
                }
            }


            double distortion = vq.BuildCodebook(vqSize);
            Logger.WriteLine("Distortion: {0}", distortion);

            //Save VQ codebook into file
            fo.Write(vqSize);
            for (int j = 0; j < vqSize; j++)
            {
                fo.Write(vq.CodeBook[j]);
            }

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    fo.Write((byte)vq.ComputeVQ(weight[i][j].wInputCell));
                    fo.Write((byte)vq.ComputeVQ(weight[i][j].wInputForgetGate));
                    fo.Write((byte)vq.ComputeVQ(weight[i][j].wInputInputGate));
                    fo.Write((byte)vq.ComputeVQ(weight[i][j].wInputOutputGate));
                }
            }

        }

        public override void loadNetBin(string filename)
        {
            Logger.WriteLine(Logger.Level.info, "Loading LSTM-RNN model: {0}", filename);

            StreamReader sr = new StreamReader(filename);
            BinaryReader br = new BinaryReader(sr.BaseStream);

            ModelType = (MODELTYPE)br.ReadInt32();
            if (ModelType != MODELTYPE.LSTM)
            {
                throw new Exception("Invalidated model format: must be LSTM-RNN format");
            }

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

            //Create cells of each layer
            CreateCell(br);

            //Load weight matrix between each two layer pairs
            //weight input->hidden
            Logger.WriteLine("Loading input2hidden weights...");
            input2hidden = loadLSTMWeight(br);

            if (DenseFeatureSize > 0)
            {
                //weight fea->hidden
                Logger.WriteLine("Loading feature2hidden weights...");
                feature2hidden = loadLSTMWeight(br);
            }

            //weight hidden->output
            Logger.WriteLine("Loading hidden2output weights...");
            Hidden2OutputWeight = loadMatrixBin(br);

            if (iflag == 1)
            {
                Logger.WriteLine("Loading CRF tag trans weights...");
                CRFTagTransWeights = loadMatrixBin(br);
            }

            sr.Close();
        }


        public void SaveHiddenLayerWeights(BinaryWriter fo)
        {
            for (int i = 0; i < L1; i++)
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

            //Save hidden layer weights
            Logger.WriteLine("Saving hidden layer weights...");
            SaveHiddenLayerWeights(fo);

            //weight input->hidden
            Logger.WriteLine("Saving input2hidden weights...");
            saveLSTMWeight(input2hidden, fo);
  
            if (DenseFeatureSize > 0)
            {
                //weight fea->hidden
                Logger.WriteLine("Saving feature2hidden weights...");
                saveLSTMWeight(feature2hidden, fo);
            }

            //weight hidden->output
            Logger.WriteLine("Saving hidden2output weights...");
            saveMatrixBin(Hidden2OutputWeight, fo);

            if (iflag == 1)
            {
                // Save Bigram
                Logger.WriteLine("Saving CRF tag trans weights...");
                saveMatrixBin(CRFTagTransWeights, fo);
            }

            fo.Close();
        }


        double Sigmoid2(double x)
        {
            //sigmoid function return a bounded output between [-2,2]
            return (4.0 / (1.0 + Math.Exp(-x))) - 2.0;
        }

        double Sigmoid2Derivative(double x)
        {
            return 4.0 * Sigmoid(x) * (1.0 - Sigmoid(x));
        }

        double Sigmoid(double x)
        {
            return (1.0 / (1.0 + Math.Exp(-x)));
        }

        double SigmoidDerivative(double x)
        {
            return Sigmoid(x) * (1.0 - Sigmoid(x));
        }


        public LSTMWeight LSTMWeightInit()
        {
            LSTMWeight w;

            //initialise each weight to random value
            w.wInputCell = RandInitWeight();
            w.wInputInputGate = RandInitWeight();
            w.wInputForgetGate = RandInitWeight();
            w.wInputOutputGate = RandInitWeight();

            return w;
        }

        public override void initWeights()
        {
            //create and initialise the weights from input to hidden layer
            input2hidden = new LSTMWeight[L1][];
            for (int i = 0; i < L1; i++)
            {
                input2hidden[i] = new LSTMWeight[L0];
                for (int j = 0; j < L0; j++)
                {
                    input2hidden[i][j] = LSTMWeightInit();
                }
            }

            if (DenseFeatureSize > 0)
            {
                feature2hidden = new LSTMWeight[L1][];
                for (int i = 0; i < L1; i++)
                {
                    feature2hidden[i] = new LSTMWeight[DenseFeatureSize];
                    for (int j = 0; j < DenseFeatureSize; j++)
                    {
                        feature2hidden[i][j] = LSTMWeightInit();
                    }
                }
            }

            //Create and intialise the weights from hidden to output layer, these are just normal weights
            Hidden2OutputWeight = new Matrix<double>(L2, L1);

            for (int i = 0; i < Hidden2OutputWeight.GetHeight(); i++)
            {
                for (int j = 0; j < Hidden2OutputWeight.GetWidth(); j++)
                {
                    Hidden2OutputWeight[i][j] = RandInitWeight();
                }
            }
        }

        public void LSTMCellInit(LSTMCell c)
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
            c.cellOutput = 0;
        }

        public override void initMem()
        {
            CreateCell(null);

            input2hiddenDeri = new LSTMWeightDerivative[L1][];
            if (DenseFeatureSize > 0)
            {
                feature2hiddenDeri = new LSTMWeightDerivative[L1][];
            }

            for (int i = 0; i < L1; i++)
            {
                input2hiddenDeri[i] = new LSTMWeightDerivative[L0];

                if (DenseFeatureSize > 0)
                {
                    feature2hiddenDeri[i] = new LSTMWeightDerivative[DenseFeatureSize];
                }
            }

            Logger.WriteLine(Logger.Level.info, "[TRACE] Initializing weights, random value is {0}", rand.NextDouble());// yy debug
            initWeights();
        }

        private void CreateCell(BinaryReader br)
        {
            neuFeatures = new SingleVector(DenseFeatureSize);
            OutputLayer = new neuron[L2];

            for (int a = 0; a < L2; a++)
            {
                OutputLayer[a].cellOutput = 0;
                OutputLayer[a].er = 0;
            }

            neuHidden = new LSTMCell[L1];
            for (int i = 0; i < L1; i++)
            {
                neuHidden[i] = new LSTMCell();
                LSTMCellInit(neuHidden[i]);
            }

            if (br != null)
            {
                //Load weight from input file
                for (int i = 0; i < L1; i++)
                {
                    neuHidden[i].wCellIn = br.ReadSingle();
                    neuHidden[i].wCellForget = br.ReadSingle();
                    neuHidden[i].wCellOut = br.ReadSingle();
                }
            }
            else
            {
                //Initialize weight by random number
                for (int i = 0; i < L1; i++)
                {
                    //internal weights, also important
                    neuHidden[i].wCellIn = RandInitWeight();
                    neuHidden[i].wCellForget = RandInitWeight();
                    neuHidden[i].wCellOut = RandInitWeight();
                }
            }
        }

        public void matrixXvectorADD(neuron[] dest, LSTMCell[] srcvec, Matrix<double> srcmatrix, int from, int to, int from2, int to2)
        {
            //ac mod
            Parallel.For(0, (to - from), parallelOption, i =>
            {
                dest[i + from].cellOutput = 0;
                for (int j = 0; j < to2 - from2; j++)
                {
                    dest[i + from].cellOutput += srcvec[j + from2].cellOutput * srcmatrix[i][j];
                }
            });
        }

        public override void LearnBackTime(State state, int numStates, int curState)
        {
        }

        public override void ComputeHiddenLayerErr()
        {
            Parallel.For(0, L1, parallelOption, i =>
            {
                //find the error by find the product of the output errors and their weight connection.
                neuHidden[i].er = 0.0;

                if (neuHidden[i].mask == false)
                {
                    for (int k = 0; k < L2; k++)
                    {
                        neuHidden[i].er += OutputLayer[k].er * Hidden2OutputWeight[k][i];
                    }
                    neuHidden[i].er = NormalizeErr(neuHidden[i].er);
                }
            });
        }

        public override void LearnOutputWeight()
        {
            //update weights for hidden to output layer
            Parallel.For(0, L1, parallelOption, i =>
            {
                for (int k = 0; k < L2; k++)
                {
                    Hidden2OutputWeight[k][i] += LearningRate * neuHidden[i].cellOutput * OutputLayer[k].er;
                }
            });
        }

        public override void learnNet(State state)
        {
            //Get sparse feature and apply it into hidden layer
            var sparse = state.SparseData;
            int sparseFeatureSize = sparse.GetNumberOfEntries();

            //put variables for derivaties in weight class and cell class
            Parallel.For(0, L1, parallelOption, i =>
            {
                LSTMCell c = neuHidden[i];

                //using the error find the gradient of the output gate
                var gradientOutputGate = LearningRate * SigmoidDerivative(c.netOut) * c.cellState * c.er;

                //internal cell state error
                var cellStateError = LearningRate * c.yOut * c.er;


                LSTMWeight[] w_i = input2hidden[i];
                LSTMWeightDerivative[] wd_i = input2hiddenDeri[i];

                var Sigmoid2Derivative_ci_netCellState_mul_ci_yIn = Sigmoid2Derivative(c.netCellState) * c.yIn;
                var Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn = Sigmoid2(c.netCellState) * SigmoidDerivative(c.netIn);
                var ci_previousCellState_mul_SigmoidDerivative_ci_netForget = c.previousCellState * SigmoidDerivative(c.netForget);

                for (int k = 0; k < sparseFeatureSize; k++)
                {
                    var entry = sparse.GetEntry(k);
                    LSTMWeightDerivative wd = wd_i[entry.Key];
                    LSTMWeight w = w_i[entry.Key];

                    wd.dSInputCell = (float)(wd.dSInputCell * c.yForget + Sigmoid2Derivative_ci_netCellState_mul_ci_yIn * entry.Value);
                    wd.dSInputInputGate = (float)(wd.dSInputInputGate * c.yForget + Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn * entry.Value);
                    wd.dSInputForgetGate = (float)(wd.dSInputForgetGate * c.yForget + ci_previousCellState_mul_SigmoidDerivative_ci_netForget * entry.Value);

                    //updates weights for input to hidden layer
                    w.wInputCell += (float)(cellStateError * wd.dSInputCell);
                    w.wInputInputGate += (float)(cellStateError * wd.dSInputInputGate);
                    w.wInputForgetGate += (float)(cellStateError * wd.dSInputForgetGate);
                    w.wInputOutputGate += (float)(gradientOutputGate * entry.Value);

                    wd_i[entry.Key] = wd;
                    w_i[entry.Key] = w;
                }

                if (DenseFeatureSize > 0)
                {
                    w_i = feature2hidden[i];
                    wd_i = feature2hiddenDeri[i];
                    for (int j = 0; j < DenseFeatureSize; j++)
                    {
                        LSTMWeightDerivative wd = wd_i[j];
                        LSTMWeight w = w_i[j];

                        wd.dSInputCell = (float)(wd.dSInputCell * c.yForget + Sigmoid2Derivative_ci_netCellState_mul_ci_yIn * neuFeatures[j]);
                        wd.dSInputInputGate = (float)(wd.dSInputInputGate * c.yForget + Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn * neuFeatures[j]);
                        wd.dSInputForgetGate = (float)(wd.dSInputForgetGate * c.yForget + ci_previousCellState_mul_SigmoidDerivative_ci_netForget * neuFeatures[j]);

                        //make the delta equal to the learning rate multiplied by the gradient multipled by the input for the connection
                        //update connection weights
                        w.wInputCell += (float)(cellStateError * wd.dSInputCell);
                        w.wInputInputGate += (float)(cellStateError * wd.dSInputInputGate);
                        w.wInputForgetGate += (float)(cellStateError * wd.dSInputForgetGate);
                        w.wInputOutputGate += (float)(gradientOutputGate * neuFeatures[j]);

                        wd_i[j] = wd;
                        w_i[j] = w;
                    }
                }

                //partial derivatives for internal connections
                c.dSWCellIn = (float)(c.dSWCellIn * c.yForget + Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn * c.cellState);

                //partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
                c.dSWCellForget = (float)(c.dSWCellForget * c.yForget + ci_previousCellState_mul_SigmoidDerivative_ci_netForget * c.previousCellState);


                //update internal weights
                c.wCellIn += (float)(cellStateError * c.dSWCellIn);
                c.wCellForget += (float)(cellStateError * c.dSWCellForget);
                c.wCellOut += (float)(gradientOutputGate * c.cellState);

                neuHidden[i] = c;
            });
        }


        // forward process. output layer consists of tag value
        public override void computeHiddenLayer(State state, bool isTrain = true)
        {
            //inputs(t) -> hidden(t)
            //Get sparse feature and apply it into hidden layer
            var sparse = state.SparseData;
            int sparseFeatureSize = sparse.GetNumberOfEntries();

            Parallel.For(0, L1, parallelOption, j =>
            {
                LSTMCell cell_j = neuHidden[j];

                //hidden(t-1) -> hidden(t)
                cell_j.previousCellState = cell_j.cellState;

                //rest the value of the net input to zero
                cell_j.netIn = 0;

                cell_j.netForget = 0;
                //reset each netCell state to zero
                cell_j.netCellState = 0;
                //reset each netOut to zero
                cell_j.netOut = 0;
                for (int i = 0; i < sparseFeatureSize; i++)
                {
                    var entry = sparse.GetEntry(i);
                    LSTMWeight w = input2hidden[j][entry.Key];
                    //loop through all forget gates in hiddden layer
                    cell_j.netIn += entry.Value * w.wInputInputGate;
                    cell_j.netForget += entry.Value * w.wInputForgetGate;
                    cell_j.netCellState += entry.Value * w.wInputCell;
                    cell_j.netOut += entry.Value * w.wInputOutputGate;
                }


                //fea(t) -> hidden(t) 
                if (DenseFeatureSize > 0)
                {
                    for (int i = 0; i < DenseFeatureSize; i++)
                    {
                        LSTMWeight w = feature2hidden[j][i];
                        cell_j.netIn += neuFeatures[i] * w.wInputInputGate;
                        cell_j.netForget += neuFeatures[i] * w.wInputForgetGate;
                        cell_j.netCellState += neuFeatures[i] * w.wInputCell;
                        cell_j.netOut += neuFeatures[i] * w.wInputOutputGate;
                    }
                }

                //include internal connection multiplied by the previous cell state
                cell_j.netIn += cell_j.previousCellState * cell_j.wCellIn;
                //squash input
                cell_j.yIn = (float)Sigmoid(cell_j.netIn);

                //include internal connection multiplied by the previous cell state
                cell_j.netForget += cell_j.previousCellState * cell_j.wCellForget;
                cell_j.yForget = (float)Sigmoid(cell_j.netForget);

                if (cell_j.mask == true)
                {
                    cell_j.cellState = 0;
                }
                else
                {
                    //cell state is equal to the previous cell state multipled by the forget gate and the cell inputs multiplied by the input gate
                    cell_j.cellState = (float)(cell_j.yForget * cell_j.previousCellState + cell_j.yIn * Sigmoid2(cell_j.netCellState));
                }

                if (isTrain == false)
                {
                    cell_j.cellState = (float)(cell_j.cellState * (1.0 - Dropout));
                }

                ////include the internal connection multiplied by the CURRENT cell state
                cell_j.netOut += cell_j.cellState * cell_j.wCellOut;

                //squash output gate 
                cell_j.yOut = (float)(Sigmoid(cell_j.netOut));

                cell_j.cellOutput = cell_j.cellState * cell_j.yOut;


                neuHidden[j] = cell_j;
            });
        }

        public override void computeOutput(double[] doutput)
        {
            matrixXvectorADD(OutputLayer, neuHidden, Hidden2OutputWeight, 0, L2, 0, L1);
            if (doutput != null)
            {
                for (int i = 0; i < L2; i++)
                {
                    doutput[i] = OutputLayer[i].cellOutput;
                }
            }

            //activation 2   --softmax on words
            SoftmaxLayer(OutputLayer);

        }

        public override void netReset(bool updateNet = false)   //cleans hidden layer activation + bptt history
        {
            Parallel.For(0, L1, parallelOption, i =>
            {
                neuHidden[i].mask = false;
                LSTMCellInit(neuHidden[i]);

                if (updateNet == true)
                {
                    Array.Clear(input2hiddenDeri[i], 0, L0);
                    if (DenseFeatureSize > 0)
                    {
                        Array.Clear(feature2hiddenDeri[i], 0, DenseFeatureSize);
                    }
                }
            });

            if (updateNet == true)
            {
                //Train mode
                for (int a = 0; a < L1; a++)
                {
                    if (rand.NextDouble() < Dropout)
                    {
                        neuHidden[a].mask = true;
                    }
                }
            }
        }
    }

}
