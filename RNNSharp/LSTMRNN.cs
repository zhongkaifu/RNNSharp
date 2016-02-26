using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using AdvUtils;
using System.Numerics;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class LSTMCell : SimpleCell
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
    }

    public struct LSTMWeight
    {
        public float wInputInputGate;
        public float wInputForgetGate;
        public float wInputCell;
        public float wInputOutputGate;
    }

    public class LSTMRNN : RNN
    {
        public LSTMCell[] neuHidden;		//neurons in hidden layer
        protected LSTMWeight[][] input2hidden;
        protected LSTMWeight[][] feature2hidden;

        protected Vector4[][] Input2HiddenLearningRate;
        protected Vector4[][] Feature2HiddenLearningRate;
        protected Vector3[] CellLearningRate;

        protected Vector3[][] input2hiddenDeri;
        protected Vector3[][] feature2hiddenDeri;

        private Vector4 vecLearningRate;
        private Vector3 vecLearningRate3;


        public LSTMRNN()
        {
            ModelType = MODELTYPE.LSTM;
        }

        public override SimpleLayer GetHiddenLayer()
        {
            SimpleLayer m = new SimpleLayer(L1);
            for (int i = 0; i < L1; i++)
            {
                m.cellOutput[i] = neuHidden[i].cellOutput;
                m.er[i] = neuHidden[i].er;
                m.mask[i] = neuHidden[i].mask;
            }

            return m;
        }

        public LSTMWeight[][] loadLSTMWeight(BinaryReader br)
        {
            int w = br.ReadInt32();
            int h = br.ReadInt32();
            int vqSize = br.ReadInt32();
            LSTMWeight[][] m = new LSTMWeight[w][];

            Logger.WriteLine("Loading LSTM-Weight: width:{0}, height:{1}, vqSize:{2}...", w, h, vqSize);
            if (vqSize == 0)
            {
                for (int i = 0; i < w; i++)
                {
                    m[i] = new LSTMWeight[h];
                    for (int j = 0; j < h; j++)
                    {
                        m[i][j].wInputInputGate = br.ReadSingle();
                        m[i][j].wInputForgetGate = br.ReadSingle();
                        m[i][j].wInputCell = br.ReadSingle();
                        m[i][j].wInputOutputGate = br.ReadSingle();
                    }
                }
            }
            else
            {
                List<double> codeBookInputCell = new List<double>();
                List<double> codeBookInputForgetGate = new List<double>();
                List<double> codeBookInputInputGate = new List<double>();
                List<double> codeBookInputOutputGate = new List<double>();

                for (int i = 0; i < vqSize; i++)
                {
                    codeBookInputInputGate.Add(br.ReadDouble());
                }

                for (int i = 0; i < vqSize; i++)
                {
                    codeBookInputForgetGate.Add(br.ReadDouble());
                }

                for (int i = 0; i < vqSize; i++)
                {
                    codeBookInputCell.Add(br.ReadDouble());
                }

                for (int i = 0; i < vqSize; i++)
                {
                    codeBookInputOutputGate.Add(br.ReadDouble());
                }

                for (int i = 0; i < w; i++)
                {
                    m[i] = new LSTMWeight[h];
                    for (int j = 0; j < h; j++)
                    {
                        int vqIdx = br.ReadByte();
                        m[i][j].wInputInputGate = (float)codeBookInputInputGate[vqIdx];

                        vqIdx = br.ReadByte();
                        m[i][j].wInputForgetGate = (float)codeBookInputForgetGate[vqIdx];

                        vqIdx = br.ReadByte();
                        m[i][j].wInputCell = (float)codeBookInputCell[vqIdx];

                        vqIdx = br.ReadByte();
                        m[i][j].wInputOutputGate = (float)codeBookInputOutputGate[vqIdx];
                    }
                }
            }

            return m;
        }

        private void saveLSTMWeight(LSTMWeight[][] weight, BinaryWriter fo)
        {
            int w = weight.Length;
            int h = weight[0].Length;
            int vqSize = 256;

            Logger.WriteLine("Saving LSTM weight matrix. width:{0}, height:{1}, vq:{2}", w, h, bVQ);

            fo.Write(weight.Length);
            fo.Write(weight[0].Length);

            if (bVQ == false)
            {
                fo.Write(0);

                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        fo.Write(weight[i][j].wInputInputGate);
                        fo.Write(weight[i][j].wInputForgetGate);
                        fo.Write(weight[i][j].wInputCell);
                        fo.Write(weight[i][j].wInputOutputGate);
                    }
                }
            }
            else
            {
                //Build vector quantization model
                VectorQuantization vqInputCell = new VectorQuantization();
                VectorQuantization vqInputForgetGate = new VectorQuantization();
                VectorQuantization vqInputInputGate = new VectorQuantization();
                VectorQuantization vqInputOutputGate = new VectorQuantization();
                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        vqInputInputGate.Add(weight[i][j].wInputInputGate);
                        vqInputForgetGate.Add(weight[i][j].wInputForgetGate);
                        vqInputCell.Add(weight[i][j].wInputCell);
                        vqInputOutputGate.Add(weight[i][j].wInputOutputGate);
                    }
                }


                double distortion = 0.0;

                distortion = vqInputInputGate.BuildCodebook(vqSize);
                Logger.WriteLine("InputInputGate distortion: {0}", distortion);

                distortion = vqInputForgetGate.BuildCodebook(vqSize);
                Logger.WriteLine("InputForgetGate distortion: {0}", distortion);

                distortion = vqInputCell.BuildCodebook(vqSize);
                Logger.WriteLine("InputCell distortion: {0}", distortion);

                distortion = vqInputOutputGate.BuildCodebook(vqSize);
                Logger.WriteLine("InputOutputGate distortion: {0}", distortion);

                fo.Write(vqSize);

                //Save InputInputGate VQ codebook into file
                for (int j = 0; j < vqSize; j++)
                {
                    fo.Write(vqInputInputGate.CodeBook[j]);
                }

                //Save InputForgetGate VQ codebook into file
                for (int j = 0; j < vqSize; j++)
                {
                    fo.Write(vqInputForgetGate.CodeBook[j]);
                }

                //Save InputCell VQ codebook into file
                for (int j = 0; j < vqSize; j++)
                {
                    fo.Write(vqInputCell.CodeBook[j]);
                }

                //Save InputOutputGate VQ codebook into file
                for (int j = 0; j < vqSize; j++)
                {
                    fo.Write(vqInputOutputGate.CodeBook[j]);
                }

                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        fo.Write((byte)vqInputInputGate.ComputeVQ(weight[i][j].wInputInputGate));
                        fo.Write((byte)vqInputForgetGate.ComputeVQ(weight[i][j].wInputForgetGate));
                        fo.Write((byte)vqInputCell.ComputeVQ(weight[i][j].wInputCell));
                        fo.Write((byte)vqInputOutputGate.ComputeVQ(weight[i][j].wInputOutputGate));
                    }
                }
            }
        }

        public override void LoadModel(string filename)
        {
            Logger.WriteLine("Loading LSTM-RNN model: {0}", filename);

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
        public override void SaveModel(string filename)
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


        double TanH(double x)
        {
            return Math.Tanh(x);
        }

        double TanHDerivative(double x)
        {
            double tmp = Math.Tanh(x);
            return 1 - tmp * tmp;
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

            for (int i = 0; i < Hidden2OutputWeight.Height; i++)
            {
                for (int j = 0; j < Hidden2OutputWeight.Width; j++)
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

        public override void CleanStatus()
        {
            Input2HiddenLearningRate = new Vector4[L1][];
            if (DenseFeatureSize > 0)
            {
                Feature2HiddenLearningRate = new Vector4[L1][];
            }

            CellLearningRate = new Vector3[L1];
            Parallel.For(0, L1, parallelOption, i =>
            {
                Input2HiddenLearningRate[i] = new Vector4[L0];

                if (DenseFeatureSize > 0)
                {
                    Feature2HiddenLearningRate[i] = new Vector4[DenseFeatureSize];
                }

            });

            Hidden2OutputWeightLearningRate = new Matrix<double>(L2, L1);
            vecLearningRate = new Vector4(LearningRate, LearningRate, LearningRate, LearningRate);
            vecLearningRate3 = new Vector3(LearningRate, LearningRate, LearningRate);
        }

        public override void InitMem()
        {
            CreateCell(null);

            input2hiddenDeri = new Vector3[L1][];
            if (DenseFeatureSize > 0)
            {
                feature2hiddenDeri = new Vector3[L1][];
            }

            for (int i = 0; i < L1; i++)
            {
                input2hiddenDeri[i] = new Vector3[L0];

                if (DenseFeatureSize > 0)
                {
                    feature2hiddenDeri[i] = new Vector3[DenseFeatureSize];
                }
            }

            Logger.WriteLine("Initializing weights, random value is {0}", rand.NextDouble());// yy debug
            initWeights();
        }

        private void CreateCell(BinaryReader br)
        {
            neuFeatures = new SingleVector(DenseFeatureSize);
            OutputLayer = new SimpleLayer(L2);

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
                    neuHidden[i].wCellIn = br.ReadDouble();
                    neuHidden[i].wCellForget = br.ReadDouble();
                    neuHidden[i].wCellOut = br.ReadDouble();
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

        public override void ComputeHiddenLayerErr()
        {
            Parallel.For(0, L1, parallelOption, i =>
            {
                //find the error by find the product of the output errors and their weight connection.
                SimpleCell cell = neuHidden[i];

                cell.er = 0.0;

                if (cell.mask == false)
                {
                    for (int k = 0; k < L2; k++)
                    {
                        cell.er += OutputLayer.er[k] * Hidden2OutputWeight[k][i];
                    }
                }
            });
        }

        public override void LearnOutputWeight()
        {
            //update weights for hidden to output layer
            Parallel.For(0, L1, parallelOption, i =>
            {
                double cellOutput = neuHidden[i].cellOutput;
                for (int k = 0; k < L2; k++)
                {
                    double delta = NormalizeGradient(cellOutput * OutputLayer.er[k]);
                    double newLearningRate = UpdateLearningRate(Hidden2OutputWeightLearningRate, k, i, delta);

                    Hidden2OutputWeight[k][i] += newLearningRate * delta;
                }
            });
        }

        public override void LearnNet(State state, int numStates, int curState)
        {
            //Get sparse feature and apply it into hidden layer
            var sparse = state.SparseData;
            int sparseFeatureSize = sparse.Count;

            //put variables for derivaties in weight class and cell class
            Parallel.For(0, L1, parallelOption, i =>
            {
                LSTMCell c = neuHidden[i];

                //using the error find the gradient of the output gate
                var gradientOutputGate = (float)NormalizeGradient(SigmoidDerivative(c.netOut) * c.cellState * c.er);

                //internal cell state error
                var cellStateError = (float)NormalizeGradient(c.yOut * c.er);

                Vector4 vecErr = new Vector4(cellStateError, cellStateError, cellStateError, gradientOutputGate);

                LSTMWeight[] w_i = input2hidden[i];
                Vector3[] wd_i = input2hiddenDeri[i];
                Vector4[] wlr_i = Input2HiddenLearningRate[i];

                var Sigmoid2Derivative_ci_netCellState_mul_ci_yIn = TanHDerivative(c.netCellState) * c.yIn;
                var Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn = TanH(c.netCellState) * SigmoidDerivative(c.netIn);
                var ci_previousCellState_mul_SigmoidDerivative_ci_netForget = c.previousCellState * SigmoidDerivative(c.netForget);

                Vector3 vecDerivate = new Vector3((float)Sigmoid2Derivative_ci_netCellState_mul_ci_yIn,
                        (float)Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn,
                        (float)ci_previousCellState_mul_SigmoidDerivative_ci_netForget);

                for (int k = 0; k < sparseFeatureSize; k++)
                {
                    var entry = sparse.GetEntry(k);
                    LSTMWeight w = w_i[entry.Key];
                    Vector4 wlr = wlr_i[entry.Key];

                    Vector3 wd = vecDerivate * entry.Value;
                    if (curState > 0)
                    {
                        wd += wd_i[entry.Key] * (float)c.yForget;
                    }
                    wd_i[entry.Key] = wd;

                    //Get new learning rate according weight delta
                    Vector4 vecDelta = new Vector4(wd, entry.Value);
                    vecDelta = vecErr * vecDelta;

                    Vector4 vecAlpha = vecDelta * vecDelta;
                    vecAlpha = wlr + vecAlpha;
                    wlr_i[entry.Key] = vecAlpha;

                    vecAlpha = vecLearningRate / (Vector4.SquareRoot(vecAlpha) + Vector4.One);
                    vecDelta = vecAlpha * vecDelta;

                    w.wInputCell += vecDelta.X;
                    w.wInputInputGate += vecDelta.Y;
                    w.wInputForgetGate += vecDelta.Z;
                    w.wInputOutputGate += vecDelta.W;

                    wd_i[entry.Key] = wd;
                    w_i[entry.Key] = w;
                }

                if (DenseFeatureSize > 0)
                {
                    w_i = feature2hidden[i];
                    wd_i = feature2hiddenDeri[i];
                    wlr_i = Feature2HiddenLearningRate[i];
                    for (int j = 0; j < DenseFeatureSize; j++)
                    {
                        LSTMWeight w = w_i[j];
                        Vector4 wlr = wlr_i[j];
                        float feature = neuFeatures[j];

                        Vector3 wd = vecDerivate * feature;
                        if (curState > 0)
                        {
                            wd += wd_i[j] * (float)c.yForget;
                        }
                        wd_i[j] = wd;

                        //Get new learning rate according weight delta
                        Vector4 vecDelta = new Vector4(wd, feature);
                        vecDelta = vecErr * vecDelta;

                        Vector4 vecAlpha = vecDelta * vecDelta;
                        vecAlpha = wlr + vecAlpha;
                        wlr_i[j] = vecAlpha;

                        vecAlpha = vecLearningRate / (Vector4.SquareRoot(vecAlpha) + Vector4.One);
                        vecDelta = vecAlpha * vecDelta;

                        w.wInputCell += vecDelta.X;
                        w.wInputInputGate += vecDelta.Y;
                        w.wInputForgetGate += vecDelta.Z;
                        w.wInputOutputGate += vecDelta.W;

                        wd_i[j] = wd;
                        w_i[j] = w;
                    }
                }

                //partial derivatives for internal connections
                c.dSWCellIn = c.dSWCellIn * c.yForget + Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn * c.cellState;

                //partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
                c.dSWCellForget = c.dSWCellForget * c.yForget + ci_previousCellState_mul_SigmoidDerivative_ci_netForget * c.previousCellState;

                //update internal weights
                Vector3 vecCellDelta = new Vector3((float)c.dSWCellIn, (float)c.dSWCellForget, (float)c.cellState);
                Vector3 vecCellErr = new Vector3(cellStateError, cellStateError, gradientOutputGate);
                Vector3 vecCellLearningRate = CellLearningRate[i];

                vecCellDelta = vecCellErr * vecCellDelta;
                vecCellLearningRate += (vecCellDelta * vecCellDelta);
                CellLearningRate[i] = vecCellLearningRate;

                //LearningRate / (1.0 + Math.Sqrt(dg));
                vecCellLearningRate = vecLearningRate3 / (Vector3.One + Vector3.SquareRoot(vecCellLearningRate));
                vecCellDelta = vecCellLearningRate * vecCellDelta;

                c.wCellIn += vecCellDelta.X;
                c.wCellForget += vecCellDelta.Y;
                c.wCellOut += vecCellDelta.Z;

                neuHidden[i] = c;
            });
        }


        // forward process. output layer consists of tag value
        public override void computeHiddenLayer(State state, bool isTrain = true)
        {
            //inputs(t) -> hidden(t)
            //Get sparse feature and apply it into hidden layer
            var sparse = state.SparseData;
            int sparseFeatureSize = sparse.Count;

            Parallel.For(0, L1, parallelOption, j =>
            {
                LSTMCell cell_j = neuHidden[j];

                //hidden(t-1) -> hidden(t)
                cell_j.previousCellState = cell_j.cellState;

                Vector4 vecCell_j = Vector4.Zero;

                LSTMWeight[] weights = input2hidden[j];
                for (int i = 0; i < sparseFeatureSize; i++)
                {
                    var entry = sparse.GetEntry(i);
                    LSTMWeight w = weights[entry.Key];

                    Vector4 vecW = new Vector4(w.wInputInputGate, w.wInputForgetGate, w.wInputCell, w.wInputOutputGate);
                    vecW *= entry.Value;
                    vecCell_j += vecW;
                }

                //fea(t) -> hidden(t) 
                if (DenseFeatureSize > 0)
                {
                    weights = feature2hidden[j];
                    for (int i = 0; i < DenseFeatureSize; i++)
                    {
                        LSTMWeight w = weights[i];
                        Vector4 vecW = new Vector4(w.wInputInputGate, w.wInputForgetGate, w.wInputCell, w.wInputOutputGate);
                        vecW *= neuFeatures[i];
                        vecCell_j += vecW;
                    }
                }

                //rest the value of the net input to zero
                cell_j.netIn = vecCell_j.X;
                cell_j.netForget = vecCell_j.Y;
                //reset each netCell state to zero
                cell_j.netCellState = vecCell_j.Z;
                //reset each netOut to zero
                cell_j.netOut = vecCell_j.W;


                //include internal connection multiplied by the previous cell state
                cell_j.netIn += cell_j.previousCellState * cell_j.wCellIn;
                //squash input
                cell_j.yIn = Sigmoid(cell_j.netIn);

                //include internal connection multiplied by the previous cell state
                cell_j.netForget += cell_j.previousCellState * cell_j.wCellForget;
                cell_j.yForget = Sigmoid(cell_j.netForget);

                if (cell_j.mask == true)
                {
                    cell_j.cellState = 0;
                }
                else
                {
                    //cell state is equal to the previous cell state multipled by the forget gate and the cell inputs multiplied by the input gate
                    cell_j.cellState = cell_j.yForget * cell_j.previousCellState + cell_j.yIn * TanH(cell_j.netCellState);
                }

                if (isTrain == false)
                {
                    cell_j.cellState = cell_j.cellState * (1.0 - Dropout);
                }

                ////include the internal connection multiplied by the CURRENT cell state
                cell_j.netOut += cell_j.cellState * cell_j.wCellOut;

                //squash output gate 
                cell_j.yOut = Sigmoid(cell_j.netOut);

                cell_j.cellOutput = cell_j.cellState * cell_j.yOut;

                neuHidden[j] = cell_j;
            });
        }

        public override void computeOutput(double[] doutput)
        {
            matrixXvectorADD(OutputLayer, neuHidden, Hidden2OutputWeight, L2, L1, 0);
            if (doutput != null)
            {
                OutputLayer.cellOutput.CopyTo(doutput, 0);
            }

            //activation 2   --softmax on words
            SoftmaxLayer(OutputLayer);

        }

        public override void netReset(bool updateNet = false)   //cleans hidden layer activation + bptt history
        {
            for (int i = 0; i < L1; i++)
            {
                neuHidden[i].mask = false;
                LSTMCellInit(neuHidden[i]);
            }

            if (Dropout > 0 && updateNet == true)
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
