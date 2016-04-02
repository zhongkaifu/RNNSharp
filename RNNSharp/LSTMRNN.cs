using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using AdvUtils;
using System.Numerics;
using System.Runtime.CompilerServices;

/// <summary>
/// RNNSharp written by Zhongkai Fu (fuzhongkai@gmail.com)
/// </summary>
namespace RNNSharp
{
    public class LSTMLayer : SimpleLayer
    {
        public LSTMCell[] cell;
        public LSTMLayer(int m) : base(m)
        {
            cell = new LSTMCell[m];
            for (int i = 0; i < m; i++)
            {
                cell[i] = new LSTMCell();
            }
        }
    }


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
        public double yCellState;

        //internal weights and deltas
        public double wPeepholeIn;
        public double wPeepholeForget;
        public double wPeepholeOut;

        //partial derivatives
        public double dSWPeepholeIn;
        public double dSWPeepholeForget;

        public double wCellIn;
        public double wCellForget;
        public double wCellState;
        public double wCellOut;

        public double dSWCellIn;
        public double dSWCellForget;
        public double dSWCellState;

        //output gate
        public double netOut;
        public double yOut;
    }

    public class LSTMRNN : RNN
    {
        public LSTMLayer neuHidden;

        //X - wInputInputGate
        //Y - wInputForgetGate
        //Z - wInputCell
        //W - wInputOutputGate
        protected Vector4[][] input2hidden;
        protected Vector4[][] feature2hidden;

        protected Vector4[][] Input2HiddenLearningRate;
        protected Vector4[][] Feature2HiddenLearningRate;
        protected Vector3[] PeepholeLearningRate;
        protected Vector4[] CellLearningRate;

        protected Vector3[][] input2hiddenDeri;
        protected Vector3[][] feature2hiddenDeri;

        private new Vector4 vecNormalLearningRate;
        private Vector3 vecNormalLearningRate3;

        private new Vector4 vecMaxGrad;
        private new Vector4 vecMinGrad;

        private Vector3 vecMaxGrad3;
        private Vector3 vecMinGrad3;

        public LSTMRNN(SimpleLayer hiddenLayer)
        {
            neuHidden = hiddenLayer as LSTMLayer;
            HiddenLayer = hiddenLayer;
            ModelType = MODELTYPE.LSTM;
        }

        public override SimpleLayer GetHiddenLayer()
        {
            SimpleLayer m = new SimpleLayer(L1);
            for (int i = 0; i < L1; i++)
            {
                m.cellOutput[i] = neuHidden.cellOutput[i];
                m.er[i] = neuHidden.er[i];
                m.mask[i] = neuHidden.mask[i];
            }

            return m;
        }

        public Vector4[][] loadLSTMWeight(BinaryReader br)
        {
            int w = br.ReadInt32();
            int h = br.ReadInt32();
            int vqSize = br.ReadInt32();
            Vector4[][] m = new Vector4[w][];

            Logger.WriteLine("Loading LSTM-Weight: width:{0}, height:{1}, vqSize:{2}...", w, h, vqSize);
            if (vqSize == 0)
            {
                for (int i = 0; i < w; i++)
                {
                    m[i] = new Vector4[h];
                    for (int j = 0; j < h; j++)
                    {
                        m[i][j].X = br.ReadSingle();
                        m[i][j].Y = br.ReadSingle();
                        m[i][j].Z = br.ReadSingle();
                        m[i][j].W = br.ReadSingle();
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
                    m[i] = new Vector4[h];
                    for (int j = 0; j < h; j++)
                    {
                        int vqIdx = br.ReadByte();
                        m[i][j].X = (float)codeBookInputInputGate[vqIdx];

                        vqIdx = br.ReadByte();
                        m[i][j].Y = (float)codeBookInputForgetGate[vqIdx];

                        vqIdx = br.ReadByte();
                        m[i][j].Z = (float)codeBookInputCell[vqIdx];

                        vqIdx = br.ReadByte();
                        m[i][j].W = (float)codeBookInputOutputGate[vqIdx];
                    }
                }
            }

            return m;
        }

        private void saveLSTMWeight(Vector4[][] weight, BinaryWriter fo)
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
                        fo.Write(weight[i][j].X);
                        fo.Write(weight[i][j].Y);
                        fo.Write(weight[i][j].Z);
                        fo.Write(weight[i][j].W);
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
                        vqInputInputGate.Add(weight[i][j].X);
                        vqInputForgetGate.Add(weight[i][j].Y);
                        vqInputCell.Add(weight[i][j].Z);
                        vqInputOutputGate.Add(weight[i][j].W);
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
                        fo.Write((byte)vqInputInputGate.ComputeVQ(weight[i][j].X));
                        fo.Write((byte)vqInputForgetGate.ComputeVQ(weight[i][j].Y));
                        fo.Write((byte)vqInputCell.ComputeVQ(weight[i][j].Z));
                        fo.Write((byte)vqInputOutputGate.ComputeVQ(weight[i][j].W));
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
            Hidden2OutputWeight = LoadMatrix(br);

            if (iflag == 1)
            {
                Logger.WriteLine("Loading CRF tag trans weights...");
                CRFTagTransWeights = LoadMatrix(br);
            }

            sr.Close();
        }


        public void SaveHiddenLayerWeights(BinaryWriter fo)
        {
            for (int i = 0; i < L1; i++)
            {
                fo.Write(neuHidden.cell[i].wPeepholeIn);
                fo.Write(neuHidden.cell[i].wPeepholeForget);
          //      fo.Write(neuHidden[i].wCellState);
                fo.Write(neuHidden.cell[i].wPeepholeOut);

                fo.Write(neuHidden.cell[i].wCellIn);
                fo.Write(neuHidden.cell[i].wCellForget);
                fo.Write(neuHidden.cell[i].wCellState);
                fo.Write(neuHidden.cell[i].wCellOut);
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
            SaveMatrix(Hidden2OutputWeight, fo);

            if (iflag == 1)
            {
                // Save Bigram
                Logger.WriteLine("Saving CRF tag trans weights...");
                SaveMatrix(CRFTagTransWeights, fo);
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


        public Vector4 LSTMWeightInit()
        {
            Vector4 w;

            //initialise each weight to random value
            w.X = RandInitWeight();
            w.Y = RandInitWeight();
            w.Z = RandInitWeight();
            w.W = RandInitWeight();

            return w;
        }

        public override void initWeights()
        {
            //create and initialise the weights from input to hidden layer
            input2hidden = new Vector4[L1][];
            for (int i = 0; i < L1; i++)
            {
                input2hidden[i] = new Vector4[L0];
                for (int j = 0; j < L0; j++)
                {
                    input2hidden[i][j] = LSTMWeightInit();
                }
            }

            if (DenseFeatureSize > 0)
            {
                feature2hidden = new Vector4[L1][];
                for (int i = 0; i < L1; i++)
                {
                    feature2hidden[i] = new Vector4[DenseFeatureSize];
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
            c.previousCellState = 0;
            c.cellState = 0;

            //partial derivatives
            c.dSWPeepholeIn = 0;
            c.dSWPeepholeForget = 0;
            //  c.dSWCellState = 0;

            c.dSWCellIn = 0;
            c.dSWCellForget = 0;
            c.dSWCellState = 0;
        }

        public override void CleanStatus()
        {
            Input2HiddenLearningRate = new Vector4[L1][];
            if (DenseFeatureSize > 0)
            {
                Feature2HiddenLearningRate = new Vector4[L1][];
            }

            PeepholeLearningRate = new Vector3[L1];
            CellLearningRate = new Vector4[L1];
            Parallel.For(0, L1, parallelOption, i =>
            {
                Input2HiddenLearningRate[i] = new Vector4[L0];

                if (DenseFeatureSize > 0)
                {
                    Feature2HiddenLearningRate[i] = new Vector4[DenseFeatureSize];
                }

            });

            Hidden2OutputWeightLearningRate = new Matrix<double>(L2, L1);
            vecNormalLearningRate = new Vector4(LearningRate, LearningRate, LearningRate, LearningRate);
            vecNormalLearningRate3 = new Vector3(LearningRate, LearningRate, LearningRate);
            vecMaxGrad = new Vector4((float)GradientCutoff, (float)GradientCutoff, (float)GradientCutoff, (float)GradientCutoff);
            vecMinGrad = new Vector4((float)(-GradientCutoff), (float)(-GradientCutoff), (float)(-GradientCutoff), (float)(-GradientCutoff));

            vecMaxGrad3 = new Vector3((float)GradientCutoff, (float)GradientCutoff, (float)GradientCutoff);
            vecMinGrad3 = new Vector3((float)(-GradientCutoff), (float)(-GradientCutoff), (float)(-GradientCutoff));
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
            OutputLayer = new SimpleLayer(L2);

            if (br != null)
            {
                //Load weight from input file
                for (int i = 0; i < L1; i++)
                {
                    neuHidden.cell[i].wPeepholeIn = br.ReadDouble();
                    neuHidden.cell[i].wPeepholeForget = br.ReadDouble();
                    neuHidden.cell[i].wPeepholeOut = br.ReadDouble();

                    neuHidden.cell[i].wCellIn = br.ReadDouble();
                    neuHidden.cell[i].wCellForget = br.ReadDouble();
                    neuHidden.cell[i].wCellState = br.ReadDouble();
                    neuHidden.cell[i].wCellOut = br.ReadDouble();
                }
            }
            else
            {
                //Initialize weight by random number
                for (int i = 0; i < L1; i++)
                {
                    //internal weights, also important
                    neuHidden.cell[i].wPeepholeIn = RandInitWeight();
                    neuHidden.cell[i].wPeepholeForget = RandInitWeight();
                    neuHidden.cell[i].wPeepholeOut = RandInitWeight();

                    neuHidden.cell[i].wCellIn = RandInitWeight();
                    neuHidden.cell[i].wCellForget = RandInitWeight();
                    neuHidden.cell[i].wCellState = RandInitWeight();
                    neuHidden.cell[i].wCellOut = RandInitWeight();
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector3 ComputeLearningRate(Vector3 vecDelta, ref Vector3 vecWeightLearningRate)
        {
            vecWeightLearningRate += vecDelta * vecDelta;
            return vecNormalLearningRate3 / (Vector3.SquareRoot(vecWeightLearningRate) + Vector3.One);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector4 ComputeLearningRate(Vector4 vecDelta, ref Vector4 vecWeightLearningRate)
        {
            vecWeightLearningRate += vecDelta * vecDelta;
            return vecNormalLearningRate / (Vector4.SquareRoot(vecWeightLearningRate) + Vector4.One);
        }

        public override void LearnNet(State state, int numStates, int curState)
        {
            //Get sparse feature and apply it into hidden layer
            var sparse = state.SparseData;
            int sparseFeatureSize = sparse.Count;

            //put variables for derivaties in weight class and cell class
            Parallel.For(0, L1, parallelOption, i =>
            {
                LSTMCell c = neuHidden.cell[i];

                //using the error find the gradient of the output gate
                var gradientOutputGate = (float)(SigmoidDerivative(c.netOut) * TanH(c.cellState) * neuHidden.er[i]);

                //internal cell state error
                var cellStateError = (float)(c.yOut * neuHidden.er[i] * TanHDerivative(c.cellState) + gradientOutputGate * c.wPeepholeOut);

                Vector4 vecErr = new Vector4(cellStateError, cellStateError, cellStateError, gradientOutputGate);

                var Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn = TanH(c.netCellState) * SigmoidDerivative(c.netIn);
                var ci_previousCellState_mul_SigmoidDerivative_ci_netForget = c.previousCellState * SigmoidDerivative(c.netForget);
                var Sigmoid2Derivative_ci_netCellState_mul_ci_yIn = TanHDerivative(c.netCellState) * c.yIn;

                Vector3 vecDerivate = new Vector3(
                        (float)(Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn),
                        (float)(ci_previousCellState_mul_SigmoidDerivative_ci_netForget),
                        (float)(Sigmoid2Derivative_ci_netCellState_mul_ci_yIn));
                float c_yForget = (float)c.yForget;


                Vector4[] w_i = input2hidden[i];
                Vector3[] wd_i = input2hiddenDeri[i];
                Vector4[] wlr_i = Input2HiddenLearningRate[i];
                for (int k = 0; k < sparseFeatureSize; k++)
                {
                    var entry = sparse.GetEntry(k);

                    Vector3 wd = vecDerivate * entry.Value;
                    if (curState > 0)
                    {
                        //Adding historical information
                        wd += wd_i[entry.Key] * c_yForget;
                    }
                    wd_i[entry.Key] = wd;

                    //Computing final err delta
                    Vector4 vecDelta = new Vector4(wd, entry.Value);
                    vecDelta = vecErr * vecDelta;
                    vecDelta = Vector4.Clamp(vecDelta, vecMinGrad, vecMaxGrad);

                    //Computing actual learning rate
                    Vector4 vecLearningRate = ComputeLearningRate(vecDelta, ref wlr_i[entry.Key]);
                    w_i[entry.Key] += vecLearningRate * vecDelta;
                }

                if (DenseFeatureSize > 0)
                {
                    w_i = feature2hidden[i];
                    wd_i = feature2hiddenDeri[i];
                    wlr_i = Feature2HiddenLearningRate[i];
                    for (int j = 0; j < DenseFeatureSize; j++)
                    {
                        float feature = neuFeatures[j];

                        Vector3 wd = vecDerivate * feature;
                        if (curState > 0)
                        {
                            //Adding historical information
                            wd += wd_i[j] * c_yForget;
                        }
                        wd_i[j] = wd;

                        Vector4 vecDelta = new Vector4(wd, feature);
                        vecDelta = vecErr * vecDelta;
                        vecDelta = Vector4.Clamp(vecDelta, vecMinGrad, vecMaxGrad);

                        //Computing actual learning rate
                        Vector4 vecLearningRate = ComputeLearningRate(vecDelta, ref wlr_i[j]);
                        w_i[j] += vecLearningRate * vecDelta;
                    }
                }

                //Update peephols weights

                //partial derivatives for internal connections
                c.dSWPeepholeIn = c.dSWPeepholeIn * c.yForget + Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn * c.previousCellState;

                //partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
                c.dSWPeepholeForget = c.dSWPeepholeForget * c.yForget + ci_previousCellState_mul_SigmoidDerivative_ci_netForget * c.previousCellState;

                //update internal weights
                Vector3 vecCellDelta = new Vector3((float)c.dSWPeepholeIn, (float)c.dSWPeepholeForget, (float)c.cellState);
                Vector3 vecErr3 = new Vector3(cellStateError, cellStateError, gradientOutputGate);

                vecCellDelta = vecErr3 * vecCellDelta;

                //Normalize err by gradient cut-off
                vecCellDelta = Vector3.Clamp(vecCellDelta, vecMinGrad3, vecMaxGrad3);

                //Computing actual learning rate
                Vector3 vecCellLearningRate = ComputeLearningRate(vecCellDelta, ref PeepholeLearningRate[i]);

                vecCellDelta = vecCellLearningRate * vecCellDelta;

                c.wPeepholeIn += vecCellDelta.X;
                c.wPeepholeForget += vecCellDelta.Y;
                c.wPeepholeOut += vecCellDelta.Z;



                //Update cells weights
                double c_previousCellOutput = neuHidden.previousCellOutput[i];
                //partial derivatives for internal connections
                c.dSWCellIn = c.dSWCellIn * c.yForget + Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn * c_previousCellOutput;

                //partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
                c.dSWCellForget = c.dSWCellForget * c.yForget + ci_previousCellState_mul_SigmoidDerivative_ci_netForget * c_previousCellOutput;

                c.dSWCellState = c.dSWCellState * c.yForget + Sigmoid2Derivative_ci_netCellState_mul_ci_yIn * c_previousCellOutput;

                Vector4 vecCellDelta4 = new Vector4((float)c.dSWCellIn, (float)c.dSWCellForget, (float)c.dSWCellState, (float)c_previousCellOutput);
                vecCellDelta4 = vecErr * vecCellDelta4;

                //Normalize err by gradient cut-off
                vecCellDelta4 = Vector4.Clamp(vecCellDelta4, vecMinGrad, vecMaxGrad);

                //Computing actual learning rate
                Vector4 vecCellLearningRate4 = ComputeLearningRate(vecCellDelta4, ref CellLearningRate[i]);

                vecCellDelta4 = vecCellLearningRate4 * vecCellDelta4;

                c.wCellIn += vecCellDelta4.X;
                c.wCellForget += vecCellDelta4.Y;
                c.wCellState += vecCellDelta4.Z;
                c.wCellOut += vecCellDelta4.W;


                neuHidden.cell[i] = c;
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
                LSTMCell cell_j = neuHidden.cell[j];

                //hidden(t-1) -> hidden(t)
                cell_j.previousCellState = cell_j.cellState;
                neuHidden.previousCellOutput[j] = neuHidden.cellOutput[j];

                Vector4 vecCell_j = Vector4.Zero;
                //Apply sparse weights
                Vector4[] weights = input2hidden[j];
                for (int i = 0; i < sparseFeatureSize; i++)
                {
                    var entry = sparse.GetEntry(i);
                    vecCell_j += weights[entry.Key] * entry.Value;
                }

                //Apply dense weights
                if (DenseFeatureSize > 0)
                {
                    weights = feature2hidden[j];
                    for (int i = 0; i < DenseFeatureSize; i++)
                    {
                        vecCell_j += weights[i] * neuFeatures[i];
                    }
                }

                //rest the value of the net input to zero
                cell_j.netIn = vecCell_j.X;
                cell_j.netForget = vecCell_j.Y;
                //reset each netCell state to zero
                cell_j.netCellState = vecCell_j.Z;
                //reset each netOut to zero
                cell_j.netOut = vecCell_j.W;

                double cell_j_previousCellOutput = neuHidden.previousCellOutput[j];

                //include internal connection multiplied by the previous cell state
                cell_j.netIn += cell_j.previousCellState * cell_j.wPeepholeIn + cell_j_previousCellOutput * cell_j.wCellIn;
                //squash input
                cell_j.yIn = Sigmoid(cell_j.netIn);

                //include internal connection multiplied by the previous cell state
                cell_j.netForget += cell_j.previousCellState * cell_j.wPeepholeForget + cell_j_previousCellOutput * cell_j.wCellForget;
                cell_j.yForget = Sigmoid(cell_j.netForget);

                cell_j.netCellState += cell_j_previousCellOutput * cell_j.wCellState;
                cell_j.yCellState = TanH(cell_j.netCellState);

                if (neuHidden.mask[j] == true)
                {
                    cell_j.cellState = 0;
                }
                else
                {
                    //cell state is equal to the previous cell state multipled by the forget gate and the cell inputs multiplied by the input gate
                    cell_j.cellState = cell_j.yForget * cell_j.previousCellState + cell_j.yIn * cell_j.yCellState;
                }

                if (isTrain == false)
                {
                    cell_j.cellState = cell_j.cellState * (1.0 - Dropout);
                }

                ////include the internal connection multiplied by the CURRENT cell state
                cell_j.netOut += cell_j.cellState * cell_j.wPeepholeOut + cell_j_previousCellOutput * cell_j.wCellOut;

                //squash output gate 
                cell_j.yOut = Sigmoid(cell_j.netOut);

                neuHidden.cellOutput[j] = TanH(cell_j.cellState) * cell_j.yOut;

                neuHidden.cell[j] = cell_j;
            });
        }

        public override void netReset(bool updateNet = false)   //cleans hidden layer activation + bptt history
        {
            for (int i = 0; i < L1; i++)
            {
                neuHidden.mask[i] = false;
                neuHidden.cellOutput[i] = 0;
                LSTMCellInit(neuHidden.cell[i]);
            }

            if (Dropout > 0 && updateNet == true)
            {
                //Train mode
                for (int a = 0; a < L1; a++)
                {
                    if (rand.NextDouble() < Dropout)
                    {
                        neuHidden.mask[a] = true;
                    }
                }
            }
        }
    }

}
