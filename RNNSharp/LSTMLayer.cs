using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace RNNSharp
{

    public class LSTMGateWeight
    {
        public float[][] weights;
        public float[][] deri;
        public float[][] learningRate;
        private int layerSize;
        private int denseFeatureSize;
        protected ParallelOptions parallelOption = new ParallelOptions();

        public void CleanLearningRate()
        {
            learningRate = new float[layerSize][];

            Parallel.For(0, layerSize, parallelOption, i =>
            {

                if (denseFeatureSize > 0)
                {
                    learningRate[i] = new float[denseFeatureSize];
                }
            });
        }

        public void Init(int LayerSize, int DenseFeatureSize, bool hasDeri = true)
        {
            layerSize = LayerSize;
            denseFeatureSize = DenseFeatureSize;

            weights = new float[LayerSize][];

            if (hasDeri)
            {
                deri = new float[LayerSize][];
            }
            else
            {
                deri = null;
            }

            for (var i = 0; i < LayerSize; i++)
            {
                weights[i] = new float[DenseFeatureSize];

                if (hasDeri)
                {
                    deri[i] = new float[DenseFeatureSize];
                }
                for (var j = 0; j < DenseFeatureSize; j++)
                {
                    weights[i][j] = RNNHelper.RandInitWeight();
                }
            }
        }
    }

    public class LSTMNeuron : Neuron
    {
        public LSTMCell[] LSTMCells;
    }

    public class LSTMLayer : SimpleLayer
    {
        public LSTMCell[] LSTMCells;

        public LSTMCellWeight[] CellWeights;
        protected Vector4[] cellLearningRate;
        protected Vector3[] peepholeLearningRate;

        //Due to different data access patterns(random access for sparse features and continuous access for dense features), 
        //we use different data structure to keep features weights in order to improve performance 
        protected LSTMGateWeight wDenseInputGate;
        protected LSTMGateWeight wDenseForgetGate;
        protected LSTMGateWeight wDenseCellGate;
        public LSTMGateWeight wDenseOutputGate;

        //X - wInputInputGate
        //Y - wInputForgetGate
        //Z - wInputCell
        //W - wInputOutputGate
        protected Vector4[][] sparseFeatureWeights;
        protected Vector3[][] sparseFeatureToHiddenDeri;
        protected Vector4[][] sparseFeatureToHiddenLearningRate;


        private Vector4 vecMaxGrad;

        private Vector3 vecMaxGrad3;
        private Vector4 vecMinGrad;
        private Vector3 vecMinGrad3;

        private Vector4 vecNormalLearningRate;
        private Vector3 vecNormalLearningRate3;
        private Vector<float> vecNormalLearningRateFloat;

        public LSTMLayer(LSTMLayerConfig config) : base(config)
        {
            LSTMCells = new LSTMCell[LayerSize];
            for (var i = 0; i < LayerSize; i++)
            {
                LSTMCells[i] = new LSTMCell();
            }
        }

        public override Neuron CopyNeuronTo()
        {
            LSTMNeuron neuron = new LSTMNeuron();
            neuron.Cells = new float[Cells.Length];
            neuron.PrevCellOutputs = new float[previousCellOutputs.Length];
            Cells.CopyTo(neuron.Cells, 0);
            previousCellOutputs.CopyTo(neuron.PrevCellOutputs, 0);
            neuron.LSTMCells = new LSTMCell[LayerSize];

            for (int i = 0;i < LayerSize;i++)
            {
                neuron.LSTMCells[i] = new LSTMCell(LSTMCells[i]);
            }


            return neuron;
        }

        public override void PreUpdateWeights(Neuron neuron, float[] errs)
        {
            LSTMNeuron lstmNeuron = neuron as LSTMNeuron;
            Cells = lstmNeuron.Cells;
            previousCellOutputs = lstmNeuron.PrevCellOutputs;
            LSTMCells = lstmNeuron.LSTMCells;
            Errs = errs;
        }

        public override void ShallowCopyWeightTo(SimpleLayer destLayer)
        {
            LSTMLayer layer = destLayer as LSTMLayer;
            layer.SparseFeatureSize = SparseFeatureSize;
            layer.DenseFeatureSize = DenseFeatureSize;

            layer.sparseFeatureWeights = sparseFeatureWeights;
            layer.sparseFeatureToHiddenLearningRate = sparseFeatureToHiddenLearningRate;
            layer.sparseFeatureToHiddenDeri = sparseFeatureToHiddenDeri;

            layer.wDenseCellGate = wDenseCellGate;
            layer.wDenseForgetGate = wDenseForgetGate;
            layer.wDenseInputGate = wDenseInputGate;
            layer.wDenseOutputGate = wDenseOutputGate;

            layer.CellWeights = CellWeights;

        }

        public override void InitializeWeights(int sparseFeatureSize, int denseFeatureSize)
        {
            SparseFeatureSize = sparseFeatureSize;
            DenseFeatureSize = denseFeatureSize;

            InitializeCellWeights(null);

            if (SparseFeatureSize > 0)
            {
                sparseFeatureWeights = new Vector4[LayerSize][];
                sparseFeatureToHiddenDeri = new Vector3[LayerSize][];
                for (var i = 0; i < LayerSize; i++)
                {
                    sparseFeatureWeights[i] = new Vector4[SparseFeatureSize];
                    sparseFeatureToHiddenDeri[i] = new Vector3[SparseFeatureSize];
                    for (var j = 0; j < SparseFeatureSize; j++)
                    {
                        sparseFeatureWeights[i][j] = InitializeLSTMWeight();
                    }
                }
            }

            if (DenseFeatureSize > 0)
            {
                wDenseInputGate = new LSTMGateWeight();
                wDenseForgetGate = new LSTMGateWeight();
                wDenseCellGate = new LSTMGateWeight();
                wDenseOutputGate = new LSTMGateWeight();
                wDenseInputGate.Init(LayerSize, DenseFeatureSize);
                wDenseForgetGate.Init(LayerSize, DenseFeatureSize);
                wDenseCellGate.Init(LayerSize, DenseFeatureSize);
                wDenseOutputGate.Init(LayerSize, DenseFeatureSize, false);
            }

            Logger.WriteLine(
                "Initializing weights, sparse feature size: {0}, dense feature size: {1}, random value is {2}",
                SparseFeatureSize, DenseFeatureSize, RNNHelper.rand.NextDouble());
        }

        private void InitializeCellWeights(BinaryReader br)
        {
            CellWeights = new LSTMCellWeight[LayerSize];

            if (br != null)
            {
                //Load weight from input file
                for (var i = 0; i < LayerSize; i++)
                {
                    CellWeights[i] = new LSTMCellWeight();
                    CellWeights[i].wPeepholeIn = br.ReadDouble();
                    CellWeights[i].wPeepholeForget = br.ReadDouble();
                    CellWeights[i].wPeepholeOut = br.ReadDouble();

                    CellWeights[i].wCellIn = br.ReadDouble();
                    CellWeights[i].wCellForget = br.ReadDouble();
                    CellWeights[i].wCellState = br.ReadDouble();
                    CellWeights[i].wCellOut = br.ReadDouble();
                }
            }
            else
            {
                //Initialize weight by random number
                for (var i = 0; i < LayerSize; i++)
                {
                    CellWeights[i] = new LSTMCellWeight();
                    //internal weights, also important
                    CellWeights[i].wPeepholeIn = RNNHelper.RandInitWeight();
                    CellWeights[i].wPeepholeForget = RNNHelper.RandInitWeight();
                    CellWeights[i].wPeepholeOut = RNNHelper.RandInitWeight();

                    CellWeights[i].wCellIn = RNNHelper.RandInitWeight();
                    CellWeights[i].wCellForget = RNNHelper.RandInitWeight();
                    CellWeights[i].wCellState = RNNHelper.RandInitWeight();
                    CellWeights[i].wCellOut = RNNHelper.RandInitWeight();
                }
            }
        }

        private Vector4 InitializeLSTMWeight()
        {
            Vector4 w;

            //initialise each weight to random value
            w.X = RNNHelper.RandInitWeight();
            w.Y = RNNHelper.RandInitWeight();
            w.Z = RNNHelper.RandInitWeight();
            w.W = RNNHelper.RandInitWeight();

            return w;
        }

        private double TanH(double x)
        {
            return Math.Tanh(x);
        }

        private double TanHDerivative(double x)
        {
            var tmp = Math.Tanh(x);
            return 1 - tmp * tmp;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return Sigmoid(x) * (1.0 - Sigmoid(x));
        }

        private void SaveCellWeights(BinaryWriter fo)
        {
            for (var i = 0; i < LayerSize; i++)
            {
                fo.Write(CellWeights[i].wPeepholeIn);
                fo.Write(CellWeights[i].wPeepholeForget);
                fo.Write(CellWeights[i].wPeepholeOut);

                fo.Write(CellWeights[i].wCellIn);
                fo.Write(CellWeights[i].wCellForget);
                fo.Write(CellWeights[i].wCellState);
                fo.Write(CellWeights[i].wCellOut);
            }
        }

        private void SaveLSTMweights(LSTMGateWeight gateWeight, BinaryWriter fo, bool bVQ = false)
        {
            float[][] weights = gateWeight.weights;
            var w = weights.Length;
            var h = weights[0].Length;

            Logger.WriteLine($"Saving LSTM gate weight matrix. width: {w}, height: {h}");

            fo.Write(w);
            fo.Write(h);

            fo.Write(0);

            for (var i = 0; i < w; i++)
            {
                for (var j = 0; j < h; j++)
                {
                    fo.Write(weights[i][j]);
                }
            }
        }

        private void SaveLSTMWeights(Vector4[][] weight, BinaryWriter fo, bool bVQ = false)
        {
            var w = weight.Length;
            var h = weight[0].Length;
            var vqSize = 256;

            Logger.WriteLine("Saving LSTM weight matrix. width:{0}, height:{1}, vq:{2}", w, h, bVQ);

            fo.Write(weight.Length);
            fo.Write(weight[0].Length);

            if (bVQ == false)
            {
                fo.Write(0);

                for (var i = 0; i < w; i++)
                {
                    for (var j = 0; j < h; j++)
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
                var vqInputCell = new VectorQuantization();
                var vqInputForgetGate = new VectorQuantization();
                var vqInputInputGate = new VectorQuantization();
                var vqInputOutputGate = new VectorQuantization();
                for (var i = 0; i < w; i++)
                {
                    for (var j = 0; j < h; j++)
                    {
                        vqInputInputGate.Add(weight[i][j].X);
                        vqInputForgetGate.Add(weight[i][j].Y);
                        vqInputCell.Add(weight[i][j].Z);
                        vqInputOutputGate.Add(weight[i][j].W);
                    }
                }

                var distortion = vqInputInputGate.BuildCodebook(vqSize);
                Logger.WriteLine("InputInputGate distortion: {0}", distortion);

                distortion = vqInputForgetGate.BuildCodebook(vqSize);
                Logger.WriteLine("InputForgetGate distortion: {0}", distortion);

                distortion = vqInputCell.BuildCodebook(vqSize);
                Logger.WriteLine("InputCell distortion: {0}", distortion);

                distortion = vqInputOutputGate.BuildCodebook(vqSize);
                Logger.WriteLine("InputOutputGate distortion: {0}", distortion);

                fo.Write(vqSize);

                //Save InputInputGate VQ codebook into file
                for (var j = 0; j < vqSize; j++)
                {
                    fo.Write(vqInputInputGate.CodeBook[j]);
                }

                //Save InputForgetGate VQ codebook into file
                for (var j = 0; j < vqSize; j++)
                {
                    fo.Write(vqInputForgetGate.CodeBook[j]);
                }

                //Save InputCell VQ codebook into file
                for (var j = 0; j < vqSize; j++)
                {
                    fo.Write(vqInputCell.CodeBook[j]);
                }

                //Save InputOutputGate VQ codebook into file
                for (var j = 0; j < vqSize; j++)
                {
                    fo.Write(vqInputOutputGate.CodeBook[j]);
                }

                for (var i = 0; i < w; i++)
                {
                    for (var j = 0; j < h; j++)
                    {
                        fo.Write((byte)vqInputInputGate.ComputeVQ(weight[i][j].X));
                        fo.Write((byte)vqInputForgetGate.ComputeVQ(weight[i][j].Y));
                        fo.Write((byte)vqInputCell.ComputeVQ(weight[i][j].Z));
                        fo.Write((byte)vqInputOutputGate.ComputeVQ(weight[i][j].W));
                    }
                }
            }
        }

        private static LSTMGateWeight LoadLSTMGateWeights(BinaryReader br)
        {
            var w = br.ReadInt32();
            var h = br.ReadInt32();
            var vqSize = br.ReadInt32();
            LSTMGateWeight gateWeight = new LSTMGateWeight();

            Logger.WriteLine("Loading LSTM-Weight: width:{0}, height:{1}, vqSize:{2}...", w, h, vqSize);

            var m = new float[w][];
            gateWeight.weights = m;

            for (var i = 0; i < w; i++)
            {
                m[i] = new float[h];
                for (var j = 0; j < h; j++)
                {
                    m[i][j] = br.ReadSingle();
                }
            }

            return gateWeight;
        }

        private static Vector4[][] LoadLSTMWeights(BinaryReader br)
        {
            var w = br.ReadInt32();
            var h = br.ReadInt32();
            var vqSize = br.ReadInt32();
            var m = new Vector4[w][];

            Logger.WriteLine("Loading LSTM-Weight: width:{0}, height:{1}, vqSize:{2}...", w, h, vqSize);
            if (vqSize == 0)
            {
                for (var i = 0; i < w; i++)
                {
                    m[i] = new Vector4[h];
                    for (var j = 0; j < h; j++)
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
                var codeBookInputCell = new List<float>();
                var codeBookInputForgetGate = new List<float>();
                var codeBookInputInputGate = new List<float>();
                var codeBookInputOutputGate = new List<float>();

                for (var i = 0; i < vqSize; i++)
                {
                    codeBookInputInputGate.Add(br.ReadSingle());
                }

                for (var i = 0; i < vqSize; i++)
                {
                    codeBookInputForgetGate.Add(br.ReadSingle());
                }

                for (var i = 0; i < vqSize; i++)
                {
                    codeBookInputCell.Add(br.ReadSingle());
                }

                for (var i = 0; i < vqSize; i++)
                {
                    codeBookInputOutputGate.Add(br.ReadSingle());
                }

                for (var i = 0; i < w; i++)
                {
                    m[i] = new Vector4[h];
                    for (var j = 0; j < h; j++)
                    {
                        int vqIdx = br.ReadByte();
                        m[i][j].X = codeBookInputInputGate[vqIdx];

                        vqIdx = br.ReadByte();
                        m[i][j].Y = codeBookInputForgetGate[vqIdx];

                        vqIdx = br.ReadByte();
                        m[i][j].Z = codeBookInputCell[vqIdx];

                        vqIdx = br.ReadByte();
                        m[i][j].W = codeBookInputOutputGate[vqIdx];
                    }
                }
            }

            return m;
        }

        public override void Save(BinaryWriter fo)
        {
            fo.Write(LayerSize);
            fo.Write(SparseFeatureSize);
            fo.Write(DenseFeatureSize);

            //Save hidden layer weights
            Logger.WriteLine(
                $"Saving LSTM layer, size = '{LayerSize}', sparse feature size = '{SparseFeatureSize}', dense feature size = '{DenseFeatureSize}'");

            SaveCellWeights(fo);

            if (SparseFeatureSize > 0)
            {
                //weight input->hidden
                Logger.WriteLine("Saving sparse feature weights...");
                SaveLSTMWeights(sparseFeatureWeights, fo);
            }

            if (DenseFeatureSize > 0)
            {
                //weight fea->hidden
                Logger.WriteLine("Saving dense feature weights...");

                SaveLSTMweights(wDenseInputGate, fo);
                SaveLSTMweights(wDenseCellGate, fo);
                SaveLSTMweights(wDenseForgetGate, fo);
                SaveLSTMweights(wDenseOutputGate, fo);
            }
        }

        public static LSTMLayer Load(BinaryReader br, LayerType layerType)
        {
            LSTMLayerConfig config = new LSTMLayerConfig();
            config.LayerSize = br.ReadInt32();
            config.LayerType = layerType;
            LSTMLayer layer = new LSTMLayer(config);

            layer.SparseFeatureSize = br.ReadInt32();
            layer.DenseFeatureSize = br.ReadInt32();

            //Create cells of each layer
            layer.InitializeCellWeights(br);

            //Load weight matrix between each two layer pairs
            //weight input->hidden
            if (layer.SparseFeatureSize > 0)
            {
                Logger.WriteLine("Loading sparse feature weights...");
                layer.sparseFeatureWeights = LoadLSTMWeights(br);
            }

            if (layer.DenseFeatureSize > 0)
            {
                //weight fea->hidden
                Logger.WriteLine("Loading dense feature weights...");
                layer.wDenseInputGate = LoadLSTMGateWeights(br);
                layer.wDenseCellGate = LoadLSTMGateWeights(br);
                layer.wDenseForgetGate = LoadLSTMGateWeights(br);
                layer.wDenseOutputGate = LoadLSTMGateWeights(br);
            }

            return layer;
        }

        public override void CleanLearningRate()
        {
            if (SparseFeatureSize > 0)
            {
                sparseFeatureToHiddenLearningRate = new Vector4[LayerSize][];
            }

            if (DenseFeatureSize > 0)
            {
                wDenseCellGate.CleanLearningRate();
                wDenseForgetGate.CleanLearningRate();
                wDenseInputGate.CleanLearningRate();
                wDenseOutputGate.CleanLearningRate();
            }

            peepholeLearningRate = new Vector3[LayerSize];
            cellLearningRate = new Vector4[LayerSize];
            Parallel.For(0, LayerSize, parallelOption, i =>
            {
                if (SparseFeatureSize > 0)
                {
                    sparseFeatureToHiddenLearningRate[i] = new Vector4[SparseFeatureSize];
                }
            });

            vecNormalLearningRate = new Vector4(RNNHelper.LearningRate, RNNHelper.LearningRate, RNNHelper.LearningRate,
                RNNHelper.LearningRate);
            vecNormalLearningRate3 = new Vector3(RNNHelper.LearningRate, RNNHelper.LearningRate, RNNHelper.LearningRate);

            vecNormalLearningRateFloat = new Vector<float>(RNNHelper.LearningRate);

            vecMaxGrad = new Vector4(RNNHelper.GradientCutoff, RNNHelper.GradientCutoff, RNNHelper.GradientCutoff,
                RNNHelper.GradientCutoff);
            vecMinGrad = new Vector4(-RNNHelper.GradientCutoff, -RNNHelper.GradientCutoff, -RNNHelper.GradientCutoff,
                -RNNHelper.GradientCutoff);

            vecMaxGrad3 = new Vector3(RNNHelper.GradientCutoff, RNNHelper.GradientCutoff, RNNHelper.GradientCutoff);
            vecMinGrad3 = new Vector3(-RNNHelper.GradientCutoff, -RNNHelper.GradientCutoff, -RNNHelper.GradientCutoff);
        }

        // forward process. output layer consists of tag value
        public override void ForwardPass(SparseVector sparseFeature, float[] denseFeature)
        {
            //inputs(t) -> hidden(t)
            //Get sparse feature and apply it into hidden layer
            SparseFeature = sparseFeature;
            DenseFeature = denseFeature;

            Parallel.For(0, LayerSize, parallelOption, j =>
            {
                var cell_j = LSTMCells[j];
                var cellWeight_j = CellWeights[j];

                //hidden(t-1) -> hidden(t)
                cell_j.previousCellState = cell_j.cellState;
                previousCellOutputs[j] = Cells[j];

                var vecCell_j = Vector4.Zero;

                if (SparseFeatureSize > 0)
                {
                    //Apply sparse weights
                    var weights = sparseFeatureWeights[j];
                    foreach (var pair in SparseFeature)
                    {
                        vecCell_j += weights[pair.Key] * pair.Value;
                    }
                }

                if (DenseFeatureSize > 0)
                {
                    //Apply dense weights
                    var k = 0;
                    float[] denseInputGateWeight_j = wDenseInputGate.weights[j];
                    float[] denseForgetGateWeight_j = wDenseForgetGate.weights[j];
                    float[] denseCellGateWeight_j = wDenseCellGate.weights[j];
                    float[] denseOutputGateWeight_j = wDenseOutputGate.weights[j];
                    while (k < DenseFeatureSize - Vector<float>.Count)
                    {
                        var vX = new Vector<float>(denseInputGateWeight_j, k);
                        var vY = new Vector<float>(denseForgetGateWeight_j, k);
                        var vZ = new Vector<float>(denseCellGateWeight_j, k);
                        var vW = new Vector<float>(denseOutputGateWeight_j, k);
                        var vFeature = new Vector<float>(DenseFeature, k);

                        vecCell_j.X += Vector.Dot(vX, vFeature);
                        vecCell_j.Y += Vector.Dot(vY, vFeature);
                        vecCell_j.Z += Vector.Dot(vZ, vFeature);
                        vecCell_j.W += Vector.Dot(vW, vFeature);

                        k += Vector<float>.Count;
                    }

                    while (k < DenseFeatureSize)
                    {
                        vecCell_j.X += denseInputGateWeight_j[k] * DenseFeature[k];
                        vecCell_j.Y += denseForgetGateWeight_j[k] * DenseFeature[k];
                        vecCell_j.Z += denseCellGateWeight_j[k] * DenseFeature[k];
                        vecCell_j.W += denseOutputGateWeight_j[k] * DenseFeature[k];
                        k++;
                    }
                }

                //rest the value of the net input to zero
                cell_j.netIn = vecCell_j.X;
                cell_j.netForget = vecCell_j.Y;
                //reset each netCell state to zero
                cell_j.netCellState = vecCell_j.Z;
                //reset each netOut to zero
                cell_j.netOut = vecCell_j.W;

                var cell_j_previousCellOutput = previousCellOutputs[j];

                //include internal connection multiplied by the previous cell state
                cell_j.netIn += cell_j.previousCellState * cellWeight_j.wPeepholeIn + cell_j_previousCellOutput * cellWeight_j.wCellIn;
                //squash input
                cell_j.yIn = Sigmoid(cell_j.netIn);

                //include internal connection multiplied by the previous cell state
                cell_j.netForget += cell_j.previousCellState * cellWeight_j.wPeepholeForget +
                                    cell_j_previousCellOutput * cellWeight_j.wCellForget;
                cell_j.yForget = Sigmoid(cell_j.netForget);

                cell_j.netCellState += cell_j_previousCellOutput * cellWeight_j.wCellState;
                cell_j.yCellState = TanH(cell_j.netCellState);

                //cell state is equal to the previous cell state multipled by the forget gate and the cell inputs multiplied by the input gate
                cell_j.cellState = cell_j.yForget * cell_j.previousCellState + cell_j.yIn * cell_j.yCellState;

                ////include the internal connection multiplied by the CURRENT cell state
                cell_j.netOut += cell_j.cellState * cellWeight_j.wPeepholeOut + cell_j_previousCellOutput * cellWeight_j.wCellOut;

                //squash output gate
                cell_j.yOut = Sigmoid(cell_j.netOut);

                Cells[j] = (float)(TanH(cell_j.cellState) * cell_j.yOut);

                LSTMCells[j] = cell_j;
            });
        }


        private void UpdateOutputGateWeights(LSTMGateWeight gateWeight, int i, float err)
        {
            var j = 0;
            float[] learningrate_i = gateWeight.learningRate[i];
            float[] weights_i = gateWeight.weights[i];
            while (j < DenseFeatureSize - Vector<float>.Count)
            {
                Vector<float> vecDelta = new Vector<float>(DenseFeature, j);
                vecDelta = vecDelta * err;
                vecDelta = RNNHelper.NormalizeGradient(vecDelta);
                var wlr_i = new Vector<float>(learningrate_i, j);
                var vecLearningRate = ComputeLearningRate(vecDelta, ref wlr_i);

                var w_i = new Vector<float>(weights_i, j);
                w_i += vecLearningRate * vecDelta;

                w_i.CopyTo(weights_i, j);
                wlr_i.CopyTo(learningrate_i, j);

                j += Vector<float>.Count;
            }

            while (j < DenseFeatureSize)
            {
                float delta = DenseFeature[j] * err;
                delta = RNNHelper.NormalizeGradient(delta);
                var wlr_i = learningrate_i[j];
                var learningRate = ComputeLearningRate(delta, ref wlr_i);

                weights_i[j] += learningRate * delta;
                learningrate_i[j] = wlr_i;

                j++;
            }
        }

        private void UpdateGateWeights(LSTMGateWeight gateWeight, int i, float featureDerivate, float c_yForget, float err)
        {
            var j = 0;
            float[] deri_i = gateWeight.deri[i];
            float[] learningrate_i = gateWeight.learningRate[i];
            float[] weights_i = gateWeight.weights[i];
            while (j < DenseFeatureSize - Vector<float>.Count)
            {
                var feature = new Vector<float>(DenseFeature, j);
                var wd = feature * featureDerivate;
                var wd_i = new Vector<float>(deri_i, j);
                wd += wd_i * c_yForget;
                wd.CopyTo(deri_i, j);

                Vector<float> vecDelta = wd * err;
                vecDelta = RNNHelper.NormalizeGradient(vecDelta);
                var wlr_i = new Vector<float>(learningrate_i, j);
                var vecLearningRate = ComputeLearningRate(vecDelta, ref wlr_i);

                var w_i = new Vector<float>(weights_i, j);
                w_i += vecLearningRate * vecDelta;

                w_i.CopyTo(weights_i, j);
                wlr_i.CopyTo(learningrate_i, j);

                j += Vector<float>.Count;
            }

            while (j < DenseFeatureSize)
            {
                var wd = DenseFeature[j] * featureDerivate;
                wd += deri_i[j] * c_yForget;
                deri_i[j] = wd;

                float delta = wd * err;
                delta = RNNHelper.NormalizeGradient(delta);
                var wlr_i = learningrate_i[j];
                var learningRate = ComputeLearningRate(delta, ref wlr_i);

                weights_i[j] += learningRate * delta;
                learningrate_i[j] = wlr_i;

                j++;
            }
        }

        public override void BackwardPass()
        {
            //put variables for derivaties in weight class and cell class
            Parallel.For(0, LayerSize, parallelOption, i =>
            {
                var c = LSTMCells[i];
                var cellWeight = CellWeights[i];

                //using the error find the gradient of the output gate
                var gradientOutputGate = (float)(SigmoidDerivative(c.netOut) * TanH(c.cellState) * Errs[i]);

                //internal cell state error
                var cellStateError =
                    (float)(c.yOut * Errs[i] * TanHDerivative(c.cellState) + gradientOutputGate * cellWeight.wPeepholeOut);

                var vecErr = new Vector4(cellStateError, cellStateError, cellStateError, gradientOutputGate);

                var Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn = TanH(c.netCellState) *
                                                                              SigmoidDerivative(c.netIn);
                var ci_previousCellState_mul_SigmoidDerivative_ci_netForget = c.previousCellState *
                                                                              SigmoidDerivative(c.netForget);
                var Sigmoid2Derivative_ci_netCellState_mul_ci_yIn = TanHDerivative(c.netCellState) * c.yIn;

                var vecDerivate = new Vector3(
                    (float)Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn,
                    (float)ci_previousCellState_mul_SigmoidDerivative_ci_netForget,
                    (float)Sigmoid2Derivative_ci_netCellState_mul_ci_yIn);
                var c_yForget = (float)c.yForget;

                if (SparseFeatureSize > 0)
                {
                    //Get sparse feature and apply it into hidden layer
                    var w_i = sparseFeatureWeights[i];
                    var wd_i = sparseFeatureToHiddenDeri[i];
                    var wlr_i = sparseFeatureToHiddenLearningRate[i];

                    foreach (var entry in SparseFeature)
                    {
                        var wd = vecDerivate * entry.Value;
                        //Adding historical information
                        wd += wd_i[entry.Key] * c_yForget;
                        wd_i[entry.Key] = wd;

                        //Computing final err delta
                        var vecDelta = new Vector4(wd, entry.Value);
                        vecDelta = vecErr * vecDelta;
                        vecDelta = Vector4.Clamp(vecDelta, vecMinGrad, vecMaxGrad);

                        //Computing actual learning rate
                        var vecLearningRate = ComputeLearningRate(vecDelta, ref wlr_i[entry.Key]);
                        w_i[entry.Key] += vecLearningRate * vecDelta;
                    }
                }

                if (DenseFeatureSize > 0)
                {
                    UpdateGateWeights(wDenseInputGate, i, vecDerivate.X, c_yForget, cellStateError);
                    UpdateGateWeights(wDenseForgetGate, i, vecDerivate.Y, c_yForget, cellStateError);
                    UpdateGateWeights(wDenseCellGate, i, vecDerivate.Z, c_yForget, cellStateError);
                    UpdateOutputGateWeights(wDenseOutputGate, i, gradientOutputGate);
                }

                //Update peephols weights
                //partial derivatives for internal connections
                cellWeight.dSWPeepholeIn = cellWeight.dSWPeepholeIn * c.yForget +
                                  Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn * c.previousCellState;

                //partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
                cellWeight.dSWPeepholeForget = cellWeight.dSWPeepholeForget * c.yForget +
                                      ci_previousCellState_mul_SigmoidDerivative_ci_netForget * c.previousCellState;

                //update internal weights
                var vecCellDelta = new Vector3((float)cellWeight.dSWPeepholeIn, (float)cellWeight.dSWPeepholeForget, (float)c.cellState);
                var vecErr3 = new Vector3(cellStateError, cellStateError, gradientOutputGate);

                vecCellDelta = vecErr3 * vecCellDelta;

                //Normalize err by gradient cut-off
                vecCellDelta = Vector3.Clamp(vecCellDelta, vecMinGrad3, vecMaxGrad3);

                //Computing actual learning rate
                var vecCellLearningRate = ComputeLearningRate(vecCellDelta, ref peepholeLearningRate[i]);

                vecCellDelta = vecCellLearningRate * vecCellDelta;

                cellWeight.wPeepholeIn += vecCellDelta.X;
                cellWeight.wPeepholeForget += vecCellDelta.Y;
                cellWeight.wPeepholeOut += vecCellDelta.Z;

                //Update cells weights
                var c_previousCellOutput = previousCellOutputs[i];
                //partial derivatives for internal connections
                cellWeight.dSWCellIn = cellWeight.dSWCellIn * c.yForget +
                              Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn * c_previousCellOutput;

                //partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
                cellWeight.dSWCellForget = cellWeight.dSWCellForget * c.yForget +
                                  ci_previousCellState_mul_SigmoidDerivative_ci_netForget * c_previousCellOutput;

                cellWeight.dSWCellState = cellWeight.dSWCellState * c.yForget +
                                 Sigmoid2Derivative_ci_netCellState_mul_ci_yIn * c_previousCellOutput;

                var vecCellDelta4 = new Vector4((float)cellWeight.dSWCellIn, (float)cellWeight.dSWCellForget, (float)cellWeight.dSWCellState,
                    c_previousCellOutput);
                vecCellDelta4 = vecErr * vecCellDelta4;

                //Normalize err by gradient cut-off
                vecCellDelta4 = Vector4.Clamp(vecCellDelta4, vecMinGrad, vecMaxGrad);

                //Computing actual learning rate
                var vecCellLearningRate4 = ComputeLearningRate(vecCellDelta4, ref cellLearningRate[i]);

                vecCellDelta4 = vecCellLearningRate4 * vecCellDelta4;

                cellWeight.wCellIn += vecCellDelta4.X;
                cellWeight.wCellForget += vecCellDelta4.Y;
                cellWeight.wCellState += vecCellDelta4.Z;
                cellWeight.wCellOut += vecCellDelta4.W;

                LSTMCells[i] = c;
            });
        }

        public override void ComputeLayerErr(SimpleLayer nextLayer)
        {
            var layer = nextLayer as LSTMLayer;

            if (layer != null)
            {
                Parallel.For(0, LayerSize, parallelOption, i =>
                {
                    var err = 0.0f;
                    for (var k = 0; k < nextLayer.LayerSize; k++)
                    {
                        err += layer.Errs[k] * layer.wDenseOutputGate.weights[k][i];
                    }
                    Errs[i] = RNNHelper.NormalizeGradient(err);
                });
            }
            else
            {
                base.ComputeLayerErr(nextLayer);
            }
        }

        public override void Reset()
        {
            for (var i = 0; i < LayerSize; i++)
            {
                Cells[i] = 0;
                InitializeLSTMCell(LSTMCells[i], CellWeights[i]);
            }
        }

        private void InitializeLSTMCell(LSTMCell c, LSTMCellWeight cw)
        {
            c.previousCellState = 0;
            c.cellState = 0;

            //partial derivatives
            cw.dSWPeepholeIn = 0;
            cw.dSWPeepholeForget = 0;

            cw.dSWCellIn = 0;
            cw.dSWCellForget = 0;
            cw.dSWCellState = 0;
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector<float> ComputeLearningRate(Vector<float> vecDelta, ref Vector<float> vecWeightLearningRate)
        {
            vecWeightLearningRate += vecDelta * vecDelta;

            return vecNormalLearningRateFloat / (Vector.SquareRoot(vecWeightLearningRate) + Vector<float>.One);

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ComputeLearningRate(float delta, ref float weightLearningRate)
        {
            weightLearningRate += delta * delta;

            return (float)(RNNHelper.LearningRate / (Math.Sqrt(weightLearningRate) + 1.0));
        }
    }

    public class LSTMCellWeight
    {
        //The following fields are for both forward and backward
        public double dSWCellForget;
        public double dSWCellIn;
        public double dSWCellState;
        public double dSWPeepholeForget;
        public double dSWPeepholeIn;

        //Weights for each gate
        public double wCellForget;
        public double wCellIn;
        public double wCellOut;
        public double wCellState;
        public double wPeepholeForget;
        public double wPeepholeIn;
        public double wPeepholeOut;
    }

    public class LSTMCell
    {
        //The following fields are only for forward
        public double previousCellState;
        public double cellState;

        public double netCellState;
        public double netForget;
        public double netIn;
        public double netOut;

        public double yCellState;
        public double yForget;
        public double yIn;
        public double yOut;

        public LSTMCell()
        {

        }

        public LSTMCell(LSTMCell cell)
        {
            Set(cell);
        }

        public void Set(LSTMCell cell)
        {
            previousCellState = cell.previousCellState;
            cellState = cell.cellState;
            netCellState = cell.netCellState;
            netForget = cell.netForget;
            netIn = cell.netIn;
            netOut = cell.netOut;
            yCellState = cell.yCellState;
            yForget = cell.yForget;
            yIn = cell.yIn;
            yOut = cell.yOut;
        }
    }
}