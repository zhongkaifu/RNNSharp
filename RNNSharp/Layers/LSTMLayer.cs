using AdvUtils;
using RNNSharp.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace RNNSharp
{

    public class LSTMGateWeight
    {
        public float[][] weights;
        public float[][] weightsDelta;

        public float[][] deri;
        public float[][] learningRate;
        public int layerSize;
        public int denseFeatureSize;
        protected ParallelOptions parallelOption = new ParallelOptions();

        public LSTMGateWeight CloneSharedWeights()
        {
            LSTMGateWeight gateWeight = new LSTMGateWeight();
            gateWeight.InitWeights(layerSize, denseFeatureSize);
            gateWeight.weights = weights;
            gateWeight.weightsDelta = weightsDelta;
            gateWeight.learningRate = learningRate;

            return gateWeight;
        }

        /// <summary>
        /// Update weights
        /// </summary>
        public void UpdateWeights()
        {
            Vector<float> vecMiniBatchSize = new Vector<float>(RNNHelper.MiniBatchSize);
            for (var i = 0; i < layerSize; i++)
            {
                var j = 0;
                var weights_i = weights[i];
                var weightsDelta_i = weightsDelta[i];
                var learningrate_i = learningRate[i];
                var moreItems = (denseFeatureSize % Vector<float>.Count);
                while (j < denseFeatureSize - moreItems)
                {
                    //Vectorize weights delta
                    Vector<float> vecDelta = new Vector<float>(weightsDelta_i, j);
                    Vector<float>.Zero.CopyTo(weightsDelta_i, j);

                    //Normalize weights delta
                    vecDelta = vecDelta / vecMiniBatchSize;
                    vecDelta = RNNHelper.NormalizeGradient(vecDelta);

                    //Get learning rate dymanticly
                    var wlr_i = new Vector<float>(learningrate_i, j);
                    var vecLearningRate = RNNHelper.ComputeLearningRate(vecDelta, ref wlr_i);
                    wlr_i.CopyTo(learningrate_i, j);

                    //Update weights
                    Vector<float> vecWeights = new Vector<float>(weights_i, j);
                    vecWeights += vecLearningRate * vecDelta;
                    vecWeights.CopyTo(weights_i, j);

                    j += Vector<float>.Count;
                }

                while (j < denseFeatureSize)
                {
                    var delta = weightsDelta_i[j];
                    weightsDelta_i[j] = 0;

                    delta = delta / RNNHelper.MiniBatchSize;
                    delta = RNNHelper.NormalizeGradient(delta);

                    var wlr_i = learningrate_i[j];
                    var learningRate = ComputeLearningRate(delta, ref wlr_i);
                    learningrate_i[j] = wlr_i;

                    weights_i[j] += learningRate * delta;
                    j++;
                }
            }

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ComputeLearningRate(float delta, ref float weightLearningRate)
        {
            if (RNNHelper.IsConstAlpha)
            {
                return RNNHelper.LearningRate;
            }

            weightLearningRate += delta * delta;
            return (float)(RNNHelper.LearningRate / (Math.Sqrt(weightLearningRate) + 1.0));
        }

        public void CleanStatusForTraining()
        {
            if (denseFeatureSize > 0)
            {
                for (var i = 0; i < layerSize; i++)
                {
                    Array.Clear(learningRate[i], 0, learningRate[i].Length);
                    Array.Clear(weightsDelta[i], 0, weightsDelta[i].Length);
                }
            }
        }

        public void InitWeights(int LayerSize, int DenseFeatureSize)
        {
            layerSize = LayerSize;
            denseFeatureSize = DenseFeatureSize;

            weights = new float[layerSize][];
            for (var i = 0; i < layerSize; i++)
            {
                weights[i] = new float[denseFeatureSize];            
                for (var j = 0; j < denseFeatureSize; j++)
                {
                    weights[i][j] = RNNHelper.RandInitWeight();
                }
            }

        }

        public void InitInternaTrainingParameters(bool hasDeri = true)
        {
            if (learningRate == null)
            {
                learningRate = new float[layerSize][];
                weightsDelta = new float[layerSize][];
                for (var i = 0; i < layerSize; i++)
                {
                    learningRate[i] = new float[denseFeatureSize];
                    weightsDelta[i] = new float[denseFeatureSize];
                }
            }

            if (hasDeri)
            {
                deri = new float[layerSize][];
                for (var i = 0; i < layerSize; i++)
                {
                    deri[i] = new float[denseFeatureSize];
                }
            }
        }
    }

    public class LSTMNeuron : Neuron
    {
        public LSTMCell[] LSTMCells;

        public LSTMNeuron(int length) : base(length)
        {
            LSTMCells = new LSTMCell[length];
            LSTMCells = new LSTMCell[length];

            for (int k = 0; k < length; k++)
            {
                LSTMCells[k] = new LSTMCell();
                LSTMCells[k] = new LSTMCell();
            }
        }
    }

    public class LSTMLayer : ILayer
    {
        LSTMCell[] LSTMCells;
        LSTMCellWeight[] CellWeights;
        LSTMCellWeightDeri[] CellWeightsDeri;
        Vector4[] cellLearningRate;
        Vector3[] peepholeLearningRate;

        Vector4[] cellDelta;
        Vector3[] peepholeDelta;

        //Due to different data access patterns(random access for sparse features and continuous access for dense features), 
        //we use different data structure to keep features weights in order to improve performance 
        LSTMGateWeight wDenseInputGate;
        LSTMGateWeight wDenseForgetGate;
        LSTMGateWeight wDenseCellGate;
        LSTMGateWeight wDenseOutputGate;

        //X - wInputInputGate
        //Y - wInputForgetGate
        //Z - wInputCell
        //W - wInputOutputGate
        Vector4[][] sparseFeatureWeights;
        Vector4[][] sparseFeatureLearningRate;
        Dictionary<int, Vector3>[] sparseFeatureToHiddenDeri;

        Vector4[][] sparseFeatureWeightsDelta;

        Vector4 vecMaxGrad;
        Vector3 vecMaxGrad3;
        Vector4 vecMinGrad;
        Vector3 vecMinGrad3;

        Vector4 vecNormalLearningRate;
        Vector3 vecNormalLearningRate3;

        LSTMLayerConfig config;

        public float[] Cells { get; set; }
        public float[] Errs { get; set; }
        public int SparseFeatureSize { get; set; }
        public int DenseFeatureSize { get; set; }
        public int LayerSize
        {
            get { return config.LayerSize; }
            set { config.LayerSize = value; }
        }

        public LayerType LayerType
        {
            get { return config.LayerType; }
            set { config.LayerType = value; }
        }

        public LayerConfig LayerConfig { get; set; }

        public List<SparseVector> SparseFeatureGroups { get; set; }
        public List<float[]> DenseFeatureGroups { get; set; }

        public LSTMLayer() { }
        public LSTMLayer(LSTMLayerConfig config)
        {
            this.config = config;

            Cells = new float[LayerSize];
            Errs = new float[LayerSize];

            LSTMCells = new LSTMCell[LayerSize];
            for (var i = 0; i < LayerSize; i++)
            {
                LSTMCells[i] = new LSTMCell();
            }
        }

        public ILayer CreateLayerSharedWegiths()
        {
            LSTMLayer layer = new LSTMLayer(config);
            ShallowCopyWeightTo(layer);

            return layer;
        }

        public Neuron CopyNeuronTo(Neuron neuron)
        {
            LSTMNeuron lstmNeuron = neuron as LSTMNeuron;
            Cells.CopyTo(lstmNeuron.Cells, 0);
            for (int i = 0; i < LayerSize; i++)
            {
                lstmNeuron.LSTMCells[i].Set(LSTMCells[i]);
            }

            return lstmNeuron;
        }

        public void SetNeuron(Neuron neuron)
        {
            LSTMNeuron lstmNeuron = neuron as LSTMNeuron;
            for (int i = 0; i < LayerSize; i++)
            {
                LSTMCells[i].Set(lstmNeuron.LSTMCells[i]);
            }

            //  neuron.Errs.CopyTo(Errs, 0);

            Errs = neuron.Errs;
        }

        public void ShallowCopyWeightTo(ILayer destLayer)
        {
            LSTMLayer layer = destLayer as LSTMLayer;
            layer.SparseFeatureSize = SparseFeatureSize;
            layer.DenseFeatureSize = DenseFeatureSize;

            layer.sparseFeatureWeights = sparseFeatureWeights;
            layer.sparseFeatureWeightsDelta = sparseFeatureWeightsDelta;
            layer.sparseFeatureLearningRate = sparseFeatureLearningRate;

            layer.wDenseCellGate = wDenseCellGate.CloneSharedWeights();
            layer.wDenseForgetGate = wDenseForgetGate.CloneSharedWeights();
            layer.wDenseInputGate = wDenseInputGate.CloneSharedWeights();
            layer.wDenseOutputGate = wDenseOutputGate.CloneSharedWeights();

            layer.CellWeights = CellWeights;
            layer.cellDelta = cellDelta;
            layer.peepholeDelta = peepholeDelta;

            layer.InitializeInternalTrainingParameters();
        }

        public void InitializeInternalTrainingParameters()
        {
            if (SparseFeatureSize > 0)
            {
                sparseFeatureToHiddenDeri = new Dictionary<int, Vector3>[LayerSize];
                for (var i = 0; i < LayerSize; i++)
                {
                    sparseFeatureToHiddenDeri[i] = new Dictionary<int, Vector3>();
                }

                if (sparseFeatureLearningRate == null)
                {
                    sparseFeatureLearningRate = new Vector4[LayerSize][];
                    sparseFeatureWeightsDelta = new Vector4[LayerSize][];
                    for (var i = 0; i < LayerSize; i++)
                    {
                        sparseFeatureLearningRate[i] = new Vector4[SparseFeatureSize];
                        sparseFeatureWeightsDelta[i] = new Vector4[SparseFeatureSize];
                    }
                }
            }

            if (DenseFeatureSize > 0)
            {
                wDenseInputGate.InitInternaTrainingParameters();
                wDenseForgetGate.InitInternaTrainingParameters();
                wDenseCellGate.InitInternaTrainingParameters();
                wDenseOutputGate.InitInternaTrainingParameters(false);
            }

            cellDelta = new Vector4[LayerSize];
            peepholeDelta = new Vector3[LayerSize];
            CellWeightsDeri = new LSTMCellWeightDeri[LayerSize];
            for (var i = 0; i < LayerSize; i++)
            {
                CellWeightsDeri[i] = new LSTMCellWeightDeri();
            }

        }


        public void InitializeWeights(int sparseFeatureSize, int denseFeatureSize)
        {
            SparseFeatureSize = sparseFeatureSize;
            DenseFeatureSize = denseFeatureSize;

            InitializeCellWeights(null);

            if (SparseFeatureSize > 0)
            {
                sparseFeatureWeights = new Vector4[LayerSize][];
                for (var i = 0; i < LayerSize; i++)
                {
                    sparseFeatureWeights[i] = new Vector4[SparseFeatureSize];
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
                wDenseInputGate.InitWeights(LayerSize, DenseFeatureSize);
                wDenseForgetGate.InitWeights(LayerSize, DenseFeatureSize);
                wDenseCellGate.InitWeights(LayerSize, DenseFeatureSize);
                wDenseOutputGate.InitWeights(LayerSize, DenseFeatureSize);
            }

            InitializeInternalTrainingParameters();

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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double TanH(double x)
        {
            return Math.Tanh(x);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double TanHDerivative(double x)
        {
            var tmp = Math.Tanh(x);
            return 1 - tmp * tmp;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double SigmoidDerivative(double x)
        {
            var sigmoid = Sigmoid(x);
            return sigmoid * (1.0 - sigmoid);
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
            gateWeight.layerSize = w;
            gateWeight.denseFeatureSize = h;

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

        public void Save(BinaryWriter fo)
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

        public void Load(BinaryReader br, LayerType layerType, bool forTraining = false)
        {
            config = new LSTMLayerConfig();
            config.LayerSize = br.ReadInt32();
            config.LayerType = layerType;

            Cells = new float[LayerSize];
            Errs = new float[LayerSize];

            LSTMCells = new LSTMCell[LayerSize];
            for (var i = 0; i < LayerSize; i++)
            {
                LSTMCells[i] = new LSTMCell();
            }

            SparseFeatureSize = br.ReadInt32();
            DenseFeatureSize = br.ReadInt32();

            //Create cells of each layer
            InitializeCellWeights(br);

            //Load weight matrix between each two layer pairs
            //weight input->hidden
            if (SparseFeatureSize > 0)
            {
                Logger.WriteLine("Loading sparse feature weights...");
                sparseFeatureWeights = LoadLSTMWeights(br);
            }

            if (DenseFeatureSize > 0)
            {
                //weight fea->hidden
                Logger.WriteLine("Loading dense feature weights...");
                wDenseInputGate = LoadLSTMGateWeights(br);
                wDenseCellGate = LoadLSTMGateWeights(br);
                wDenseForgetGate = LoadLSTMGateWeights(br);
                wDenseOutputGate = LoadLSTMGateWeights(br);
            }

            if (forTraining)
            {
                InitializeInternalTrainingParameters();
            }
        }

        public void CleanForTraining()
        {
            if (SparseFeatureSize > 0)
            {
                for (var i = 0; i < LayerSize; i++)
                {
                    Array.Clear(sparseFeatureLearningRate[i], 0, SparseFeatureSize);
                    Array.Clear(sparseFeatureWeightsDelta[i], 0, sparseFeatureWeightsDelta[i].Length);
                }
            }

            if (DenseFeatureSize > 0)
            {
                wDenseCellGate.CleanStatusForTraining();
                wDenseForgetGate.CleanStatusForTraining();
                wDenseInputGate.CleanStatusForTraining();
                wDenseOutputGate.CleanStatusForTraining();
            }

            peepholeLearningRate = new Vector3[LayerSize];
            peepholeDelta = new Vector3[LayerSize];
            cellLearningRate = new Vector4[LayerSize];
            cellDelta = new Vector4[LayerSize];


            for (var i = 0; i < LayerSize; i++)
            {
                peepholeDelta[i] = Vector3.Zero;
                peepholeLearningRate[i] = Vector3.Zero;

                cellDelta[i] = Vector4.Zero;
                cellLearningRate[i] = Vector4.Zero;
            }

            vecNormalLearningRate = new Vector4(RNNHelper.LearningRate, RNNHelper.LearningRate, RNNHelper.LearningRate,
                RNNHelper.LearningRate);
            vecNormalLearningRate3 = new Vector3(RNNHelper.LearningRate, RNNHelper.LearningRate, RNNHelper.LearningRate);

            vecMaxGrad = new Vector4(RNNHelper.GradientCutoff, RNNHelper.GradientCutoff, RNNHelper.GradientCutoff,
                RNNHelper.GradientCutoff);
            vecMinGrad = new Vector4(-RNNHelper.GradientCutoff, -RNNHelper.GradientCutoff, -RNNHelper.GradientCutoff,
                -RNNHelper.GradientCutoff);

            vecMaxGrad3 = new Vector3(RNNHelper.GradientCutoff, RNNHelper.GradientCutoff, RNNHelper.GradientCutoff);
            vecMinGrad3 = new Vector3(-RNNHelper.GradientCutoff, -RNNHelper.GradientCutoff, -RNNHelper.GradientCutoff);
        }

        public void ForwardPass(SparseVector sparseFeature, float[] denseFeature)
        {
            ForwardPass(new List<SparseVector> {sparseFeature }, new List <float[]>{ denseFeature});

        }

        // forward process. output layer consists of tag value
        public void ForwardPass(List<SparseVector> sparseFeatureGroups, List<float[]> denseFeatureGroup)
        {
            //inputs(t) -> hidden(t)
            //Get sparse feature and apply it into hidden layer
            SparseFeatureGroups = sparseFeatureGroups;
            DenseFeatureGroups = denseFeatureGroup;

            for (var j = 0; j < LayerSize; j++)
            {
                var cell_j = LSTMCells[j];
                var cellWeight_j = CellWeights[j];

                //hidden(t-1) -> hidden(t)
                cell_j.previousCellState = cell_j.cellState;
                cell_j.previousCellOutput = Cells[j];

                var vecCell_j = Vector4.Zero;

                if (SparseFeatureSize > 0)
                {
                    //Apply sparse weights
                    var weights = sparseFeatureWeights[j];
                    var deri = sparseFeatureToHiddenDeri[j];

                    foreach (var sparseFeature in SparseFeatureGroups)
                    {
                        foreach (var pair in sparseFeature)
                        {
                            vecCell_j += weights[pair.Key] * pair.Value;
                            if (deri.ContainsKey(pair.Key) == false)
                            {
                                deri.Add(pair.Key, Vector3.Zero);
                            }
                        }
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

                    foreach (var denseFeature in DenseFeatureGroups)
                    {
                        int i = 0;
                        var denseFeatureSize = denseFeature.Length;
                        var moreItems = (denseFeatureSize % Vector<float>.Count);
                        while (i < denseFeatureSize - moreItems)
                        {
                            var vX = new Vector<float>(denseInputGateWeight_j, k);
                            var vY = new Vector<float>(denseForgetGateWeight_j, k);
                            var vZ = new Vector<float>(denseCellGateWeight_j, k);
                            var vW = new Vector<float>(denseOutputGateWeight_j, k);
                            var vFeature = new Vector<float>(denseFeature, i);

                            vecCell_j.X += Vector.Dot(vX, vFeature);
                            vecCell_j.Y += Vector.Dot(vY, vFeature);
                            vecCell_j.Z += Vector.Dot(vZ, vFeature);
                            vecCell_j.W += Vector.Dot(vW, vFeature);

                            k += Vector<float>.Count;
                            i += Vector<float>.Count;
                        }

                        while (i < denseFeatureSize)
                        {
                            vecCell_j.X += denseInputGateWeight_j[k] * denseFeature[i];
                            vecCell_j.Y += denseForgetGateWeight_j[k] * denseFeature[i];
                            vecCell_j.Z += denseCellGateWeight_j[k] * denseFeature[i];
                            vecCell_j.W += denseOutputGateWeight_j[k] * denseFeature[i];
                            k++;
                            i++;
                        }
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
                cell_j.netIn += cell_j.previousCellState * cellWeight_j.wPeepholeIn + cell_j.previousCellOutput * cellWeight_j.wCellIn;
                //squash input
                cell_j.yIn = Sigmoid(cell_j.netIn);

                //include internal connection multiplied by the previous cell state
                cell_j.netForget += cell_j.previousCellState * cellWeight_j.wPeepholeForget +
                                    cell_j.previousCellOutput * cellWeight_j.wCellForget;
                cell_j.yForget = Sigmoid(cell_j.netForget);

                cell_j.netCellState += cell_j.previousCellOutput * cellWeight_j.wCellState;
                cell_j.yCellState = TanH(cell_j.netCellState);

                //cell state is equal to the previous cell state multipled by the forget gate and the cell inputs multiplied by the input gate
                cell_j.cellState = cell_j.yForget * cell_j.previousCellState + cell_j.yIn * cell_j.yCellState;

                ////include the internal connection multiplied by the CURRENT cell state
                cell_j.netOut += cell_j.cellState * cellWeight_j.wPeepholeOut + cell_j.previousCellOutput * cellWeight_j.wCellOut;

                //squash output gate
                cell_j.yOut = Sigmoid(cell_j.netOut);

                Cells[j] = (float)(TanH(cell_j.cellState) * cell_j.yOut);

                LSTMCells[j] = cell_j;
            }
        }


        private void UpdateOutputGateWeights(LSTMGateWeight gateWeight, int i, float err)
        {
            var j = 0;
            float[] learningrate_i = gateWeight.learningRate[i];
            float[] weights_i = gateWeight.weightsDelta[i];

            foreach (var denseFeature in DenseFeatureGroups)
            {
                int k = 0;
                var denseFeatureSize = denseFeature.Length;
                var moreItems = (denseFeatureSize % Vector<float>.Count);
                while (k < denseFeatureSize - moreItems)
                {
                    Vector<float> vecDelta = new Vector<float>(denseFeature, k);
                    vecDelta = vecDelta * err;

                    var w_i = new Vector<float>(weights_i, j);
                    w_i += vecDelta;
                    w_i.CopyTo(weights_i, j);

                    j += Vector<float>.Count;
                    k += Vector<float>.Count;
                }

                while (k < denseFeatureSize)
                {
                    float delta = denseFeature[k] * err;
                    weights_i[j] += delta;

                    j++;
                    k++;
                }
            }
        }

        public void ComputeLayerErr(ILayer prevLayer)
        {
            ComputeLayerErr(prevLayer.Errs);
        }

        public void ComputeLayerErr(List<float[]> destErrsList, bool cleanDest = true)
        {
            if (cleanDest == true)
            {
                foreach (float[] destErr in destErrsList)
                {
                    Array.Clear(destErr, 0, destErr.Length);
                }
            }


            int height = wDenseOutputGate.weights.Length;
            for (var k = 0; k < height; k++)
            {
                int i = 0;
                float[] weights = wDenseOutputGate.weights[k];
                float err = Errs[k];

                foreach (float[] destErr in destErrsList)
                {
                    int j = 0;
                    var moreItems = (destErr.Length % Vector<float>.Count);
                    while (j < destErr.Length - moreItems)
                    {
                        Vector<float> vecWeights = new Vector<float>(weights, i);
                        Vector<float> vecErrs = new Vector<float>(destErr, j);
                        vecErrs += err * vecWeights;

                        vecErrs.CopyTo(destErr, j);
                        i += Vector<float>.Count;
                        j += Vector<float>.Count;
                    }

                    while (j < destErr.Length)
                    {
                        destErr[j] += err * weights[i];
                        i++;
                        j++;
                    }
                }
            }
        }


        public void ComputeLayerErr(float[] destErr, bool cleanDest = true)
        {
            if (cleanDest == true)
            {
                Array.Clear(destErr, 0, destErr.Length);
            }
            int height = wDenseOutputGate.weights.Length;
            int weight = wDenseOutputGate.weights[0].Length;
            for (var k = 0; k < height; k++)
            {
                int i = 0;
                float[] weights = wDenseOutputGate.weights[k];
                float err = Errs[k];

                var moreItems = (weight % Vector<float>.Count);
                while (i < weight - moreItems)
                {
                    Vector<float> vecWeights = new Vector<float>(weights, i);
                    Vector<float> vecErrs = new Vector<float>(destErr, i);
                    vecErrs += err * vecWeights;

                    vecErrs.CopyTo(destErr, i);
                    i += Vector<float>.Count;
                }

                while (i < weight)
                {
                    destErr[i] += err * weights[i];
                    i++;
                }
            }
        }

        private void UpdateGateWeights(LSTMGateWeight gateWeight, int i, float featureDerivate, float c_yForget, float err)
        {
            var j = 0;
            float[] deri_i = gateWeight.deri[i];
            float[] learningrate_i = gateWeight.learningRate[i];
            float[] weights_i = gateWeight.weightsDelta[i];

            foreach (var denseFeature in DenseFeatureGroups)
            {
                int k = 0;
                var denseFeatureSize = denseFeature.Length;
                var moreItems = (denseFeatureSize % Vector<float>.Count);
                while (k < denseFeatureSize - moreItems)
                {
                    var feature = new Vector<float>(denseFeature, k);
                    var wd = feature * featureDerivate;
                    var wd_i = new Vector<float>(deri_i, j);
                    wd += wd_i * c_yForget;
                    wd.CopyTo(deri_i, j);

                    Vector<float> vecDelta = wd * err;

                    var w_i = new Vector<float>(weights_i, j);
                    w_i += vecDelta;
                    w_i.CopyTo(weights_i, j);

                    j += Vector<float>.Count;
                    k += Vector<float>.Count;
                }

                while (k < denseFeatureSize)
                {
                    var wd = denseFeature[k] * featureDerivate;
                    wd += deri_i[j] * c_yForget;
                    deri_i[j] = wd;
                    weights_i[j] += wd * err;

                    j++;
                    k++;
                }
            }
        }

        public void BackwardPass()
        {
            //put variables for derivaties in weight class and cell class

            for (var i = 0; i < LayerSize; i++)
            {
                var c = LSTMCells[i];
                var cellWeight = CellWeights[i];
                var cellWeightDeri = CellWeightsDeri[i];

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
                    var w_i = sparseFeatureWeightsDelta[i];
                    var wd_i = sparseFeatureToHiddenDeri[i];
                    var wlr_i = sparseFeatureLearningRate[i];

                    foreach (var sparseFeature in SparseFeatureGroups)
                    {
                        foreach (var entry in sparseFeature)
                        {
                            var wd = vecDerivate * entry.Value;
                            //Adding historical information
                            wd += wd_i[entry.Key] * c_yForget;
                            wd_i[entry.Key] = wd;

                            //Computing final err delta
                            var vecDelta = new Vector4(wd, entry.Value);
                            w_i[entry.Key] += vecErr * vecDelta;
                        }
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
                cellWeightDeri.dSWPeepholeIn = cellWeightDeri.dSWPeepholeIn * c.yForget +
                                  Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn * c.previousCellState;

                //partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
                cellWeightDeri.dSWPeepholeForget = cellWeightDeri.dSWPeepholeForget * c.yForget +
                                      ci_previousCellState_mul_SigmoidDerivative_ci_netForget * c.previousCellState;

                //update internal weights
                var vecCellDelta = new Vector3((float)cellWeightDeri.dSWPeepholeIn, (float)cellWeightDeri.dSWPeepholeForget, (float)c.cellState);
                var vecErr3 = new Vector3(cellStateError, cellStateError, gradientOutputGate);
                peepholeDelta[i] += vecErr3 * vecCellDelta;

                //Update cells weights
                //partial derivatives for internal connections
                cellWeightDeri.dSWCellIn = cellWeightDeri.dSWCellIn * c.yForget +
                              Sigmoid2_ci_netCellState_mul_SigmoidDerivative_ci_netIn * c.previousCellOutput;

                //partial derivatives for internal connections, initially zero as dS is zero and previous cell state is zero
                cellWeightDeri.dSWCellForget = cellWeightDeri.dSWCellForget * c.yForget +
                                  ci_previousCellState_mul_SigmoidDerivative_ci_netForget * c.previousCellOutput;

                cellWeightDeri.dSWCellState = cellWeightDeri.dSWCellState * c.yForget +
                                 Sigmoid2Derivative_ci_netCellState_mul_ci_yIn * c.previousCellOutput;

                var vecCellDelta4 = new Vector4((float)cellWeightDeri.dSWCellIn, (float)cellWeightDeri.dSWCellForget, (float)cellWeightDeri.dSWCellState, (float)c.previousCellOutput);
                cellDelta[i] += vecErr * vecCellDelta4;

                LSTMCells[i] = c;
            }
        }
        public void UpdateWeights()
        {
            wDenseInputGate.UpdateWeights();
            wDenseForgetGate.UpdateWeights();
            wDenseCellGate.UpdateWeights();
            wDenseOutputGate.UpdateWeights();

            for (var i = 0; i < LayerSize; i++)
            {
                LSTMCellWeight cellWeights_i = CellWeights[i];

                //Normalize cell peephole weights delta
                var vecPeepholeDelta = peepholeDelta[i];
                peepholeDelta[i] = Vector3.Zero;

                vecPeepholeDelta = vecPeepholeDelta / RNNHelper.MiniBatchSize;
                vecPeepholeDelta = Vector3.Clamp(vecPeepholeDelta, vecMinGrad3, vecMaxGrad3);

                //Normalize cell weights delta
                var vecCellDelta = cellDelta[i];
                cellDelta[i] = Vector4.Zero;

                vecCellDelta = vecCellDelta / RNNHelper.MiniBatchSize;
                vecCellDelta = Vector4.Clamp(vecCellDelta, vecMinGrad, vecMaxGrad);

                //Computing actual learning rate
                var vecPeepholeLearningRate = ComputeLearningRate(vecPeepholeDelta, ref peepholeLearningRate[i]);
                vecPeepholeDelta = vecPeepholeLearningRate * vecPeepholeDelta;

                var vecCellLearningRate = ComputeLearningRate(vecCellDelta, ref cellLearningRate[i]);
                vecCellDelta = vecCellLearningRate * vecCellDelta;

                cellWeights_i.wPeepholeIn += vecPeepholeDelta.X;
                cellWeights_i.wPeepholeForget += vecPeepholeDelta.Y;
                cellWeights_i.wPeepholeOut += vecPeepholeDelta.Z;
                cellWeights_i.wCellIn += vecCellDelta.X;
                cellWeights_i.wCellForget += vecCellDelta.Y;
                cellWeights_i.wCellState += vecCellDelta.Z;
                cellWeights_i.wCellOut += vecCellDelta.W;

                //Update weights for sparse features
                if (SparseFeatureSize > 0)
                {
                    var wlr_i = sparseFeatureLearningRate[i];
                    var sparseFeatureWeightsDelta_i = sparseFeatureWeightsDelta[i];
                    var sparseFeatureWeights_i = sparseFeatureWeights[i];
                    for (var j = 0; j < SparseFeatureSize; j++)
                    {
                        if (sparseFeatureWeightsDelta_i[j] != Vector4.Zero)
                        {
                            Vector4 vecDelta = sparseFeatureWeightsDelta_i[j];
                            sparseFeatureWeightsDelta_i[j] = Vector4.Zero;

                            vecDelta = vecDelta / RNNHelper.MiniBatchSize;

                            vecDelta = Vector4.Clamp(vecDelta, vecMinGrad, vecMaxGrad);

                            var vecLearningRate = ComputeLearningRate(vecDelta, ref wlr_i[j]);
                            sparseFeatureWeights_i[j] += vecDelta * vecLearningRate;
                        }
                    }
                }
            }
        }

        public void Reset()
        {
            for (var i = 0; i < LayerSize; i++)
            {
                Cells[i] = 0;
                InitializeLSTMCell(LSTMCells[i], CellWeights[i], CellWeightsDeri[i]);

                if (SparseFeatureSize > 0)
                {
                    sparseFeatureToHiddenDeri[i].Clear();
                }
            }
        }

        protected RunningMode runningMode;
        public void SetRunningMode(RunningMode mode)
        {
            runningMode = mode;
        }

        private void InitializeLSTMCell(LSTMCell c, LSTMCellWeight cw, LSTMCellWeightDeri deri)
        {
            c.cellState = 0;

            //partial derivatives
            deri.dSWPeepholeIn = 0;
            deri.dSWPeepholeForget = 0;

            deri.dSWCellIn = 0;
            deri.dSWCellForget = 0;
            deri.dSWCellState = 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector3 ComputeLearningRate(Vector3 vecDelta, ref Vector3 vecWeightLearningRate)
        {
            if (RNNHelper.IsConstAlpha)
            {
                return vecNormalLearningRate3;
            }

            vecWeightLearningRate += vecDelta * vecDelta;
            return vecNormalLearningRate3 / (Vector3.SquareRoot(vecWeightLearningRate) + Vector3.One);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector4 ComputeLearningRate(Vector4 vecDelta, ref Vector4 vecWeightLearningRate)
        {
            if (RNNHelper.IsConstAlpha)
            {
                return vecNormalLearningRate;
            }

            vecWeightLearningRate += vecDelta * vecDelta;
            return vecNormalLearningRate / (Vector4.SquareRoot(vecWeightLearningRate) + Vector4.One);
        }
    }

    public class LSTMCellWeightDeri
    {
        public double dSWCellForget;
        public double dSWCellIn;
        public double dSWCellState;
        public double dSWPeepholeForget;
        public double dSWPeepholeIn;
    }


    public class LSTMCellWeight
    {
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
        public double previousCellOutput;
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
            previousCellOutput = cell.previousCellOutput;
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