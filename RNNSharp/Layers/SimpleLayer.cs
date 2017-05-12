using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class Neuron
    {
        public float[] Cells { get; set; }
        public float[] PrevCellOutputs { get; set; }
    }

    public class SimpleLayer
    {
        public float[] Cells { get; set; }
        public float[] Errs { get; set; }

        public List<int> LabelShortList { get; set; }
        public List<SparseVector> SparseFeatureGroups { get; set; }
        public List<float[]> DenseFeatureGroups { get; set; }

        protected ParallelOptions parallelOption = new ParallelOptions();
        protected RunningMode runningMode;


        public SimpleLayer(LayerConfig config)
        {
            LayerConfig = config;
            Cells = new float[LayerSize];
            Errs = new float[LayerSize];
            LabelShortList = new List<int>();
        }

        public int LayerSize
        {
            get { return LayerConfig.LayerSize; }
            set { LayerConfig.LayerSize = value; }
        }

        public LayerType LayerType
        {
            get { return LayerConfig.LayerType; }
            set { LayerConfig.LayerType = value; }
        }

        public LayerConfig LayerConfig { get; set; }
        public Matrix<float> SparseWeights { get; set; }
        public Matrix<float> SparseWeightsLearningRate { get; set; }

        public Matrix<float> SparseWeightsDelta { get; set; }

        public Matrix<float> DenseWeights { get; set; }
        public Matrix<float> DenseWeightsLearningRate { get; set; }
        public virtual int SparseFeatureSize { get; set; }
        public virtual int DenseFeatureSize { get; set; }


        public Matrix<float> DenseWeightsDelta { get; set; }

        public void SetRunningMode(RunningMode mode)
        {
            runningMode = mode;
        }

        public virtual Neuron CopyNeuronTo(Neuron neuron)
        {
            Cells.CopyTo(neuron.Cells, 0);

            return neuron;
        }

        public virtual SimpleLayer CreateLayerSharedWegiths()
        {
            SimpleLayer layer = new SimpleLayer(LayerConfig);
            ShallowCopyWeightTo(layer);
            return layer;
        }

        public virtual void ShallowCopyWeightTo(SimpleLayer destLayer)
        {
            destLayer.DenseWeights = DenseWeights;
            destLayer.DenseWeightsDelta = DenseWeightsDelta;
            destLayer.DenseWeightsLearningRate = DenseWeightsLearningRate;
            destLayer.DenseFeatureSize = DenseFeatureSize;

            destLayer.SparseWeights = SparseWeights;
            destLayer.SparseWeightsDelta = SparseWeightsDelta;
            destLayer.SparseWeightsLearningRate = SparseWeightsLearningRate;
            destLayer.SparseFeatureSize = SparseFeatureSize;

            destLayer.InitializeInternalTrainingParameters();
        }

        public virtual void PreUpdateWeights(Neuron neuron, float[] errs)
        {
            neuron.Cells.CopyTo(Cells, 0);
            errs.CopyTo(Errs, 0);
        }

        /// <summary>
        /// Initialize parameters only used for training
        /// </summary>
        public virtual void InitializeInternalTrainingParameters()
        {
            if (DenseFeatureSize > 0 && DenseWeightsLearningRate == null)
            {
                DenseWeightsLearningRate = new Matrix<float>(LayerSize, DenseFeatureSize);
                DenseWeightsDelta = new Matrix<float>(LayerSize, DenseFeatureSize);
            }

            if (SparseFeatureSize > 0 && SparseWeightsLearningRate == null)
            {
                SparseWeightsLearningRate = new Matrix<float>(LayerSize, SparseFeatureSize);
                SparseWeightsDelta = new Matrix<float>(LayerSize, SparseFeatureSize);
            }
        }

        public virtual void InitializeWeights(int sparseFeatureSize, int denseFeatureSize)
        {
            if (denseFeatureSize > 0)
            {
                Logger.WriteLine("Initializing dense feature matrix. layer size = {0}, feature size = {1}", LayerSize, denseFeatureSize);
                DenseFeatureSize = denseFeatureSize;
                DenseWeights = new Matrix<float>(LayerSize, denseFeatureSize);
                for (var i = 0; i < DenseWeights.Height; i++)
                {
                    for (var j = 0; j < DenseWeights.Width; j++)
                    {
                        DenseWeights[i][j] = RNNHelper.RandInitWeight();
                    }
                }
            }

            if (sparseFeatureSize > 0)
            {
                Logger.WriteLine("Initializing sparse feature matrix. layer size = {0}, feature size = {1}", LayerSize, sparseFeatureSize);
                SparseFeatureSize = sparseFeatureSize;
                SparseWeights = new Matrix<float>(LayerSize, SparseFeatureSize);
                for (var i = 0; i < SparseWeights.Height; i++)
                {
                    for (var j = 0; j < SparseWeights.Width; j++)
                    {
                        SparseWeights[i][j] = RNNHelper.RandInitWeight();
                    }
                }            
            }

            InitializeInternalTrainingParameters();
        }

        public virtual void Save(BinaryWriter fo)
        {
            fo.Write(LayerSize);
            fo.Write(SparseFeatureSize);
            fo.Write(DenseFeatureSize);

            Logger.WriteLine(
                $"Saving simple layer, size = '{LayerSize}', sparse feature size = '{SparseFeatureSize}', dense feature size = '{DenseFeatureSize}'");

            if (SparseFeatureSize > 0)
            {
                Logger.WriteLine("Saving sparse feature weights...");
                RNNHelper.SaveMatrix(SparseWeights, fo);
            }

            if (DenseFeatureSize > 0)
            {
                //weight fea->hidden
                Logger.WriteLine("Saving dense feature weights...");
                RNNHelper.SaveMatrix(DenseWeights, fo);
            }
        }

        public static SimpleLayer Load(BinaryReader br, LayerType layerType)
        {
            LayerConfig config = new LayerConfig();
            config.LayerSize = br.ReadInt32();
            config.LayerType = layerType;
            SimpleLayer layer = new SimpleLayer(config);

            layer.SparseFeatureSize = br.ReadInt32();
            layer.DenseFeatureSize = br.ReadInt32();

            if (layer.SparseFeatureSize > 0)
            {
                Logger.WriteLine("Loading sparse feature weights...");
                layer.SparseWeights = RNNHelper.LoadMatrix(br);
            }

            if (layer.DenseFeatureSize > 0)
            {
                Logger.WriteLine("Loading dense feature weights...");
                layer.DenseWeights = RNNHelper.LoadMatrix(br);
            }

            return layer;
        }

        public virtual void CleanForTraining()
        {
            if (SparseWeightsLearningRate != null)
            {
                SparseWeightsLearningRate.Clean();
                SparseWeightsDelta.Clean();
            }

            if (DenseWeightsLearningRate != null)
            {
                DenseWeightsLearningRate.Clean();
                DenseWeightsDelta.Clean();
            }
        }


        public virtual void ForwardPass(SparseVector sparseFeature, float[] denseFeature)
        {
            List<SparseVector> sparseFeatureGroups = new List<SparseVector>();
            sparseFeatureGroups.Add(sparseFeature);

            List<float[]> denseFeatureGroups = new List<float[]>();
            denseFeatureGroups.Add(denseFeature);

            ForwardPass(sparseFeatureGroups, denseFeatureGroups);
        }


        public virtual void ForwardPass(List<SparseVector> sparseFeatureGroups, List<float[]> denseFeatureGroups)
        {
            if (DenseFeatureSize > 0)
            {
                DenseFeatureGroups = denseFeatureGroups;
                RNNHelper.matrixXvectorADD(Cells, DenseFeatureGroups, DenseWeights, LayerSize);
            }

            if (SparseFeatureSize > 0)
            {
                //Apply sparse features
                SparseFeatureGroups = sparseFeatureGroups;
                for (var b = 0; b < LayerSize; b++)
                {
                    float score = 0;
                    var vector_b = SparseWeights[b];

                    foreach (var sparseFeature in SparseFeatureGroups)
                    {
                        foreach (var pair in sparseFeature)
                        {
                            score += pair.Value * vector_b[pair.Key];
                        }
                    }
                    Cells[b] += score;
                }
            }
        }

        public virtual void BackwardPass()
        {
            //Update hidden-output weights
            for (var c = 0; c < LayerSize; c++)
            {
                UpdateLayerAt(c);
            }
        }

        protected void UpdateLayerAt(int c)
        {
            var err = Errs[c];
            if (DenseFeatureSize > 0)
            {
                var denseWeightsDelta_c = DenseWeightsDelta[c];
                var j = 0;

                foreach (var denseFeature in DenseFeatureGroups)
                {
                    int k = 0;
                    var denseFeatureSize = denseFeature.Length;
                    var moreItems = (denseFeatureSize % Vector<float>.Count);
                    while (k < denseFeatureSize - moreItems)
                    {
                        var vecDenseFeature = new Vector<float>(denseFeature, k);
                        var vecDelta = vecDenseFeature * err;

                        var vecVector_C = new Vector<float>(denseWeightsDelta_c, j);
                        vecVector_C += vecDelta;
                        vecVector_C.CopyTo(denseWeightsDelta_c, j);

                        j += Vector<float>.Count;
                        k += Vector<float>.Count;
                    }

                    while (k < denseFeatureSize)
                    {
                        denseWeightsDelta_c[j] += err * denseFeature[k];

                        j++;
                        k++;
                    }
                }
            }

            if (SparseFeatureSize > 0)
            {
                var sparseWeightsDelta_c = SparseWeightsDelta[c];

                foreach (var sparseFeature in SparseFeatureGroups)
                {
                    foreach (var pair in sparseFeature)
                    {
                        var pos = pair.Key;
                        var val = pair.Value;
                        sparseWeightsDelta_c[pos] += err * val;
                    }
                }
            }
        }

        public virtual void UpdateWeights()
        {
            Vector<float> vecMiniBatchSize = new Vector<float>(RNNHelper.MiniBatchSize);
            for (var i = 0; i < LayerSize; i++)
            {
                if (SparseFeatureSize > 0)
                {
                    var sparseWeights_i = SparseWeights[i];
                    var sparseDelta_i = SparseWeightsDelta[i];
                    var sparseLearningRate_i = SparseWeightsLearningRate[i];
                    var j = 0;

                    var moreItems = (SparseFeatureSize % Vector<float>.Count);
                    while (j < SparseFeatureSize - moreItems)
                    {
                        Vector<float> vecDelta = new Vector<float>(sparseDelta_i, j);
                        Vector<float>.Zero.CopyTo(sparseDelta_i, j);

                        if (vecDelta != Vector<float>.Zero)
                        {
                            vecDelta = vecDelta / vecMiniBatchSize;
                            vecDelta = RNNHelper.NormalizeGradient(vecDelta);

                            var wlr_i = new Vector<float>(sparseLearningRate_i, j);
                            var vecLearningRate = RNNHelper.ComputeLearningRate(vecDelta, ref wlr_i);
                            wlr_i.CopyTo(sparseLearningRate_i, j);

                            vecDelta = vecLearningRate * vecDelta;
                            Vector<float> vecWeights = new Vector<float>(sparseWeights_i, j);
                            vecWeights += vecDelta;
                            vecWeights.CopyTo(sparseWeights_i, j);

                        }

                        j += Vector<float>.Count;
                    }

                    while (j < SparseFeatureSize)
                    {
                        var delta = sparseDelta_i[j];
                        sparseDelta_i[j] = 0;
                        delta = delta / RNNHelper.MiniBatchSize;
                        delta = RNNHelper.NormalizeGradient(delta);

                        var newLearningRate = RNNHelper.ComputeLearningRate(SparseWeightsLearningRate, i, j, delta);
                        sparseWeights_i[j] += newLearningRate * delta;

                        j++;
                    }

                }

                if (DenseFeatureSize > 0)
                {
                    var denseWeights_i = DenseWeights[i];
                    var denseDelta_i = DenseWeightsDelta[i];
                    var denseLearningRate_i = DenseWeightsLearningRate[i];
                    var j = 0;

                    var moreItems = (DenseFeatureSize % Vector<float>.Count);
                    while (j < DenseFeatureSize - moreItems)
                    {
                        Vector<float> vecDelta = new Vector<float>(denseDelta_i, j);
                        Vector<float>.Zero.CopyTo(denseDelta_i, j);

                        vecDelta = vecDelta / vecMiniBatchSize;

                        vecDelta = RNNHelper.NormalizeGradient(vecDelta);

                        var wlr_i = new Vector<float>(denseLearningRate_i, j);
                        var vecLearningRate = RNNHelper.ComputeLearningRate(vecDelta, ref wlr_i);
                        wlr_i.CopyTo(denseLearningRate_i, j);

                        vecDelta = vecLearningRate * vecDelta;
                        Vector<float> vecWeights = new Vector<float>(denseWeights_i, j);
                        vecWeights += vecDelta;
                        vecWeights.CopyTo(denseWeights_i, j);

                        j += Vector<float>.Count;
                    }

                    while (j < DenseFeatureSize)
                    {
                        var delta = denseDelta_i[j];
                        denseDelta_i[j] = 0;

                        delta = delta / RNNHelper.MiniBatchSize;
                        delta = RNNHelper.NormalizeGradient(delta);

                        var newLearningRate = RNNHelper.ComputeLearningRate(DenseWeightsLearningRate, i, j, delta);
                        denseWeights_i[j] += newLearningRate * delta;

                        j++;
                    }
                }
            }

        }

        public virtual void Reset()
        {
        }

        public virtual void ComputeLayerErr(SimpleLayer nextLayer, float[] destErrLayer, float[] srcErrLayer)
        {
            var sampledSoftmaxLayer = nextLayer as SampledSoftmaxLayer;
            if (sampledSoftmaxLayer != null)
            {
                RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, sampledSoftmaxLayer.DenseWeights, LayerSize,
                    sampledSoftmaxLayer.negativeSampleWordList);
            }
            else
            {
                var lstmLayer = nextLayer as LSTMLayer;
                if (lstmLayer != null)
                {
                    Array.Clear(destErrLayer, 0, destErrLayer.Length);
                    for (var k = 0; k < nextLayer.LayerSize; k++)
                    {
                        int i = 0;
                        float[] weights = lstmLayer.wDenseOutputGate.weights[k];
                        float err = srcErrLayer[k];

                        var moreItems = (LayerSize % Vector<float>.Count);
                        while (i < LayerSize - moreItems)
                        {
                            Vector<float> vecWeights = new Vector<float>(weights, i);
                            Vector<float> vecErrs = new Vector<float>(destErrLayer, i);
                            vecErrs += err * vecWeights;

                            vecErrs.CopyTo(destErrLayer, i);
                            i += Vector<float>.Count;
                        }

                        while (i < LayerSize)
                        {
                            destErrLayer[i] += err * weights[i];
                            i++;
                        }
                    }
                }
                else
                {
                    //error output->hidden for words from specific class
                    RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, nextLayer.DenseWeights, LayerSize,
                        nextLayer.LayerSize);
                }
            }
        }
        public virtual void ComputeLayerErr(SimpleLayer nextLayer)
        {
            ComputeLayerErr(nextLayer, Errs, nextLayer.Errs);
        }

        public virtual void ComputeLayerErr(Matrix<float> CRFSeqOutput, State state, int timeat)
        {
            if (CRFSeqOutput != null)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                for (var c = 0; c < LayerSize; c++)
                {
                    Errs[c] = -CRFSeqOutput[timeat][c];
                }
                Errs[state.Label] = (float)(1.0 - CRFSeqOutput[timeat][state.Label]);
            }
            else
            {
                //For standard RNN
                for (var c = 0; c < LayerSize; c++)
                {
                    Errs[c] = -Cells[c];
                }
                Errs[state.Label] = (float)(1.0 - Cells[state.Label]);
            }
        }

        public virtual int GetBestOutputIndex()
        {
            var imax = 0;
            var dmax = Cells[0];
            for (var k = 1; k < LayerSize; k++)
            {
                if (Cells[k] > dmax)
                {
                    dmax = Cells[k];
                    imax = k;
                }
            }
            return imax;
        }
    }
}