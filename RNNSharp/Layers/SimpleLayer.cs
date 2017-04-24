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
        public SparseVector SparseFeature { get; set; }
        public float[] DenseFeature { get; set; }

        protected ParallelOptions parallelOption = new ParallelOptions();
        protected RunningMode runningMode;


        protected object[] lockerDenseFeature;
        protected object[] lockerSparseFeature;

        public SimpleLayer(LayerConfig config)
        {
            LayerConfig = config;
            if (LayerSize % Vector<float>.Count != 0)
            {
                LayerSize += (Vector<float>.Count - (LayerSize % Vector<float>.Count));
            }

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

        public Matrix<float> DenseWeights { get; set; }
        public Matrix<float> DenseWeightsLearningRate { get; set; }
        public virtual int SparseFeatureSize { get; set; }
        public virtual int DenseFeatureSize { get; set; }

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
            destLayer.DenseWeightsLearningRate = DenseWeightsLearningRate;
            destLayer.DenseFeatureSize = DenseFeatureSize;

            destLayer.SparseWeights = SparseWeights;
            destLayer.SparseWeightsLearningRate = SparseWeightsLearningRate;
            destLayer.SparseFeatureSize = SparseFeatureSize;

            destLayer.lockerDenseFeature = lockerDenseFeature;
            destLayer.lockerSparseFeature = lockerSparseFeature;

            destLayer.InitializeInternalTrainingParameters();
        }

        public virtual void PreUpdateWeights(Neuron neuron, float[] errs)
        {
            neuron.Cells.CopyTo(Cells, 0);
            errs.CopyTo(Errs, 0);
        }

        public virtual void InitializeInternalTrainingParameters()
        {
            if (DenseFeatureSize > 0 && DenseWeightsLearningRate == null)
            {
                DenseWeightsLearningRate = new Matrix<float>(LayerSize, DenseFeatureSize);
            }

            if (SparseFeatureSize > 0 && SparseWeightsLearningRate == null)
            {
                SparseWeightsLearningRate = new Matrix<float>(LayerSize, SparseFeatureSize);
            }

            if (lockerDenseFeature == null)
            {
                lockerDenseFeature = new object[LayerSize];
                for (int i = 0; i < LayerSize; i++)
                {
                    lockerDenseFeature[i] = new object();
                }
            }

            if (lockerSparseFeature == null)
            {
                lockerSparseFeature = new object[LayerSize];
                for (int i = 0; i < LayerSize; i++)
                {
                    lockerSparseFeature[i] = new object();
                }
            }
        }

        public virtual void InitializeWeights(int sparseFeatureSize, int denseFeatureSize)
        {
            DenseFeatureSize = denseFeatureSize;
            SparseFeatureSize = sparseFeatureSize;

            if (DenseFeatureSize % Vector<float>.Count != 0)
            {
                DenseFeatureSize += (Vector<float>.Count - (DenseFeatureSize % Vector<float>.Count));
            }

            if (denseFeatureSize > 0)
            {
                Logger.WriteLine("Initializing dense feature matrix. layer size = {0}, feature size = {1}", LayerSize, denseFeatureSize);    
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

        public virtual void CleanLearningRate()
        {
            if (SparseWeightsLearningRate != null)
            {
                SparseWeightsLearningRate.Clean();
            }

            if (DenseWeightsLearningRate != null)
            {
                DenseWeightsLearningRate.Clean();
            }
        }

        public virtual void ForwardPass(SparseVector sparseFeature, float[] denseFeature)
        {
            if (DenseFeatureSize > 0)
            {
                DenseFeature = denseFeature;
                RNNHelper.matrixXvectorADD(Cells, denseFeature, DenseWeights, LayerSize, DenseFeatureSize);
            }

            if (SparseFeatureSize > 0)
            {
                //Apply sparse features
                SparseFeature = sparseFeature;
                for (var b = 0;b < LayerSize;b++)
                {
                    float score = 0;
                    var vector_b = SparseWeights[b];
                    foreach (var pair in SparseFeature)
                    {
                        score += pair.Value * vector_b[pair.Key];
                    }
                    Cells[b] += score;
                }
            }
        }

        protected void UpdateFeatureWeights(float[] feature, float[] featureWeight, float[] learningRateWeight,
           float err, int idx, int c)
        {
            //Computing error delta
            var vecDenseFeature = new Vector<float>(feature, idx);
            var vecDelta = vecDenseFeature * err;
            vecDelta = RNNHelper.NormalizeGradient(vecDelta);
            //Computing learning rate
            var vecDenseWeightLearningRateCol = new Vector<float>(learningRateWeight, idx);
            vecDenseWeightLearningRateCol += vecDelta * vecDelta;
            vecDenseWeightLearningRateCol.CopyTo(learningRateWeight, idx);

            var vecNewLearningRate = RNNHelper.vecNormalLearningRate /
                                     (Vector<float>.One + Vector.SquareRoot(vecDenseWeightLearningRateCol));

            lock (lockerDenseFeature[c])
            {
                var vecVector_C = new Vector<float>(featureWeight, idx);
                vecVector_C += vecNewLearningRate * vecDelta;
                vecVector_C.CopyTo(featureWeight, idx);
            }
        }

        public virtual void BackwardPass()
        {
            if (DenseFeatureSize > 0)
            {
                //Update hidden-output weights
                for (var c = 0; c < LayerSize; c++)
                {
                    var err = Errs[c];
                    var featureWeightCol = DenseWeights[c];
                    var featureWeightsLearningRateCol = DenseWeightsLearningRate[c];
                    var j = 0;
                    while (j < DenseFeatureSize)
                    {
                        UpdateFeatureWeights(DenseFeature, featureWeightCol, featureWeightsLearningRateCol,
                            err, j, c);
                        j += Vector<float>.Count;
                    }
                }
            }

            if (SparseFeatureSize > 0)
            {
                //Update hidden-output weights
                for (var c = 0; c < LayerSize; c++)
                {
                    var er2 = Errs[c];
                    var vector_c = SparseWeights[c];
                    foreach (var pair in SparseFeature)
                    {
                        var pos = pair.Key;
                        var val = pair.Value;
                        var delta = RNNHelper.NormalizeGradient(er2 * val);
                        var newLearningRate = RNNHelper.UpdateLearningRate(SparseWeightsLearningRate, c, pos, delta);

                        vector_c[pos] += newLearningRate * delta;

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
                    for (var k = 0; k < nextLayer.LayerSize; k++)
                    {
                        int i = 0;
                        float[] weights = lstmLayer.wDenseOutputGate.weights[k];
                        float err = srcErrLayer[k];
                        while (i < LayerSize)
                        {
                            Vector<float> vecWeights = new Vector<float>(weights, i);
                            Vector<float> vecErrs = new Vector<float>(destErrLayer, i);

                            if (k == 0)
                            {
                                vecErrs = err * vecWeights;
                            }
                            else
                            {
                                vecErrs = vecErrs + err * vecWeights;
                            }

                            if (k == nextLayer.LayerSize - 1)
                            {
                                vecErrs = RNNHelper.NormalizeGradient(vecErrs);
                            }

                            vecErrs.CopyTo(destErrLayer, i);
                            i += Vector<float>.Count;
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