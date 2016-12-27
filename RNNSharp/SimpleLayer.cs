using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class SimpleLayer
    {
        public float[] cellOutput;

        //Dense feature set
        public float[] DenseFeature;

        public float[] er;

        public List<int> LabelShortList;

        protected ParallelOptions parallelOption = new ParallelOptions();
        public float[] previousCellOutput;

        //Sparse feature set
        public SparseVector SparseFeature;

        public SimpleLayer(LayerConfig config)
        {
            LayerConfig = config;
            AllocateMemoryForCells();
        }

        public SimpleLayer()
        {
            LayerConfig = new LayerConfig();
        }

        public int LayerSize
        {
            get { return LayerConfig.LayerSize; }
            set { LayerConfig.LayerSize = value; }
        }

        public LayerConfig LayerConfig { get; set; }
        public Matrix<float> SparseWeights { get; set; }
        protected Matrix<float> SparseWeightsDelta { get; set; }
        public Matrix<float> SparseWeightsLearningRate { get; set; }

        public Matrix<float> DenseWeights { get; set; }
        protected Matrix<float> DenseWeightsDelta { get; set; }
        public Matrix<float> DenseWeightsLearningRate { get; set; }
        public virtual int SparseFeatureSize { get; set; }
        public virtual int DenseFeatureSize { get; set; }

        public void AllocateMemoryForCells()
        {
            cellOutput = new float[LayerSize];
            previousCellOutput = new float[LayerSize];
            er = new float[LayerSize];
        }

        public SimpleLayer CloneHiddenLayer()
        {
            var m = new SimpleLayer(LayerConfig);

            var j = 0;
            while (j < LayerSize - Vector<float>.Count)
            {
                var vCellOutput = new Vector<float>(cellOutput, j);
                vCellOutput.CopyTo(m.cellOutput, j);
                var vEr = new Vector<float>(er, j);
                vEr.CopyTo(m.er, j);

                j += Vector<float>.Count;
            }

            while (j < LayerSize)
            {
                m.cellOutput[j] = cellOutput[j];
                m.er[j] = er[j];
                j++;
            }

            return m;
        }

        public virtual void InitializeWeights(int sparseFeatureSize, int denseFeatureSize)
        {
            if (denseFeatureSize > 0)
            {
                Logger.WriteLine("Initializing dense feature matrix. layer size = {0}, feature size = {1}", LayerSize,
                    denseFeatureSize);
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
                Logger.WriteLine("Initializing sparse feature matrix. layer size = {0}, feature size = {1}", LayerSize,
                    sparseFeatureSize);
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

        public virtual void Load(BinaryReader br)
        {
            //Load basic parameters
            LayerSize = br.ReadInt32();
            SparseFeatureSize = br.ReadInt32();
            DenseFeatureSize = br.ReadInt32();

            AllocateMemoryForCells();

            if (SparseFeatureSize > 0)
            {
                Logger.WriteLine("Loading sparse feature weights...");
                SparseWeights = RNNHelper.LoadMatrix(br);
            }

            if (DenseFeatureSize > 0)
            {
                Logger.WriteLine("Loading dense feature weights...");
                DenseWeights = RNNHelper.LoadMatrix(br);
            }
        }

        public virtual void CleanLearningRate()
        {
            SparseWeightsLearningRate = new Matrix<float>(LayerSize, SparseFeatureSize);
            DenseWeightsLearningRate = new Matrix<float>(LayerSize, DenseFeatureSize);
        }

        public virtual void ForwardPass(SparseVector sparseFeature, float[] denseFeature, bool isTrain = true)
        {
            if (DenseFeatureSize > 0)
            {
                DenseFeature = denseFeature;
                RNNHelper.matrixXvectorADD(cellOutput, denseFeature, DenseWeights, LayerSize, DenseFeatureSize);
            }

            if (SparseFeatureSize > 0)
            {
                //Apply sparse features
                SparseFeature = sparseFeature;
                Parallel.For(0, LayerSize, parallelOption, b =>
                {
                    float score = 0;
                    var vector_b = SparseWeights[b];
                    foreach (var pair in SparseFeature)
                    {
                        score += pair.Value * vector_b[pair.Key];
                    }
                    cellOutput[b] += score;
                });
            }
        }

        public virtual void BackwardPass(int numStates, int curState)
        {
            if (DenseFeatureSize > 0)
            {
                //Update hidden-output weights
                Parallel.For(0, LayerSize, parallelOption, c =>
                {
                    var err = er[c];
                    var featureWeightCol = DenseWeights[c];
                    var featureWeightsLearningRateCol = DenseWeightsLearningRate[c];
                    var j = 0;
                    while (j < DenseFeatureSize - Vector<float>.Count)
                    {
                        RNNHelper.UpdateFeatureWeights(DenseFeature, featureWeightCol, featureWeightsLearningRateCol,
                            err, j);
                        j += Vector<float>.Count;
                    }

                    while (j < DenseFeatureSize)
                    {
                        var delta = RNNHelper.NormalizeGradient(err * DenseFeature[j]);
                        var newLearningRate = RNNHelper.UpdateLearningRate(DenseWeightsLearningRate, c, j, delta);
                        featureWeightCol[j] += newLearningRate * delta;
                        j++;
                    }
                });
            }

            if (SparseFeatureSize > 0)
            {
                //Update hidden-output weights
                Parallel.For(0, LayerSize, parallelOption, c =>
                {
                    var er2 = er[c];
                    var vector_c = SparseWeights[c];
                    foreach (var pair in SparseFeature)
                    {
                        var pos = pair.Key;
                        var val = pair.Value;
                        var delta = RNNHelper.NormalizeGradient(er2 * val);
                        var newLearningRate = RNNHelper.UpdateLearningRate(SparseWeightsLearningRate, c, pos, delta);
                        vector_c[pos] += newLearningRate * delta;
                    }
                });
            }
        }

        public virtual void Reset(bool updateNet = false)
        {
        }

        public virtual void ComputeLayerErr(SimpleLayer nextLayer, float[] destErrLayer, float[] srcErrLayer)
        {
            var largeOutputLayer = nextLayer as NCEOutputLayer;
            if (largeOutputLayer != null)
            {
                RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, largeOutputLayer.DenseWeights, LayerSize,
                    largeOutputLayer.negativeSampleWordList);
            }
            else
            {
                //error output->hidden for words from specific class
                RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, nextLayer.DenseWeights, LayerSize,
                    nextLayer.LayerSize);
            }
        }

        public virtual void ComputeLayerErr(SimpleLayer nextLayer)
        {
            var largeOutputLayer = nextLayer as NCEOutputLayer;
            if (largeOutputLayer != null)
            {
                RNNHelper.matrixXvectorADDErr(er, largeOutputLayer.er, largeOutputLayer.DenseWeights, LayerSize,
                    largeOutputLayer.negativeSampleWordList);
            }
            else
            {
                //error output->hidden for words from specific class
                RNNHelper.matrixXvectorADDErr(er, nextLayer.er, nextLayer.DenseWeights, LayerSize, nextLayer.LayerSize);
            }
        }

        public virtual void ComputeLayerErr(Matrix<float> CRFSeqOutput, State state, int timeat)
        {
            if (CRFSeqOutput != null)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                for (var c = 0; c < LayerSize; c++)
                {
                    er[c] = -CRFSeqOutput[timeat][c];
                }
                er[state.Label] = (float)(1.0 - CRFSeqOutput[timeat][state.Label]);
            }
            else
            {
                //For standard RNN
                for (var c = 0; c < LayerSize; c++)
                {
                    er[c] = -cellOutput[c];
                }
                er[state.Label] = (float)(1.0 - cellOutput[state.Label]);
            }
        }

        public virtual void ShallowCopyWeightTo(SimpleLayer destLayer)
        {
            destLayer.DenseWeights = DenseWeights;
            destLayer.DenseWeightsLearningRate = DenseWeightsLearningRate;
            destLayer.DenseFeatureSize = DenseFeatureSize;

            destLayer.SparseWeights = SparseWeights;
            destLayer.SparseWeightsLearningRate = SparseWeightsLearningRate;
            destLayer.SparseFeatureSize = SparseFeatureSize;
        }

        public virtual int GetBestOutputIndex(bool isTrain)
        {
            var imax = 0;
            var dmax = cellOutput[0];
            for (var k = 1; k < LayerSize; k++)
            {
                if (cellOutput[k] > dmax)
                {
                    dmax = cellOutput[k];
                    imax = k;
                }
            }
            return imax;
        }

        public virtual void Softmax(bool isTrain)
        {
            float sum = 0;
            for (var c = 0; c < LayerSize; c++)
            {
                var cell = cellOutput[c];
                if (cell > 50) cell = 50;
                else if (cell < -50) cell = -50;
                var val = (float)Math.Exp(cell);
                sum += val;
                cellOutput[c] = val;
            }
            var i = 0;
            var vecSum = new Vector<float>(sum);
            while (i < LayerSize - Vector<float>.Count)
            {
                var v = new Vector<float>(cellOutput, i);
                v /= vecSum;
                v.CopyTo(cellOutput, i);
                i += Vector<float>.Count;
            }

            while (i < LayerSize)
            {
                cellOutput[i] /= sum;
                i++;
            }
        }
    }
}