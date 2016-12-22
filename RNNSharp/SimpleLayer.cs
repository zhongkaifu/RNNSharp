using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public class SimpleLayer
    {
        public float[] cellOutput;
        public float[] previousCellOutput;
        public float[] er;
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

        protected ParallelOptions parallelOption = new ParallelOptions();
        
        //Sparse feature set
        public SparseVector SparseFeature;
        public virtual int SparseFeatureSize { get; set; }

        //Dense feature set
        public float[] DenseFeature;
        public virtual int DenseFeatureSize { get; set; }

        public List<int> LabelShortList;

        public SimpleLayer(LayerConfig config)
        {
            LayerConfig = config;
            AllocateMemoryForCells();
        }

        public SimpleLayer()
        {
            LayerConfig = new LayerConfig();
        }

        public void AllocateMemoryForCells()
        {
            cellOutput = new float[LayerSize];
            previousCellOutput = new float[LayerSize];
            er = new float[LayerSize];
        }

        public SimpleLayer CloneHiddenLayer()
        {
            SimpleLayer m = new SimpleLayer(LayerConfig);

            int j = 0;
            while (j < LayerSize - Vector<float>.Count)
            {
                Vector<float> vCellOutput = new Vector<float>(cellOutput, j);
                vCellOutput.CopyTo(m.cellOutput, j);
                Vector<float> vEr = new Vector<float>(er, j);
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
                Logger.WriteLine("Initializing dense feature matrix. layer size = {0}, feature size = {1}", LayerSize, denseFeatureSize);
                DenseFeatureSize = denseFeatureSize;
                DenseWeights = new Matrix<float>(LayerSize, denseFeatureSize);
                for (int i = 0; i < DenseWeights.Height; i++)
                {
                    for (int j = 0; j < DenseWeights.Width; j++)
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
                for (int i = 0; i < SparseWeights.Height; i++)
                {
                    for (int j = 0; j < SparseWeights.Width; j++)
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

            Logger.WriteLine($"Saving simple layer, size = '{LayerSize}', sparse feature size = '{SparseFeatureSize}', dense feature size = '{DenseFeatureSize}'");

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
                    float[] vector_b = SparseWeights[b];
                    foreach (KeyValuePair<int, float> pair in SparseFeature)
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
                    float err = er[c];
                    float[] featureWeightCol = DenseWeights[c];
                    float[] featureWeightsLearningRateCol = DenseWeightsLearningRate[c];
                    int j = 0;
                    while (j < DenseFeatureSize - Vector<float>.Count)
                    {
                        RNNHelper.UpdateFeatureWeights(DenseFeature, featureWeightCol, featureWeightsLearningRateCol, err, j);
                        j += Vector<float>.Count;
                    }

                    while (j < DenseFeatureSize)
                    {
                        float delta = RNNHelper.NormalizeGradient(err * DenseFeature[j]);
                        float newLearningRate = RNNHelper.UpdateLearningRate(DenseWeightsLearningRate, c, j, delta);
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
                    float er2 = er[c];
                    float[] vector_c = SparseWeights[c];
                    foreach (KeyValuePair<int, float> pair in SparseFeature)
                    {
                        int pos = pair.Key;
                        float val = pair.Value;
                        float delta = RNNHelper.NormalizeGradient(er2 * val);
                        float newLearningRate = RNNHelper.UpdateLearningRate(SparseWeightsLearningRate, c, pos, delta);
                        vector_c[pos] += newLearningRate * delta;
                    }
                });

            }
        }

        public virtual void Reset(bool updateNet = false) { }

        public virtual void ComputeLayerErr(SimpleLayer nextLayer, float[] destErrLayer, float[] srcErrLayer)
        {
            NCEOutputLayer largeOutputLayer = nextLayer as NCEOutputLayer;
            if (largeOutputLayer != null)
            {
                RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, largeOutputLayer.DenseWeights, LayerSize, largeOutputLayer.negativeSampleWordList);
            }
            else
            {
                //error output->hidden for words from specific class    	
                RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, nextLayer.DenseWeights, LayerSize, nextLayer.LayerSize);
            }
        }

        public virtual void ComputeLayerErr(SimpleLayer nextLayer)
        {
            NCEOutputLayer largeOutputLayer = nextLayer as NCEOutputLayer;
            if (largeOutputLayer != null)
            {
                RNNHelper.matrixXvectorADDErr(er, largeOutputLayer.er, largeOutputLayer.DenseWeights, LayerSize, largeOutputLayer.negativeSampleWordList);
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
                for (int c = 0; c < LayerSize; c++)
                {
                    er[c] = -CRFSeqOutput[timeat][c];
                }
                er[state.Label] = (float)(1.0 - CRFSeqOutput[timeat][state.Label]);
            }
            else
            {
                //For standard RNN
                for (int c = 0; c < LayerSize; c++)
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
            int imax = 0;
            float dmax = cellOutput[0];
            for (int k = 1; k < LayerSize; k++)
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
            for (int c = 0; c < LayerSize; c++)
            {
                float cell = cellOutput[c];
                if (cell > 50) cell = 50;
                else if (cell < -50) cell = -50;
                float val = (float)Math.Exp(cell);
                sum += val;
                cellOutput[c] = val;
            }
            int i = 0;
            Vector<float> vecSum = new Vector<float>(sum);
            while (i < LayerSize - Vector<float>.Count)
            {
                Vector<float> v = new Vector<float>(cellOutput, i);
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
