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
        public double[] cellOutput;
        public double[] previousCellOutput;
        public double[] er;
        public int LayerSize;

        public Matrix<double> SparseWeights { get; set; }
        protected Matrix<double> SparseWeightsDelta { get; set; }
        public Matrix<double> SparseWeightsLearningRate { get; set; }

        public Matrix<double> DenseWeights { get; set; }
        protected Matrix<double> DenseWeightsDelta { get; set; }
        public Matrix<double> DenseWeightsLearningRate { get; set; }

        protected ParallelOptions parallelOption = new ParallelOptions();
        
        //Sparse feature set
        public SparseVector SparseFeature;
        public virtual int SparseFeatureSize { get; set; }

        //Dense feature set
        public double[] DenseFeature;
        public virtual int DenseFeatureSize { get; set; }

        public int CurrentLabelId;

        public SimpleLayer(int hiddenLayerSize)
        {
            LayerSize = hiddenLayerSize;
            AllocateMemoryForCells();
        }

        public SimpleLayer()
        {

        }

        public void AllocateMemoryForCells()
        {
            cellOutput = new double[LayerSize];
            previousCellOutput = new double[LayerSize];
            er = new double[LayerSize];
        }

        public SimpleLayer GetHiddenLayer()
        {
            SimpleLayer m = new SimpleLayer(LayerSize);
            for (int i = 0; i < LayerSize; i++)
            {
                m.cellOutput[i] = cellOutput[i];
                m.er[i] = er[i];
            }

            return m;
        }

        public virtual void InitializeWeights(int sparseFeatureSize, int denseFeatureSize)
        {
            if (denseFeatureSize > 0)
            {
                Logger.WriteLine("Initializing dense feature matrix. layer size = {0}, feature size = {1}", LayerSize, denseFeatureSize);
                DenseFeatureSize = denseFeatureSize;
                DenseWeights = new Matrix<double>(LayerSize, denseFeatureSize);
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
                SparseWeights = new Matrix<double>(LayerSize, SparseFeatureSize);
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

            if (SparseFeatureSize > 0)
            {
                Logger.WriteLine("Saving input2hidden weights...");
                RNNHelper.SaveMatrix(SparseWeights, fo);
            }

            if (DenseFeatureSize > 0)
            {
                //weight fea->hidden
                Logger.WriteLine("Saving feature2hidden weights...");
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
                Logger.WriteLine("Loading input2hidden weights...");
                SparseWeights = RNNHelper.LoadMatrix(br);
            }

            if (DenseFeatureSize > 0)
            {
                Logger.WriteLine("Loading feature2hidden weights...");
                DenseWeights = RNNHelper.LoadMatrix(br);
            }
        }

        public virtual void CleanLearningRate()
        {
            SparseWeightsLearningRate = new Matrix<double>(LayerSize, SparseFeatureSize);
            DenseWeightsLearningRate = new Matrix<double>(LayerSize, DenseFeatureSize);
        }

        public virtual void computeLayer(SparseVector sparseFeature, double[] denseFeature, bool isTrain = true)
        {
            if (SparseFeatureSize > 0)
            {
                //Apply sparse features
                SparseFeature = sparseFeature;
                Parallel.For(0, LayerSize, parallelOption, b =>
                {
                    double score = 0;
                    double[] vector_b = SparseWeights[b];
                    for (int i = 0; i < SparseFeature.Count; i++)
                    {
                        var entry = SparseFeature.GetEntry(i);
                        score += entry.Value * vector_b[entry.Key];
                    }
                    cellOutput[b] += score;
                });
            }

            if (DenseFeatureSize > 0)
            {
                DenseFeature = denseFeature;
                RNNHelper.matrixXvectorADD(cellOutput, denseFeature, DenseWeights, LayerSize, DenseFeatureSize);
            }
        }

        public virtual void LearnFeatureWeights(int numStates, int curState)
        {
            if (DenseFeatureSize > 0)
            {
                //Update hidden-output weights
                Parallel.For(0, LayerSize, parallelOption, c =>
                {
                    double er2 = er[c];
                    double[] vector_c = DenseWeights[c];
                    for (int a = 0; a < DenseFeatureSize; a++)
                    {
                        double delta = RNNHelper.NormalizeGradient(er2 * DenseFeature[a]);
                        double newLearningRate = RNNHelper.UpdateLearningRate(DenseWeightsLearningRate, c, a, delta);
                        vector_c[a] += newLearningRate * delta;
                    }
                });
            }


            if (SparseFeatureSize > 0)
            {
                //Update hidden-output weights
                Parallel.For(0, LayerSize, parallelOption, c =>
                {
                    double er2 = er[c];
                    double[] vector_c = SparseWeights[c];
                    for (int a = 0; a < SparseFeature.Count; a++)
                    {
                        int pos = SparseFeature.GetEntry(a).Key;
                        double val = SparseFeature.GetEntry(a).Value;
                        double delta = RNNHelper.NormalizeGradient(er2 * val);
                        double newLearningRate = RNNHelper.UpdateLearningRate(SparseWeightsLearningRate, c, pos, delta);
                        vector_c[pos] += newLearningRate * delta;
                    }
                });

            }
        }

        public virtual void netReset(bool updateNet = false) { }

        public virtual void ComputeLayerErr(SimpleLayer nextLayer, double[] destErrLayer, double[] srcErrLayer)
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

        public virtual void ComputeLayerErr(Matrix<double> CRFSeqOutput, State state, int timeat)
        {
            if (CRFSeqOutput != null)
            {
                //For RNN-CRF, use joint probability of output layer nodes and transition between contigous nodes
                for (int c = 0; c < LayerSize; c++)
                {
                    er[c] = -CRFSeqOutput[timeat][c];
                }
                er[state.Label] = 1.0 - CRFSeqOutput[timeat][state.Label];
            }
            else
            {
                //For standard RNN
                for (int c = 0; c < LayerSize; c++)
                {
                    er[c] = -cellOutput[c];
                }
                er[state.Label] = 1.0 - cellOutput[state.Label];
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
            double dmax = cellOutput[0];
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
            double sum = 0;
            for (int c = 0; c < LayerSize; c++)
            {
                double cell = cellOutput[c];
                if (cell > 50) cell = 50;
                if (cell < -50) cell = -50;
                double val = Math.Exp(cell);
                sum += val;
                cellOutput[c] = val;
            }
            int i = 0;
            Vector<double> vecSum = new Vector<double>(sum);
            while (i < LayerSize - Vector<double>.Count)
            {
                Vector<double> v = new Vector<double>(cellOutput, i);
                v /= vecSum;
                v.CopyTo(cellOutput, i);
                i += Vector<double>.Count;
            }

            while (i < LayerSize)
            {
                cellOutput[i] /= sum;
                i++;
            }
        }
    }
}
