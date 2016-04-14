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
        public bool[] mask;
        public int LayerSize;

        protected Matrix<double> SparseWeights { get; set; }
        protected Matrix<double> SparseWeightsDelta { get; set; }
        protected Matrix<double> SparseWeightsLearningRate { get; set; }

        public Matrix<double> DenseWeights { get; set; }
        protected Matrix<double> DenseWeightsDelta { get; set; }
        public Matrix<double> DenseWeightsLearningRate { get; set; }
        public virtual float Dropout { get; set; }
        protected ParallelOptions parallelOption = new ParallelOptions();
        
        //Sparse feature set
        public SparseVector SparseFeature;
        public virtual int SparseFeatureSize { get; set; }

        //Dense feature set
        public double[] DenseFeature;
        public virtual int DenseFeatureSize { get; set; }

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
            mask = new bool[LayerSize];
        }

        public SimpleLayer GetHiddenLayer()
        {
            SimpleLayer m = new SimpleLayer(LayerSize);
            for (int i = 0; i < LayerSize; i++)
            {
                m.cellOutput[i] = cellOutput[i];
                m.er[i] = er[i];
                m.mask[i] = mask[i];
            }

            return m;
        }

        public virtual void InitializeWeights(int sparseFeatureSize, int denseFeatureSize)
        {
            if (denseFeatureSize > 0)
            {
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
        }

        public virtual void Save(BinaryWriter fo)
        {
            fo.Write(LayerSize);
            fo.Write(DenseFeatureSize);

            RNNHelper.SaveMatrix(DenseWeights, fo);
        }

        public virtual void Load(BinaryReader br)
        {
            //Load basic parameters
            LayerSize = br.ReadInt32();
            DenseFeatureSize = br.ReadInt32();

            AllocateMemoryForCells();

            DenseWeights = RNNHelper.LoadMatrix(br);
        }

        public virtual void CleanLearningRate()
        {
            DenseWeightsLearningRate = new Matrix<double>(LayerSize, DenseFeatureSize);
        }

        public virtual void computeLayer(SparseVector sparseFeature, double[] denseFeature, bool isTrain = true)
        {
            DenseFeature = denseFeature;
            RNNHelper.matrixXvectorADD(cellOutput, denseFeature, DenseWeights, LayerSize, DenseFeatureSize);
        }

        public virtual void LearnFeatureWeights(int numStates, int curState)
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

        public virtual void netReset(bool updateNet = false) { }

        public virtual void ComputeLayerErr(SimpleLayer nextLayer, double[] destErrLayer, double[] srcErrLayer)
        {
            //error output->hidden for words from specific class    	
            RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, nextLayer.DenseWeights, LayerSize, nextLayer.LayerSize);

            if (Dropout > 0)
            {
                //Apply drop out on error in hidden layer
                for (int i = 0; i < LayerSize; i++)
                {
                    if (mask[i] == true)
                    {
                        destErrLayer[i] = 0;
                    }
                }
            }
        }

        public virtual void ComputeLayerErr(SimpleLayer nextLayer)
        {
            //error output->hidden for words from specific class    	
            RNNHelper.matrixXvectorADDErr(er, nextLayer.er, nextLayer.DenseWeights, LayerSize, nextLayer.LayerSize);

            if (Dropout > 0)
            {
                //Apply drop out on error in hidden layer
                for (int i = 0; i < LayerSize; i++)
                {
                    if (mask[i] == true)
                    {
                        er[i] = 0;
                    }
                }
            }
        }
    }
}
