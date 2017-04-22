using AdvUtils;
using System;
using System.IO;
using System.Numerics;

namespace RNNSharp
{
    public class DropoutNeuron : Neuron
    {
        public bool[] mask;
    }

    internal class DropoutLayer : SimpleLayer
    {
        private readonly float dropoutRatio;
        private readonly Random rnd;
        private bool[] mask;
        DropoutLayerConfig config;

        public DropoutLayer(DropoutLayerConfig config) : base(config)
        {
            this.config = config;
            dropoutRatio = config.DropoutRatio;
            mask = new bool[LayerSize];
            rnd = new Random();
        }

        public override SimpleLayer CreateLayerSharedWegiths()
        {
            DropoutLayer layer = new DropoutLayer(config);
            ShallowCopyWeightTo(layer);
            layer.InitializeInternalTrainingParameters();

            return layer;
        }

        public override Neuron CopyNeuronTo(Neuron neuron)
        {
            DropoutNeuron dropoutNeuron = neuron as DropoutNeuron;
            mask.CopyTo(dropoutNeuron.mask, 0);
            Cells.CopyTo(dropoutNeuron.Cells, 0);

            return dropoutNeuron;
        }

        public override void PreUpdateWeights(Neuron neuron, float[] errs)
        {
            DropoutNeuron dropoutNeuron = neuron as DropoutNeuron;
            dropoutNeuron.Cells.CopyTo(Cells, 0);
            for (int i = 0; i < LayerSize; i++)
            {
                if (dropoutNeuron.mask[i])
                {
                    Errs[i] = 0;
                }
                else
                {
                    Errs[i] = errs[i];
                }
            }
        }

        public override void InitializeWeights(int sparseFeatureSize, int denseFeatureSize)
        {
            SparseFeatureSize = sparseFeatureSize;
            DenseFeatureSize = denseFeatureSize;
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
                        DenseWeights[i][j] = 1.0f;
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
                        SparseWeights[i][j] = 1.0f;
                    }
                }
            }

            InitializeInternalTrainingParameters();
        }
        public override void ForwardPass(SparseVector sparseFeature, float[] denseFeature)
        {
            if (LayerSize != denseFeature.Length)
            {
                throw new Exception($"The layer size of dropout layer must be equal to its denseFeature size. Layer size = {LayerSize}, Dense feature size = {denseFeature.Length}");
            }

            if (runningMode == RunningMode.Training)
            {
                for (var i = 0; i < LayerSize; i++)
                {
                    var val = (float)rnd.NextDouble();
                    if (val < dropoutRatio)
                    {
                        mask[i] = true;
                        Cells[i] = 0;
                    }
                    else
                    {
                        mask[i] = false;
                        Cells[i] = denseFeature[i];
                    }
                }
            }
            else
            {
                for (var i = 0; i < LayerSize; i++)
                {
                    Cells[i] = (float)(1.0 - dropoutRatio) * denseFeature[i];
                }
            }
        }

        public override void BackwardPass()
        {
        }

        public override void ComputeLayerErr(SimpleLayer nextLayer)
        {
            base.ComputeLayerErr(nextLayer);
            //Apply drop out on error in hidden layer
            for (var i = 0; i < LayerSize; i++)
            {
                if (mask[i])
                {
                    Errs[i] = 0;
                }
            }
        }

        public override void Save(BinaryWriter fo)
        {
            base.Save(fo);
            fo.Write(dropoutRatio);
        }

        public static DropoutLayer Load(BinaryReader br, LayerType layerType)
        {
            DropoutLayer dropoutLayer;
            DropoutLayerConfig config = new DropoutLayerConfig();
            SimpleLayer simpleLayer = SimpleLayer.Load(br, layerType);
            config.DropoutRatio = br.ReadSingle();
            config.LayerSize = simpleLayer.LayerSize;

            dropoutLayer = new DropoutLayer(config);
            dropoutLayer.SparseFeatureSize = simpleLayer.SparseFeatureSize;
            dropoutLayer.DenseFeatureSize = simpleLayer.DenseFeatureSize;

            if (dropoutLayer.SparseFeatureSize > 0)
            {
                dropoutLayer.SparseWeights = simpleLayer.SparseWeights;
            }

            if (dropoutLayer.DenseFeatureSize > 0)
            {
                dropoutLayer.DenseWeights = simpleLayer.DenseWeights;
            }

            return dropoutLayer;
        }
    }
}