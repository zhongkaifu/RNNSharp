using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;

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
            if (denseFeatureSize > 0)
            {
                Logger.WriteLine("Initializing dense feature matrix. layer size = {0}, feature size = {1}", LayerSize, denseFeatureSize);
                DenseFeatureSize = denseFeatureSize;
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
                SparseFeatureSize = sparseFeatureSize;
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

        public override void UpdateWeights()
        {

        }

        public override void ForwardPass(List<SparseVector> sparseFeatureGroups, List<float[]> denseFeatureGroups)
        {
            if (runningMode == RunningMode.Training)
            {
                int i = 0;
                foreach (var denseFeature in denseFeatureGroups)
                {
                    for (var j = 0; j < denseFeature.Length; j++)
                    {
                        var r = (float)rnd.NextDouble();
                        if (r < dropoutRatio)
                        {
                            mask[i] = true;
                            Cells[i] = 0;
                        }
                        else
                        {
                            mask[i] = false;
                            Cells[i] = denseFeature[j];
                        }
                        i++;
                    }
                }
            }
            else
            {
                int i = 0;
                foreach (var denseFeature in denseFeatureGroups)
                {
                    for (var j = 0; j < denseFeature.Length; j++)
                    {
                        Cells[i] = (float)(1.0 - dropoutRatio) * denseFeature[j];
                        i++;
                    }
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