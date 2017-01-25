using AdvUtils;
using System;

namespace RNNSharp
{
    internal class DropoutLayer : SimpleLayer
    {
        private readonly float dropoutRatio;
        private readonly Random rnd;
        private bool[] mask;

        public DropoutLayer(DropoutLayerConfig config) : base(config)
        {
            dropoutRatio = config.DropoutRatio;
            rnd = new Random();
        }

        public DropoutLayer()
        {
            rnd = new Random();
        }

        public override void ForwardPass(SparseVector sparseFeature, float[] denseFeature)
        {
            if (LayerSize != denseFeature.Length)
            {
                throw new Exception("The layer size of dropout layer must be equal to its denseFeature size.");
            }

            if (runningMode == RunningMode.Training)
            {
                mask = new bool[LayerSize];
                for (var i = 0; i < LayerSize; i++)
                {
                    var val = (float)rnd.NextDouble();
                    if (val < dropoutRatio)
                    {
                        mask[i] = true;
                        Cell[i] = 0;
                    }
                    else
                    {
                        mask[i] = false;
                        Cell[i] = denseFeature[i];
                    }
                }
            }
            else
            {
                for (var i = 0; i < LayerSize; i++)
                {
                    Cell[i] = (float)(1.0 - dropoutRatio) * denseFeature[i];
                }
            }
        }

        public override void BackwardPass()
        {
        }

        public override void ComputeLayerErr(SimpleLayer nextLayer, float[] destErrLayer, float[] srcErrLayer)
        {
            //error output->hidden for words from specific class
            RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, nextLayer.DenseWeights, LayerSize,
                nextLayer.LayerSize);

            for (var i = 0; i < LayerSize; i++)
            {
                if (mask[i])
                {
                    destErrLayer[i] = 0;
                }
            }
        }

        public override void ComputeLayerErr(SimpleLayer nextLayer)
        {
            //error output->hidden for words from specific class
            Err = nextLayer.Err;
            DenseWeights = nextLayer.DenseWeights;

            //Apply drop out on error in hidden layer
            for (var i = 0; i < LayerSize; i++)
            {
                if (mask[i])
                {
                    Err[i] = 0;
                }
            }
        }
    }
}