using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    class DropoutLayer : SimpleLayer
    {
        bool[] mask;
        Random rnd;
        float dropoutRatio;

        public DropoutLayer(DropoutLayerConfig config) : base(config)
        {
            dropoutRatio = config.DropoutRatio;
            rnd = new Random();

        }

        public DropoutLayer()
        {
            rnd = new Random();
        }

        public override void ForwardPass(SparseVector sparseFeature, float[] denseFeature, bool isTrain = true)
        {
            if (LayerSize != denseFeature.Length)
            {
                throw new Exception("The layer size of dropout layer must be equal to its denseFeature size.");
            }

            if (isTrain == true)
            {
                mask = new bool[LayerSize];
                for (int i = 0; i < LayerSize; i++)
                {
                    float val = (float)rnd.NextDouble();
                    if (val < dropoutRatio)
                    {
                        mask[i] = true;
                        cellOutput[i] = 0;
                    }
                    else
                    {
                        mask[i] = false;
                        cellOutput[i] = denseFeature[i];
                    }
                }
            }
            else
            {
                for (int i = 0; i < LayerSize; i++)
                {
                    cellOutput[i] = (float)(1.0 - dropoutRatio) * denseFeature[i];
                }
            }
        }

        public override void BackwardPass(int numStates, int curState)
        {

        }

        public override void ComputeLayerErr(SimpleLayer nextLayer, float[] destErrLayer, float[] srcErrLayer)
        {
            //error output->hidden for words from specific class    	
            RNNHelper.matrixXvectorADDErr(destErrLayer, srcErrLayer, nextLayer.DenseWeights, LayerSize, nextLayer.LayerSize);

            for (int i = 0; i < LayerSize; i++)
            {
                if (mask[i] == true)
                {
                    destErrLayer[i] = 0;
                }
            }
        }

        public override void ComputeLayerErr(SimpleLayer nextLayer)
        {
            //error output->hidden for words from specific class    	
            RNNHelper.matrixXvectorADDErr(er, nextLayer.er, nextLayer.DenseWeights, LayerSize, nextLayer.LayerSize);

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
