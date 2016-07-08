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
        ModelSetting m_modelSetting;
        Random rnd;

        public DropoutLayer(int hiddenLayerSize, ModelSetting modelSetting) : base(hiddenLayerSize)
        {
            rnd = new Random();
            m_modelSetting = modelSetting;
        }

        public DropoutLayer()
        {
            rnd = new Random();
        }

        public override void computeLayer(SparseVector sparseFeature, double[] denseFeature, bool isTrain = true)
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
                    double val = rnd.NextDouble();
                    if (val < m_modelSetting.Dropout)
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
                    cellOutput[i] = (1.0 - m_modelSetting.Dropout) * denseFeature[i];
                }
            }
        }

        public override void LearnFeatureWeights(int numStates, int curState)
        {

        }

        public override void ComputeLayerErr(SimpleLayer nextLayer, double[] destErrLayer, double[] srcErrLayer)
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
