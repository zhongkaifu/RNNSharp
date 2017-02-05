using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    class SoftmaxLayer : SimpleLayer
    {
        public SoftmaxLayer(SoftmaxLayerConfig config) : base(config)
        {
        }

        public override void ForwardPass(SparseVector sparseFeature, float[] denseFeature)
        {
            base.ForwardPass(sparseFeature, denseFeature);

            //Softmax
            float sum = 0;
            for (var c = 0; c < LayerSize; c++)
            {
                var cell = Cells[c];
                if (cell > 50) cell = 50;
                else if (cell < -50) cell = -50;
                var val = (float)Math.Exp(cell);
                sum += val;
                Cells[c] = val;
            }
            var i = 0;
            var vecSum = new Vector<float>(sum);
            while (i < LayerSize - Vector<float>.Count)
            {
                var v = new Vector<float>(Cells, i);
                v /= vecSum;
                v.CopyTo(Cells, i);
                i += Vector<float>.Count;
            }

            while (i < LayerSize)
            {
                Cells[i] /= sum;
                i++;
            }
        }
    }
}
