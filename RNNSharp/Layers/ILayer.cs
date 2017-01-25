using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp.Layers
{
    interface ILayer
    {
        float[] Cell { get; set; }
        float[] Err { get; set; }

        void ForwardPass(SparseVector sparseFeature, float[] denseFeature);
        void BackwardPass();
        void ComputeLayerErr(SimpleLayer nextLayer);
        void ComputeLayerErr(SimpleLayer nextLayer, float[] destErrLayer, float[] srcErrLayer);
        void Reset();

        void Load(BinaryReader br);
        void Save(BinaryWriter fo);
    }
}
