using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp
{
    public interface ISequence
    {
        int DenseFeatureSize { get; }
        int SparseFeatureSize { get; }
    }
}
