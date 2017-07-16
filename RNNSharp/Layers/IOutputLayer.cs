using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNNSharp.Layers
{
    public interface IOutputLayer : ILayer
    {
        List<int> LabelShortList { get; set; }
        int GetBestOutputIndex();
        void ComputeOutputLoss(Matrix<float> CRFSeqOutput, State state, int timeat);
    }
}
